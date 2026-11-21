"""Wires audio capture, the ASR worker process, rendering, NDI sending, config
hot-reload into one running application; the web admin UI (narrator.web)
attaches to it from main.

Concurrency model (see the plan's threading table):
  - audio capture: sounddevice callback thread (inside MicCapture)
  - a feed thread drains MicCapture's local queue into the ASR worker's
    multiprocessing.Queue
  - ASR worker: separate OS process (see narrator.asr.worker), supervised
    here and restarted if it crashes
  - an event-drain thread applies CaptionEvents from the worker to the renderer
  - render + NDI send: fixed-fps thread (narrator.ndi.sender.run_render_send_loop)
  - config watcher: watchdog thread, applied via one atomic swap path
  - web UI (optional, attached by main): one thread running its own asyncio
    loop (narrator.web.server). It only reads controller state and calls the
    operator actions; caption events reach it through `ui_events`, a bounded
    drop-oldest deque, so a slow browser can never stall these threads.
  - setup checks (narrator.health.HealthMonitor): one thread running the
    static checks at start, on config change, on request and periodically;
    the problems it reports are read by status_snapshot().
  - processing-unit usage (narrator.health.usage.UnitUsageMonitor): one
    thread sampling every unit's load at ~1 Hz (plus an nvidia-smi loop
    process while the usage is being read); status_snapshot() reads its
    last result.

Failure policy (user decision 2026-10-17): nothing that goes wrong stops the
app, and the app never changes the unit, backend or model by itself. A part
that fails (ASR worker crash-looping, audio capture or NDI output failing to
start) is left stopped and reported as a problem with fix actions; the
operator's fix (apply_fix) or a config change restarts only that part.
"""
from __future__ import annotations

import collections
import logging
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path

from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.text_settings import TextSettings, asr_config_without_text, text_settings_from_config
from narrator.asr.vocabulary import VOCABULARY_DIRNAME
from narrator.asr.worker import run_asr_worker
from narrator.audio.capture import MicCapture
from narrator.audio.factory import create_capture
from narrator.audio.ndi_capture import NdiAudioCapture
from narrator.captions.blocklist import apply_blocklist, blocklist_from_config
from narrator.config.loader import ConfigWatcher
from narrator.config.schema import RootConfig
from narrator.health import problems as health_problems
from narrator.health.checks import RuntimeState
from narrator.health.monitor import HealthMonitor
from narrator.health.usage import UnitUsageMonitor
from narrator.ndi.sender import NdiCaptionSender, run_render_send_loop
from narrator.render.canvas import CaptionRenderer

logger = logging.getLogger(__name__)

_ASR_RESTART_BACKOFF_SECONDS = 2.0
_ASR_RESTART_BACKOFF_MAX_SECONDS = 30.0


# How long to let an ASR worker shut down cooperatively before escalating.
# Generous because the worker can be inside a multi-second decode, or loading
# a 3GB model, when it is asked to stop -- but bounded, because a worker that
# outlives its replacement holds a whole whisper model resident.
_ASR_GRACEFUL_STOP_SECONDS = 10.0
_ASR_FORCE_STOP_SECONDS = 5.0

# A worker that dies within this many seconds of starting is considered a
# "quick crash" (as opposed to dying after having run fine for a while, e.g.
# because the operator edited the ASR config). After this many quick crashes
# in a row, the supervisor halts ASR: no more respawns, an "ASR stopped"
# problem with fix actions instead. The rest of the app keeps running, and
# the unit is never changed behind the operator's back (a silent CPU
# fallback would hide a broken GPU setup behind a 10x slowdown) -- see
# _supervise_asr_loop.
_QUICK_CRASH_WINDOW_SECONDS = 15.0
_QUICK_CRASHES_BEFORE_HALT = 3

# Caption events kept for the web UI's live feed. A deque with maxlen drops
# the oldest entry on overflow and its append is atomic, so the event-drain
# thread can never block on (or be slowed by) a stalled browser.
_UI_EVENT_BACKLOG = 256


class AppController:
    # Class-level defaults so controllers built with __new__ (the unit tests)
    # still have them; instances shadow them on first assignment.
    asr_restart_count: int = 0  # deliberate restarts (config changes)
    asr_crash_count: int = 0  # worker deaths the supervisor restarted
    asr_last_exit_code: int | None = None
    _captions_paused: bool = False
    _web = None  # narrator.web.server.WebServer, attached by main
    ui_events: "collections.deque | None" = None
    _started_at: float | None = None
    _asr_busy_value = None  # mp.Value("d") the current worker publishes its busy fraction into
    _health: HealthMonitor | None = None  # setup checks (narrator.health)
    _unit_usage: UnitUsageMonitor | None = None  # live load of every processing unit
    # Failure state the setup checks report (see the module docstring):
    _asr_halted: bool = False  # ASR crash-looped; no respawn until a fix/config change
    asr_last_error: str | None = None  # the dying worker's own explanation (ERROR event)
    _audio_error: str | None = None  # audio capture failed to start
    _ndi_error: str | None = None  # NDI output failed to start

    def __init__(self, config_path: Path) -> None:
        # Must precede every mp.Queue/mp.Event creation below: those bind to
        # the current start-method context, and mixing contexts (a fork-
        # created SemLock shared with a spawn-launched Process) raises at
        # spawn time.
        self._set_mp_start_method()
        self._config_path = config_path
        self._config: RootConfig | None = None
        self._config_lock = threading.Lock()

        self._renderer: CaptionRenderer | None = None
        self._sender: NdiCaptionSender | None = None
        self._mic: MicCapture | NdiAudioCapture | None = None  # either source; same surface

        self._audio_queue: mp.Queue = mp.Queue(maxsize=200)
        self._event_queue: mp.Queue = mp.Queue()
        # One stop event per worker *generation*, not one shared for the life
        # of the app: a single reused event cannot express "stop the old
        # worker but not the new one". The previous shared event was cleared
        # by the incoming generation in _spawn_asr_process, which resurrected
        # any outgoing worker that hadn't exited yet -- see
        # _restart_asr_process for the full failure.
        self._asr_stop_event: "mp.synchronize.Event | None" = None
        self._asr_process: mp.Process | None = None
        # Serializes every ASR lifecycle transition. Without it the config
        # watcher thread and the supervisor thread can both decide to spawn a
        # worker for the same death, leaking one of them.
        self._asr_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._watcher: ConfigWatcher | None = None
        self.resolved_asr_backend: str = "unresolved"

        self._ndi_stop_event: threading.Event | None = None
        self._ndi_thread: threading.Thread | None = None

        self._audio_stop_event: threading.Event | None = None
        self._audio_thread: threading.Thread | None = None

        self._stopped = False
        self._stop_lock = threading.Lock()

        self._asr_spawn_time: float = 0.0
        self._consecutive_quick_crashes: int = 0

        # "auto": captions come from the ASR worker, as before. "manual": the
        # operator types captions in the web UI instead -- ASR events other than
        # STATUS are ignored while manual, so a stray partial/final/clear from
        # the (still-running) worker can't clobber manually-entered text. See
        # set_caption_mode/submit_manual_text/clear_captions.
        self._caption_mode: str = "auto"

        # Every CaptionEvent (plus whether it reached the screen), for the web
        # UI's live feed; see _UI_EVENT_BACKLOG.
        self.ui_events = collections.deque(maxlen=_UI_EVENT_BACKLOG)

    @property
    def config(self) -> RootConfig | None:
        """The live config snapshot (swapped atomically on reload)."""
        return self._config

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def renderer(self) -> CaptionRenderer | None:
        return self._renderer

    def attach_web(self, web) -> None:
        """Register the web UI server so config changes reach it and stop()
        shuts it down first (no requests during teardown)."""
        self._web = web

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._watcher = ConfigWatcher(self._config_path, self._on_config_changed)
        self._config = self._watcher.start()
        logging.basicConfig(level=self._config.app.log_level)

        self._renderer = CaptionRenderer(self._config.canvas, self._config.layout, self._config.style)
        self._health = HealthMonitor(lambda: self._config, self.runtime_state)
        # Neither raises: a failure is kept as a problem (see _ndi_error /
        # _audio_error) and the app starts without that part.
        self._start_ndi_stream()
        self._start_audio_capture()

        with self._asr_lock:
            self._spawn_asr_process()
        self._health.start()
        self._unit_usage = UnitUsageMonitor(self._health.env, lambda: self._config)
        self._unit_usage.start()

        self._spawn_thread(self._drain_events_loop)
        self._spawn_thread(self._supervise_asr_loop)

    def _spawn_thread(self, target) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _start_ndi_stream(self) -> None:
        """(Re)creates the NDI sender and its render+send thread against the
        current config. Uses its own stop event (distinct from the app-wide
        `_stop_event`) so a property change (resolution, source name, fps)
        can restart just this stream -- see `_on_config_changed` -- without
        tearing down the rest of the app.
        """
        self._ndi_stop_event = threading.Event()
        try:
            self._sender = NdiCaptionSender(self._config.ndi, self._config.canvas)
        except Exception as exc:
            # No NDI runtime / sender failure: the rest of the app (ASR,
            # web UI preview, manual captions) keeps running.
            logger.error("NDI output could not start: %s", exc)
            self._sender = None
            self._ndi_thread = None
            self._ndi_error = str(exc) or type(exc).__name__
            self._request_health_check()
            return
        self._ndi_error = None
        ndi_stop_event = self._ndi_stop_event
        sender = self._sender
        fps = self._config.ndi.fps
        self._ndi_thread = threading.Thread(
            target=lambda: run_render_send_loop(self._renderer, sender, fps, ndi_stop_event),
            daemon=True,
        )
        self._ndi_thread.start()

    def _restart_ndi_stream(self) -> None:
        logger.info("restarting NDI stream")
        old_stop_event, old_thread, old_sender = self._ndi_stop_event, self._ndi_thread, self._sender
        if old_stop_event is not None:
            old_stop_event.set()
        if old_thread is not None:
            old_thread.join(timeout=5)
        if old_sender is not None:
            old_sender.close()
        self._start_ndi_stream()
        self._request_health_check()

    def _start_audio_capture(self) -> None:
        """(Re)creates mic capture and its feed thread against the current
        config. Uses its own stop event (distinct from the app-wide
        `_stop_event`) so an `audio` config change can restart just capture
        -- see `_on_config_changed` -- without tearing down the rest of the app.
        """
        self._audio_stop_event = threading.Event()
        try:
            # `audio.device` of the form "ndi:<source>" selects NdiAudioCapture.
            mic = create_capture(self._config.audio, own_ndi_name=self._config.ndi.source_name)
            mic.start()
        except Exception as exc:
            # Missing device, wrong channels, PortAudio trouble: no audio
            # input, everything else keeps running; the problem panel offers
            # Retry / another input.
            logger.error("audio input could not start: %s", exc)
            self._mic = None
            self._audio_thread = None
            self._audio_error = str(exc) or type(exc).__name__
            self._request_health_check()
            return
        self._audio_error = None
        self._mic = mic
        audio_stop_event = self._audio_stop_event
        self._audio_thread = threading.Thread(
            target=lambda: self._feed_audio_loop(mic, audio_stop_event),
            daemon=True,
        )
        self._audio_thread.start()

    def _restart_audio_capture(self, rescan: bool = False) -> None:
        """`rescan` (the operator's Retry fix) makes PortAudio list devices
        again first, so a microphone plugged in since startup is found."""
        logger.info("restarting audio capture")
        old_stop_event, old_thread, old_mic = self._audio_stop_event, self._audio_thread, self._mic
        if old_stop_event is not None:
            old_stop_event.set()
        if old_thread is not None:
            old_thread.join(timeout=5)
        if old_mic is not None:
            old_mic.stop()
        if rescan:
            try:
                from narrator.audio.devices import rescan_devices

                rescan_devices()
            except Exception as exc:
                logger.warning("audio device rescan failed: %s", exc)
        self._start_audio_capture()
        self._request_health_check()

    def _drain_audio_queue(self) -> int:
        """Empty the shared audio queue; returns the number of frames dropped.

        Called between ASR worker generations. Frames queued for the outgoing
        worker are stale by the time the successor starts (its decoder window
        is empty); letting them through would make the new generation decode
        a fragment of pre-restart audio and emit a caption for speech the
        operator heard long ago.
        """
        if self._audio_queue is None:  # lifecycle tests / never started
            return 0
        drained = 0
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                return drained
            drained += 1

    def _spawn_asr_process(self) -> None:
        """Start a new ASR worker generation. Caller must hold `_asr_lock`."""
        asr_config = self._config.asr
        stop_event = mp.Event()
        # Per generation, like the stop event: live text-settings updates
        # (vocabulary, snapping, blocklist) for this worker only.
        control_queue = mp.Queue()
        busy_value = mp.Value("d", -1.0)  # decode busy fraction, -1 = not measured yet
        process = mp.Process(
            target=run_asr_worker,
            args=(
                asr_config,
                self._audio_queue,
                self._event_queue,
                stop_event,
                self._config.app.log_level,
                self._text_settings(self._config),
                control_queue,
                busy_value,
            ),
            daemon=True,
        )
        process.start()
        self._asr_process = process
        self._asr_stop_event = stop_event
        self._asr_control_queue = control_queue
        self._asr_busy_value = busy_value
        self._asr_spawn_time = time.monotonic()
        self._asr_halted = False

    @staticmethod
    def _set_mp_start_method() -> None:
        """Force the 'spawn' start method for the ASR worker process.

        The default on Linux is 'fork': the worker inherits the parent's
        memory, including every loaded library and its locks. Forking from a
        process that already runs Vulkan/PortAudio/watchdog threads is a
        known recipe for native-level breakage in the child -- measured here:
        a forked ASR worker SIGSEGVs inside ggml's Vulkan init during any
        post-startup ASR restart (exit -11 within ~2s, three times in a row),
        which the old crash-loop policy then converted into a silent,
        permanent CPU downgrade. 'spawn' starts a fresh interpreter that
        imports the worker module cleanly; model load is no slower in any
        measurable way (the 3GB model load dominates either way), and the
        first spawn (from a process that has only just started) happened to
        work with fork -- spawn makes every generation equally clean.
        """
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set by a previous start() in this interpreter

    @staticmethod
    def _shut_down_asr_process(
        process: mp.Process | None, stop_event: "mp.synchronize.Event | None"
    ) -> None:
        """Stop one ASR worker generation and *guarantee* it is gone.

        A cooperative stop alone is not enough. The worker can be several
        seconds from noticing the event (mid-decode) or tens of seconds away
        (loading a 3GB large-v3 model), so a bare `join(timeout=5)` routinely
        returns with the process still running. Leaving it there is what
        turned a config edit into a resource leak: each survivor keeps a
        whole whisper model resident -- ~3GB of VRAM on the GPU path -- and
        keeps draining the shared audio queue, so a few edits exhausted VRAM,
        pushed subsequent workers onto CPU, and eventually ran the machine
        out of memory. Escalate to terminate/kill rather than orphan it.
        """
        if process is None:
            return
        if stop_event is not None:
            stop_event.set()
        process.join(timeout=_ASR_GRACEFUL_STOP_SECONDS)
        if process.is_alive():
            logger.warning(
                "ASR worker did not exit within %.0fs of being asked to stop; terminating",
                _ASR_GRACEFUL_STOP_SECONDS,
            )
            process.terminate()
            process.join(timeout=_ASR_FORCE_STOP_SECONDS)
        if process.is_alive():
            logger.error("ASR worker ignored SIGTERM; killing")
            process.kill()
            process.join(timeout=_ASR_FORCE_STOP_SECONDS)

    def _restart_asr_process(self) -> None:
        """Replace the running ASR worker with one built from current config.

        Detaching the outgoing process *before* stopping it is deliberate:
        `_supervise_asr_loop` treats "the current worker is not alive" as a
        crash, and a deliberate restart would otherwise be miscounted as one.
        That miscount was not cosmetic -- three of them tripped the
        crash-loop breaker and pinned the app to CPU permanently, and the
        supervisor's own restart raced this one, leaking a worker each time.
        """
        with self._asr_lock:
            old_process, old_stop_event = self._asr_process, self._asr_stop_event
            self._asr_process, self._asr_stop_event = None, None
            # A deliberate restart (config change, the operator's Retry) is
            # a fresh attempt, not a continuation of whatever crash streak
            # preceded it -- and it lifts a halt.
            self._consecutive_quick_crashes = 0
            self._asr_halted = False
            self.asr_last_error = None
            self.resolved_asr_backend = "unresolved"

        self._shut_down_asr_process(old_process, old_stop_event)

        # The outgoing worker can no longer consume these, and the incoming
        # one must not start mid-way through old speech.
        drained = self._drain_audio_queue()
        if drained:
            logger.info("discarded %d stale audio frame(s) from before the ASR restart", drained)

        with self._asr_lock:
            if self._stop_event.is_set():
                return
            self._spawn_asr_process()
            self.asr_restart_count += 1

    def _feed_audio_loop(self, mic: MicCapture | NdiAudioCapture, stop_event: threading.Event) -> None:
        for frame in mic.frames(timeout=0.5, stop_event=stop_event):
            if self._asr_halted:
                # No worker reads the queue while ASR is halted. Feeding it
                # would keep its pipe full, and a full mp.Queue pipe blocks
                # the queue's feeder thread -- and with it interpreter exit.
                continue
            try:
                self._audio_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._audio_queue.put_nowait(frame)
                except queue.Full:
                    pass

    def _drain_events_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                event: CaptionEvent = self._event_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._handle_caption_event(event)

    def _handle_caption_event(self, event: CaptionEvent) -> None:
        # STATUS is always applied so the web UI's backend indicator stays
        # live even while manual mode is ignoring transcription output.
        if event.type == CaptionEventType.STATUS:
            self.resolved_asr_backend = event.text
            self.asr_last_error = None  # this generation loaded fine
            logger.info("ASR backend resolved: %s", event.text)
            self._publish_ui_event(event, applied=True)
            return
        if event.type == CaptionEventType.ERROR:
            # The worker's last words before it exits; the supervisor sees
            # the exit, the problem panel shows this text.
            self.asr_last_error = event.text
            return
        # Paused works like manual mode: the worker keeps running (resuming
        # is instant), its output just doesn't reach the screen.
        applied = self._caption_mode == "auto" and not self._captions_paused
        self._publish_ui_event(event, applied=applied)
        if not applied:
            return
        if event.type == CaptionEventType.FINAL:
            # Replaces the open partial atomically (CaptionRenderer).
            self._renderer.append_final_text(event.text)
        elif event.type == CaptionEventType.PARTIAL:
            self._renderer.set_partial_preview(event.text)
        elif event.type == CaptionEventType.WIPE:
            # True screen clear after a long silence lull (the operator-
            # equivalent path -- same renderer method as clear_captions).
            self._renderer.clear()
        elif event.type == CaptionEventType.CLEAR:
            # The ASR worker emits CLEAR only after
            # prune_history_after_silence_seconds of no speech. That timeout
            # must never wipe what the audience is currently reading: CLEAR
            # only prunes caption lines that are no longer visible under the
            # current layout settings. A real screen wipe is operator-
            # initiated (clear_captions <- the web UI's Clear button/shortcut)
            # or comes in as WIPE after wipe_screen_after_silence_seconds.
            dropped = self._renderer.prune_history()
            if dropped:
                logger.debug("pruned %d off-screen caption line(s) after silence", dropped)

    def _publish_ui_event(self, event: CaptionEvent, applied: bool) -> None:
        events = self.ui_events
        if events is not None:
            events.append((event.type.value, event.text, applied))

    def set_caption_mode(self, mode: str) -> None:
        """Switches between "auto" (ASR-driven) and "manual" (operator-typed)
        captions. The ASR worker keeps running underneath in either case --
        switching back to auto is instant, with no model reload -- but its
        output is only applied to the renderer while auto is selected.
        """
        if mode not in ("auto", "manual"):
            raise ValueError(f"invalid caption mode: {mode!r}")
        self._caption_mode = mode

    @property
    def caption_mode(self) -> str:
        return self._caption_mode

    @property
    def captions_paused(self) -> bool:
        return self._captions_paused

    def set_captions_paused(self, paused: bool) -> None:
        """Pause/resume ASR captions. Like manual mode, the ASR worker keeps
        running and only its output is ignored while paused, so resuming is
        instant; what is on screen stays (Clear is separate). Manual captions
        are unaffected."""
        paused = bool(paused)
        if paused and not self._captions_paused and self._renderer is not None:
            self._renderer.set_partial_preview("")  # no stale partial frozen on air
        self._captions_paused = paused
        logger.info("captions %s", "paused" if paused else "resumed")

    def refresh_text_settings(self) -> None:
        """Re-send the worker's text settings, e.g. after a saved vocabulary
        list file changed (the config itself did not)."""
        if self._config is not None:
            self._push_text_settings(self._config)

    def status_snapshot(self) -> dict:
        """Read-only state for the web UI / control API. Lock-free: every
        field is a single attribute read (the mic is re-read each call,
        since _restart_audio_capture swaps the capture object)."""
        process = self._asr_process
        mic = self._mic
        renderer = self._renderer
        started = self._started_at
        busy_value = self._asr_busy_value
        usage = self._unit_usage
        units = usage.snapshot() if usage is not None else {}
        try:
            on_air = list(renderer.visible_lines()) if renderer is not None else []
        except Exception:  # a concurrent reflow; next poll catches up
            on_air = []
        return {
            "caption_mode": self._caption_mode,
            "captions_paused": self._captions_paused,
            "asr_busy": (
                round(busy_value.value, 3) if busy_value is not None and busy_value.value >= 0 else None
            ),
            "asr_active": process is not None and process.is_alive(),
            "asr_halted": self._asr_halted,
            "asr_backend": self.resolved_asr_backend,
            "asr_restarts": self.asr_restart_count,
            "asr_crashes": self.asr_crash_count,
            "asr_last_exit_code": self.asr_last_exit_code,
            "mic_level": float(mic.level) if mic is not None else 0.0,
            "on_air": on_air,
            "uptime_s": None if started is None else round(time.monotonic() - started, 1),
            "audio_running": mic is not None,
            "ndi_running": self._sender is not None,
            "problems": [p.to_dict() for p in self.problems()],
            # Every processing unit with its live usage (narrator.health.usage).
            "units": units.get("units", []),
            "unit_recommended": units.get("recommended"),
            "unit_resolved": units.get("resolved"),
        }

    # ------------------------------------------------------------ setup checks

    def runtime_state(self) -> RuntimeState:
        """What the setup checks need from the running app (attribute reads
        only; called from the health thread and the web UI's snapshot)."""
        process = self._asr_process
        mic = self._mic
        ndi_source = status = detail = None
        age = 0.0
        if isinstance(mic, NdiAudioCapture):
            ndi_source = mic.source_name
            status, detail = mic.status, mic.status_detail
            age = time.monotonic() - mic.status_since
        return RuntimeState(
            asr_alive=process is not None and process.is_alive(),
            asr_halted=self._asr_halted,
            asr_crash_streak=getattr(self, "_consecutive_quick_crashes", 0),
            asr_last_exit_code=self.asr_last_exit_code,
            asr_last_error=self.asr_last_error,
            audio_running=mic is not None,
            audio_error=self._audio_error,
            ndi_audio_source=ndi_source,
            ndi_audio_status=status,
            ndi_audio_status_age_s=age,
            ndi_audio_detail=detail,
            ndi_output_error=self._ndi_error,
        )

    def problems(self) -> list:
        """Current problems (narrator.health.Problem), most severe first."""
        health = self._health
        return health.problems() if health is not None else []

    def _request_health_check(self, invalidate: bool = False) -> None:
        health = self._health
        if health is not None:
            health.request_check(invalidate=invalidate)

    def apply_fix(self, action: str) -> None:
        """Apply one fix action from the problems panel (the operator's
        click; nothing calls this on its own). Only fixes some current
        problem offers are accepted -- "Use CPU instead" exists only where
        the checks found the CPU backend usable. Restarts only the affected
        part. Blocking (joins threads/processes): call it off the web loop.
        Raises ValueError for an unknown action, LookupError for one no
        problem offers right now."""
        if not health_problems.is_server_fix(action):
            raise ValueError(f"unknown fix action: {action!r}")
        health = self._health
        # Validate against a fresh pass, not the last periodic one: a stale
        # result could still offer "Use CPU" after the build changed.
        current = health.check_now(invalidate=True) if health is not None else []
        if action == health_problems.FIX_RECHECK:
            return
        if action not in health_problems.offered_fixes(current):
            raise LookupError(f"fix {action!r} is not offered by any current problem")
        logger.info("operator fix: %s", action)
        if action == health_problems.FIX_ASR_RETRY:
            self._restart_asr_process()
        elif action == health_problems.FIX_ASR_USE_CPU:
            self._write_asr_unit("cpu", "")
        elif action == health_problems.FIX_ASR_PIN_UNIT:
            unit = self._find_unit(self._config.asr.unit)
            if unit is None:
                raise LookupError(f"asr.unit {self._config.asr.unit} is not on this machine")
            self._write_asr_unit(unit.id, unit.identity)
        elif action.startswith(health_problems.FIX_ASR_UNIT_PREFIX):
            unit = self._find_unit(action.removeprefix(health_problems.FIX_ASR_UNIT_PREFIX))
            if unit is None:
                raise LookupError(f"{action}: that unit is not on this machine")
            self._write_asr_unit(unit.id, unit.identity)
        elif action == health_problems.FIX_AUDIO_RETRY:
            self._restart_audio_capture(rescan=True)
        elif action == health_problems.FIX_NDI_RETRY:
            self._restart_ndi_stream()
        if self._health is not None:
            # In this thread, so the answer to the click already reflects it.
            self._health.check_now(invalidate=True)

    def _find_unit(self, unit_id: str):
        """The processing unit `unit_id` on this machine (the setup checks'
        cached probes), None when absent."""
        from narrator.asr.units import find_unit

        health = self._health
        if health is None:
            return None
        return find_unit(health.env.units(), unit_id)

    def _write_asr_unit(self, unit: str, identity: str) -> None:
        """Change asr.unit (+ its recorded identity) in config.yaml like a
        web UI Save: the config file stays the single source of truth, and
        the watcher applies it (an asr change restarts only the ASR worker).
        Only an operator's fix click gets here."""
        from narrator.config.loader import load_config, save_config

        current = load_config(self._config_path)
        update = {"unit": unit, "unit_identity": identity}
        updated = current.model_copy(update={"asr": current.asr.model_copy(update=update)})
        save_config(RootConfig.model_validate(updated.model_dump(mode="json")), self._config_path)
        logger.info("asr.unit set to %s by the operator (problems panel)", unit)

    def _text_settings(self, config: RootConfig) -> TextSettings:
        """The ASR worker's live text settings; the named vocabulary list is
        read from <config dir>/vocabulary/."""
        config_path = getattr(self, "_config_path", None)
        vocabulary_dir = config_path.parent / VOCABULARY_DIRNAME if config_path is not None else None
        return text_settings_from_config(config, vocabulary_dir)

    def submit_manual_text(self, text: str) -> None:
        """Appends operator-typed text as a finalized caption line. Intended
        for use while caption_mode == "manual", same as an ASR FINAL event.
        The caption blocklist applies here too (a guard against typos).
        """
        config = getattr(self, "_config", None)
        if config is not None:
            text = apply_blocklist(text, blocklist_from_config(config.captions))
        text = text.strip()
        if not text or self._renderer is None:
            return
        self._renderer.set_partial_preview("")
        self._renderer.append_final_text(text)

    def clear_captions(self) -> None:
        """Clears the on-screen caption buffer. Works in either caption mode
        -- backs the web UI's Clear button, its shortcut and the control API.
        """
        if self._renderer is not None:
            self._renderer.clear()

    def _supervise_asr_loop(self) -> None:
        while not self._stop_event.is_set():
            # Snapshot under the lock. `_asr_process` is None precisely while a
            # deliberate restart is in flight (or ASR is halted), which is how
            # an intentional stop is distinguished from a crash without a
            # separate flag.
            with self._asr_lock:
                process = self._asr_process
                spawn_time = self._asr_spawn_time
            if process is not None and not process.is_alive():
                exit_code = process.exitcode
                if not self._stop_event.is_set():
                    self.asr_crash_count += 1
                    self.asr_last_exit_code = exit_code
                    uptime = time.monotonic() - spawn_time
                    if uptime < _QUICK_CRASH_WINDOW_SECONDS:
                        self._consecutive_quick_crashes += 1
                    else:
                        self._consecutive_quick_crashes = 0

                    logger.error(
                        "ASR worker process died (exit code %s) after %.1fs "
                        "(%d consecutive quick crash%s)%s",
                        exit_code,
                        uptime,
                        self._consecutive_quick_crashes,
                        "" if self._consecutive_quick_crashes == 1 else "es",
                        f": {self.asr_last_error}" if self.asr_last_error else "",
                    )
                    # Crash-loop policy: halt ASR, keep the app. Restarting
                    # forever is pointless (the unit/model/build is broken),
                    # stopping the app would take NDI output, manual captions
                    # and the web UI down with it, and falling back to CPU
                    # would silently change the unit the operator chose. The
                    # problems panel says why and offers fixes; a fix or an
                    # asr config change starts a fresh generation.
                    if self._consecutive_quick_crashes >= _QUICK_CRASHES_BEFORE_HALT:
                        process.join(timeout=_ASR_FORCE_STOP_SECONDS)
                        with self._asr_lock:
                            if self._asr_process is process:
                                self._asr_process, self._asr_stop_event = None, None
                                self._asr_halted = True
                                self.resolved_asr_backend = "stopped"
                        if self._asr_halted:
                            logger.error(
                                "ASR stopped: the worker crashed %d times in a row shortly after starting "
                                "(asr.unit=%s, model=%s). The app keeps running without ASR captions; "
                                "fix it from the web UI's problems panel or by changing the asr config.",
                                self._consecutive_quick_crashes,
                                self._config.asr.unit,
                                self._config.asr.model,
                            )
                            self._request_health_check(invalidate=True)
                        continue

                    backoff = min(
                        _ASR_RESTART_BACKOFF_SECONDS * (2 ** max(self._consecutive_quick_crashes - 1, 0)),
                        _ASR_RESTART_BACKOFF_MAX_SECONDS,
                    )
                    # Interruptible: stop() can be called while this backoff is in
                    # progress (the crash loop's backoff climbs to 30s), and without
                    # checking the event again here, we'd spawn a fresh ASR process
                    # right after the rest of the app has already torn down -- an
                    # orphaned process stop() will never join again.
                    if self._stop_event.wait(backoff):
                        break
                    # Reap the dead process so it leaves no zombie, then take
                    # over only if nothing else replaced it while we waited
                    # out the backoff (a config change during the backoff
                    # spawns its own generation -- respawning here too would
                    # leave two workers fighting over the audio queue).
                    process.join(timeout=_ASR_FORCE_STOP_SECONDS)
                    with self._asr_lock:
                        if self._asr_process is process and not self._stop_event.is_set():
                            self._spawn_asr_process()
            self._stop_event.wait(1.0)

    def _on_config_changed(self, new_config: RootConfig) -> None:
        with self._config_lock:
            old_config = self._config
            self._config = new_config
            self._renderer.update_config(new_config.canvas, new_config.layout, new_config.style)
            if new_config.ndi != old_config.ndi or new_config.canvas != old_config.canvas:
                self._restart_ndi_stream()
            if new_config.asr != old_config.asr and asr_config_without_text(
                new_config.asr
            ) == asr_config_without_text(old_config.asr):
                # Only vocabulary/snap changed: applied live, no model reload.
                self._push_text_settings(new_config)
            elif new_config.asr != old_config.asr:
                # Also the way out of a halted ASR: a fresh generation with
                # a clean crash streak (_restart_asr_process).
                logger.info("asr config changed; restarting ASR worker")
                self._restart_asr_process()
            elif new_config.captions != old_config.captions:
                self._push_text_settings(new_config)
            if new_config.audio != old_config.audio:
                self._restart_audio_capture()
        # Re-run the setup checks against the new config (on their thread).
        self._request_health_check(invalidate=True)
        web = self._web
        if web is not None:
            # Outside the config lock: a bind/port change restarts the web
            # listener, which must not hold up the next config swap.
            web.on_config_changed(old_config, new_config)

    def _push_text_settings(self, config: RootConfig) -> None:
        """Send new text settings to the running worker (vocabulary,
        snapping, blocklist all apply from its next decode step). A worker
        spawned later reads them from the config anyway."""
        control_queue = getattr(self, "_asr_control_queue", None)
        if control_queue is None:
            return
        logger.info("vocabulary/blocklist changed; updating ASR worker live")
        control_queue.put(self._text_settings(config))

    @property
    def stop_requested(self) -> bool:
        """True once stop() has begun; headless mode uses it to exit. Since
        the 2026-10-17 failure policy no code path in the app calls stop()
        on its own (an ASR crash loop halts ASR only), so in practice this
        follows a signal."""
        return self._stopped

    def stop(self) -> None:
        """Idempotent: safe to call more than once (e.g. a signal handler
        stopping the app and then a `finally` block stopping it again) --
        without the guard, a second pass would double-close the NDI sender
        and re-join already-finished processes/threads for no reason.
        """
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        self._stop_event.set()
        if self._web is not None:
            self._web.stop()
        if self._health is not None:
            self._health.stop()
        if self._unit_usage is not None:
            self._unit_usage.stop()
        if self._ndi_stop_event is not None:
            self._ndi_stop_event.set()
        if self._audio_stop_event is not None:
            self._audio_stop_event.set()
        if self._watcher is not None:
            self._watcher.stop()
        if self._mic is not None:
            self._mic.stop()
        with self._asr_lock:
            asr_process, asr_stop_event = self._asr_process, self._asr_stop_event
            self._asr_process, self._asr_stop_event = None, None
        # Same escalation as a restart: a worker still loading a multi-GB model
        # will not honour a cooperative stop in time, and a daemon process left
        # behind here keeps the GPU and the audio queue busy after "shutdown".
        self._shut_down_asr_process(asr_process, asr_stop_event)
        if self._ndi_thread is not None:
            self._ndi_thread.join(timeout=5)
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=5)
        if self._audio_queue is not None:
            # Nothing reads the audio queue any more: drop what is buffered
            # and do not let interpreter exit wait for its feeder thread to
            # flush into a pipe nobody drains (seen: a SIGTERM that never
            # finished, the feeder blocked in a pipe write).
            self._drain_audio_queue()
            cancel = getattr(self._audio_queue, "cancel_join_thread", None)
            if cancel is not None:
                cancel()
        if self._sender is not None:
            self._sender.close()
        current = threading.current_thread()
        for thread in self._threads:
            # Never join the calling thread ("cannot join current thread"
            # would abort teardown midway, leaving the worker/mic/sender
            # uncleaned) -- the supervisor used to call stop() itself.
            if thread is not current and thread.is_alive():
                thread.join(timeout=5)
