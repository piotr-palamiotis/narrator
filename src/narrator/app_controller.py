"""Wires audio capture, the ASR worker process, rendering, NDI sending, config
hot-reload, and (optionally) the GUI into one running application.

Concurrency model (see the plan's threading table):
  - audio capture: sounddevice callback thread (inside MicCapture)
  - a feed thread drains MicCapture's local queue into the ASR worker's
    multiprocessing.Queue
  - ASR worker: separate OS process (see narrator.asr.worker), supervised
    here and restarted if it crashes
  - an event-drain thread applies CaptionEvents from the worker to the renderer
  - render + NDI send: fixed-fps thread (narrator.ndi.sender.run_render_send_loop)
  - config watcher: watchdog thread, applied via one atomic swap path
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path

from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.worker import run_asr_worker
from narrator.audio.capture import MicCapture
from narrator.config.loader import ConfigWatcher
from narrator.config.schema import AsrDevice, RootConfig
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
# in a row, the supervisor stops trusting the configured device and forces
# CPU for subsequent respawns -- see _supervise_asr_loop.
_QUICK_CRASH_WINDOW_SECONDS = 15.0
_QUICK_CRASHES_BEFORE_CPU_FALLBACK = 3


class AppController:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._config: RootConfig | None = None
        self._config_lock = threading.Lock()

        self._renderer: CaptionRenderer | None = None
        self._sender: NdiCaptionSender | None = None
        self._mic: MicCapture | None = None

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
        self._force_asr_cpu: bool = False

        # "auto": captions come from the ASR worker, as before. "manual": the
        # operator types captions in the GUI instead -- ASR events other than
        # STATUS are ignored while manual, so a stray partial/final/clear from
        # the (still-running) worker can't clobber manually-entered text. See
        # set_caption_mode/submit_manual_text/clear_captions.
        self._caption_mode: str = "auto"

    def start(self) -> None:
        self._watcher = ConfigWatcher(self._config_path, self._on_config_changed)
        self._config = self._watcher.start()
        logging.basicConfig(level=self._config.app.log_level)

        self._renderer = CaptionRenderer(self._config.canvas, self._config.layout, self._config.style)
        self._start_ndi_stream()
        self._start_audio_capture()

        with self._asr_lock:
            self._spawn_asr_process()

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
        self._sender = NdiCaptionSender(self._config.ndi, self._config.canvas)
        ndi_stop_event = self._ndi_stop_event
        sender = self._sender
        fps = self._config.ndi.fps
        self._ndi_thread = threading.Thread(
            target=lambda: run_render_send_loop(self._renderer, sender, fps, ndi_stop_event),
            daemon=True,
        )
        self._ndi_thread.start()

    def _restart_ndi_stream(self) -> None:
        logger.info("ndi/canvas config changed; restarting NDI stream")
        old_stop_event, old_thread, old_sender = self._ndi_stop_event, self._ndi_thread, self._sender
        if old_stop_event is not None:
            old_stop_event.set()
        if old_thread is not None:
            old_thread.join(timeout=5)
        if old_sender is not None:
            old_sender.close()
        self._start_ndi_stream()

    def _start_audio_capture(self) -> None:
        """(Re)creates mic capture and its feed thread against the current
        config. Uses its own stop event (distinct from the app-wide
        `_stop_event`) so an `audio` config change can restart just capture
        -- see `_on_config_changed` -- without tearing down the rest of the app.
        """
        self._audio_stop_event = threading.Event()
        self._mic = MicCapture(self._config.audio.device, self._config.audio.frame_ms)
        self._mic.start()
        mic, audio_stop_event = self._mic, self._audio_stop_event
        self._audio_thread = threading.Thread(
            target=lambda: self._feed_audio_loop(mic, audio_stop_event),
            daemon=True,
        )
        self._audio_thread.start()

    def _restart_audio_capture(self) -> None:
        logger.info("audio config changed; restarting audio capture")
        old_stop_event, old_thread, old_mic = self._audio_stop_event, self._audio_thread, self._mic
        if old_stop_event is not None:
            old_stop_event.set()
        if old_thread is not None:
            old_thread.join(timeout=5)
        if old_mic is not None:
            old_mic.stop()
        self._start_audio_capture()

    def _spawn_asr_process(self) -> None:
        """Start a new ASR worker generation. Caller must hold `_asr_lock`."""
        asr_config = self._config.asr
        if self._force_asr_cpu and asr_config.device != AsrDevice.CPU:
            asr_config = asr_config.model_copy(update={"device": AsrDevice.CPU})
        stop_event = mp.Event()
        process = mp.Process(
            target=run_asr_worker,
            args=(
                asr_config,
                self._audio_queue,
                self._event_queue,
                stop_event,
                self._config.app.log_level,
            ),
            daemon=True,
        )
        process.start()
        self._asr_process = process
        self._asr_stop_event = stop_event
        self._asr_spawn_time = time.monotonic()

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
            # A deliberate restart is a fresh attempt, not a continuation of
            # whatever crash streak preceded it.
            self._consecutive_quick_crashes = 0

        self._shut_down_asr_process(old_process, old_stop_event)

        with self._asr_lock:
            if self._stop_event.is_set():
                return
            self._spawn_asr_process()

    def _feed_audio_loop(self, mic: MicCapture, stop_event: threading.Event) -> None:
        for frame in mic.frames(timeout=0.5, stop_event=stop_event):
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
        # STATUS is always applied so the GUI's backend indicator stays live
        # even while manual mode is ignoring transcription output.
        if event.type == CaptionEventType.STATUS:
            self.resolved_asr_backend = event.text
            logger.info("ASR backend resolved: %s", event.text)
            return
        if self._caption_mode != "auto":
            return
        if event.type == CaptionEventType.FINAL:
            self._renderer.set_partial_preview("")
            self._renderer.append_final_text(event.text)
        elif event.type == CaptionEventType.PARTIAL:
            self._renderer.set_partial_preview(event.text)
        elif event.type == CaptionEventType.CLEAR:
            self._renderer.clear()

    def set_caption_mode(self, mode: str) -> None:
        """Switches between "auto" (ASR-driven) and "manual" (operator-typed)
        captions. The ASR worker keeps running underneath in either case --
        switching back to auto is instant, with no model reload -- but its
        output is only applied to the renderer while auto is selected.
        """
        if mode not in ("auto", "manual"):
            raise ValueError(f"invalid caption mode: {mode!r}")
        self._caption_mode = mode

    def submit_manual_text(self, text: str) -> None:
        """Appends operator-typed text as a finalized caption line. Intended
        for use while caption_mode == "manual", same as an ASR FINAL event.
        """
        text = text.strip()
        if not text or self._renderer is None:
            return
        self._renderer.set_partial_preview("")
        self._renderer.append_final_text(text)

    def clear_captions(self) -> None:
        """Clears the on-screen caption buffer. Works in either caption mode
        -- backs both the GUI's Clear button and its clear keyboard shortcut.
        """
        if self._renderer is not None:
            self._renderer.clear()

    def _supervise_asr_loop(self) -> None:
        while not self._stop_event.is_set():
            # Snapshot under the lock. `_asr_process` is None precisely while a
            # deliberate restart is in flight, which is how an intentional
            # stop is distinguished from a crash without a separate flag.
            with self._asr_lock:
                process = self._asr_process
                spawn_time = self._asr_spawn_time
            if process is not None and not process.is_alive():
                exit_code = process.exitcode
                if not self._stop_event.is_set():
                    uptime = time.monotonic() - spawn_time
                    if uptime < _QUICK_CRASH_WINDOW_SECONDS:
                        self._consecutive_quick_crashes += 1
                    else:
                        self._consecutive_quick_crashes = 0

                    logger.error(
                        "ASR worker process died (exit code %s) after %.1fs; restarting "
                        "(%d consecutive quick crash%s)",
                        exit_code,
                        uptime,
                        self._consecutive_quick_crashes,
                        "" if self._consecutive_quick_crashes == 1 else "es",
                    )

                    if (
                        not self._force_asr_cpu
                        and self._consecutive_quick_crashes >= _QUICK_CRASHES_BEFORE_CPU_FALLBACK
                        and self._config.asr.device != AsrDevice.CPU
                    ):
                        self._force_asr_cpu = True
                        self._consecutive_quick_crashes = 0
                        logger.error(
                            "ASR worker crashed %d times in a row shortly after starting "
                            "(configured device=%s); forcing CPU for subsequent restarts. "
                            "Change asr.device to 'cpu' to make this permanent, or restart "
                            "the app to retry the configured device.",
                            _QUICK_CRASHES_BEFORE_CPU_FALLBACK,
                            self._config.asr.device.value,
                        )

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
            if new_config.asr != old_config.asr:
                logger.info("asr config changed; restarting ASR worker")
                if new_config.asr.device != old_config.asr.device:
                    # A deliberate device change means the crash-loop breaker's
                    # state (if any) is stale -- give the newly-chosen device a
                    # fresh, un-penalized try.
                    self._force_asr_cpu = False
                    self._consecutive_quick_crashes = 0
                self._restart_asr_process()
            if new_config.audio != old_config.audio:
                self._restart_audio_capture()

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
        if self._sender is not None:
            self._sender.close()
        for thread in self._threads:
            thread.join(timeout=5)
