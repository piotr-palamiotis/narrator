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
        self._asr_stop_event = mp.Event()
        self._asr_process: mp.Process | None = None

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._watcher: ConfigWatcher | None = None
        self.resolved_asr_backend: str = "unresolved"

        self._ndi_stop_event: threading.Event | None = None
        self._ndi_thread: threading.Thread | None = None

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

        self._mic = MicCapture(self._config.audio.device, self._config.audio.frame_ms)
        self._mic.start()

        self._spawn_asr_process()

        self._spawn_thread(self._feed_audio_loop)
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

    def _spawn_asr_process(self) -> None:
        self._asr_stop_event.clear()
        asr_config = self._config.asr
        if self._force_asr_cpu and asr_config.device != AsrDevice.CPU:
            asr_config = asr_config.model_copy(update={"device": AsrDevice.CPU})
        self._asr_process = mp.Process(
            target=run_asr_worker,
            args=(
                asr_config,
                self._audio_queue,
                self._event_queue,
                self._asr_stop_event,
                self._config.app.log_level,
            ),
            daemon=True,
        )
        self._asr_process.start()
        self._asr_spawn_time = time.monotonic()

    def _feed_audio_loop(self) -> None:
        for frame in self._mic.frames(timeout=0.5, stop_event=self._stop_event):
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
            if self._asr_process is not None and not self._asr_process.is_alive():
                exit_code = self._asr_process.exitcode
                if not self._stop_event.is_set():
                    uptime = time.monotonic() - self._asr_spawn_time
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
                    time.sleep(backoff)
                    self._spawn_asr_process()
            time.sleep(1.0)

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
                self._asr_stop_event.set()
                if self._asr_process is not None:
                    self._asr_process.join(timeout=5)
                self._spawn_asr_process()
            if new_config.audio != old_config.audio:
                logger.warning("audio config changed; restart the app for this to take effect")

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
        self._asr_stop_event.set()
        if self._ndi_stop_event is not None:
            self._ndi_stop_event.set()
        if self._watcher is not None:
            self._watcher.stop()
        if self._mic is not None:
            self._mic.stop()
        if self._asr_process is not None:
            self._asr_process.join(timeout=5)
        if self._ndi_thread is not None:
            self._ndi_thread.join(timeout=5)
        if self._sender is not None:
            self._sender.close()
        for thread in self._threads:
            thread.join(timeout=5)
