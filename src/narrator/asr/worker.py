"""Multiprocessing entry point for the ASR worker: owns the whisper.cpp model,
VAD, and the streaming decode loop, running in its own OS process so a
native-code crash or a stalled GPU call can't freeze rendering/NDI output or
the GUI (see app_controller.py, which restarts this process if it dies).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import time

import numpy as np

from narrator.asr.asr_backend import AsrBackend, create_backend
from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.decoder_common import CaptionDecoder, create_decoder
from narrator.asr.text_settings import TextSettings
from narrator.asr.vad import SileroVad
from narrator.config.schema import AsrConfig

logger = logging.getLogger(__name__)

# Upper bound on frames ingested between two decode steps. Only a guard against
# a pathological producer starving the decode step entirely; app_controller's
# queue is itself bounded well below this.
_MAX_DRAIN_FRAMES = 1000

# The worker publishes the fraction of wall time it spends decoding (tick())
# over windows of this length, through a shared double (`busy_value`): the
# recording draft scheduler reads it to decide whether the device has
# headroom for an offline pass. -1 = not measured yet.
_BUSY_WINDOW_SECONDS = 5.0

# The worker owns exactly one AsrBackend per process lifetime; kept in a
# module global so the outer `run_asr_worker` can close it deterministically
# in a `finally` block regardless of how the loop exits (stop event, device
# policy error, any other crash).
_current_backend: AsrBackend | None = None


def run_asr_worker(
    config: AsrConfig,
    audio_queue: "mp.Queue[np.ndarray]",
    event_queue: "mp.Queue[CaptionEvent]",
    stop_event: "mp.synchronize.Event",
    log_level: str = "INFO",
    text_settings: TextSettings | None = None,
    control_queue: "mp.Queue[TextSettings] | None" = None,
    busy_value=None,
) -> None:
    """Blocking loop; intended to be the target of a multiprocessing.Process.
    Exits when `stop_event` is set, or when the backend raises
    DevicePolicyError (the configured device is unusable -- see backend.py).
    Any other uncaught exception also propagates and ends the process --
    app_controller.py's supervisor detects the exit and decides whether to
    restart.

    `text_settings` (vocabulary, snapping, blocklist) can be replaced live by
    putting a new TextSettings on `control_queue` -- no model reload.
    `busy_value` (an mp.Value("d")), when given, receives the decode busy
    fraction every few seconds (see _BUSY_WINDOW_SECONDS).
    """
    # cuda:<n> is nvidia-smi's (PCI) order; CUDA's own default is
    # fastest-first. Must be set before anything starts the CUDA runtime.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    logging.basicConfig(level=log_level)
    from narrator.logsetup import attach_worker_file_log

    attach_worker_file_log()
    try:
        _run(config, audio_queue, event_queue, stop_event, text_settings, control_queue, busy_value)
    except Exception as exc:
        # Tell the app why before dying: the exit code alone ("1") says
        # nothing, and the problems panel shows this text. mp.Queue flushes
        # its feeder thread at process exit, so the event is not lost.
        try:
            event_queue.put(CaptionEvent(CaptionEventType.ERROR, str(exc) or type(exc).__name__))
        except Exception:
            pass
        raise
    finally:
        # Deterministic native teardown. pywhispercpp only frees the whisper
        # context via GC (`Model.__del__`), and ONNX Runtime sessions behave
        # similarly; on a VRAM-constrained GPU the successor generation can
        # otherwise race this generation's cleanup and watch its own GPU init
        # fail for lack of memory. By the time this process exits, the model
        # is released or the process is dying anyway -- never still resident.
        if _current_backend is not None:
            try:
                _current_backend.close()
            except Exception:
                logger.exception("ASR worker backend close() failed during shutdown")


def _run(
    config: AsrConfig,
    audio_queue: "mp.Queue[np.ndarray]",
    event_queue: "mp.Queue[CaptionEvent]",
    stop_event: "mp.synchronize.Event",
    text_settings: TextSettings | None = None,
    control_queue: "mp.Queue[TextSettings] | None" = None,
    busy_value=None,
) -> None:
    global _current_backend
    backend = create_backend(config)
    _current_backend = backend
    vad = SileroVad(threshold=config.vad_threshold)
    # asr.decoder picks local_agreement (StreamingDecoder) or the segmenter;
    # both are driven the same drain-then-tick way below.
    decoder = create_decoder(backend, vad, config, event_queue.put, text_settings=text_settings)
    if text_settings is not None:
        backend.set_vocabulary(list(text_settings.vocabulary))

    busy = _BusyMeter(busy_value)
    while not stop_event.is_set():
        _apply_text_settings_updates(control_queue, backend, decoder)
        busy.maybe_publish()
        try:
            frame = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Ingest everything that queued up while the previous decode step was
        # running, *then* decode once. Feeding one frame per decode instead
        # makes the worker consume ~1 frame (30ms of audio) per decode step
        # (0.3-2.1s), so the bounded queue in app_controller saturates within
        # seconds and its drop-oldest policy starts discarding the great
        # majority of incoming audio -- the decode window ends up stitched
        # from non-adjacent fragments, which is what garbled the captions.
        # VAD is cheap enough to keep up with real time on its own, so the
        # drain always completes.
        decoder.feed(frame, run_decode_step=False)
        for _ in range(_MAX_DRAIN_FRAMES):
            try:
                decoder.feed(audio_queue.get_nowait(), run_decode_step=False)
            except queue.Empty:
                break
        tick_started = time.monotonic()
        decoder.tick()
        busy.add(time.monotonic() - tick_started)


class _BusyMeter:
    """Accumulates decode time and publishes busy = decode time / wall time
    per _BUSY_WINDOW_SECONDS window into a shared mp.Value."""

    def __init__(self, value, clock=time.monotonic) -> None:
        self._value = value
        self._clock = clock
        self._window_start = clock()
        self._busy = 0.0

    def add(self, seconds: float) -> None:
        self._busy += seconds
        self.maybe_publish()

    def maybe_publish(self) -> None:
        if self._value is None:
            return
        now = self._clock()
        elapsed = now - self._window_start
        if elapsed < _BUSY_WINDOW_SECONDS:
            return
        self._value.value = min(1.0, self._busy / elapsed)
        self._window_start, self._busy = now, 0.0


def _apply_text_settings_updates(control_queue, backend: AsrBackend, decoder: CaptionDecoder) -> None:
    """Apply the newest TextSettings waiting on the control queue, if any."""
    if control_queue is None:
        return
    latest = None
    while True:
        try:
            latest = control_queue.get_nowait()
        except queue.Empty:
            break
    if latest is not None:
        backend.set_vocabulary(list(latest.vocabulary))
        decoder.set_text_settings(latest)
        logger.info(
            "text settings updated live: %d vocabulary entries, snap=%s, blocklist=%s",
            len(latest.vocabulary),
            latest.vocabulary_snap,
            "on" if latest.blocklist else "off",
        )
