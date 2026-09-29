"""Multiprocessing entry point for the ASR worker: owns the whisper.cpp model,
VAD, and the streaming decode loop, running in its own OS process so a
native-code crash or a stalled GPU call can't freeze rendering/NDI output or
the GUI (see app_controller.py, which restarts this process if it dies).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue

import numpy as np

from narrator.asr.backend import WhisperBackend
from narrator.asr.events import CaptionEvent
from narrator.asr.streaming_decoder import StreamingDecoder
from narrator.asr.vad import SileroVad
from narrator.config.schema import AsrConfig

logger = logging.getLogger(__name__)

# Upper bound on frames ingested between two decode steps. Only a guard against
# a pathological producer starving the decode step entirely; app_controller's
# queue is itself bounded well below this.
_MAX_DRAIN_FRAMES = 1000

# The worker owns exactly one WhisperBackend per process lifetime; kept in a
# module global so the outer `run_asr_worker` can close it deterministically
# in a `finally` block regardless of how the loop exits (stop event, device
# policy error, any other crash).
_current_backend: WhisperBackend | None = None


def run_asr_worker(
    config: AsrConfig,
    audio_queue: "mp.Queue[np.ndarray]",
    event_queue: "mp.Queue[CaptionEvent]",
    stop_event: "mp.synchronize.Event",
    log_level: str = "INFO",
) -> None:
    """Blocking loop; intended to be the target of a multiprocessing.Process.
    Exits when `stop_event` is set, or when the backend raises
    DevicePolicyError (the configured device is unusable -- see backend.py).
    Any other uncaught exception also propagates and ends the process --
    app_controller.py's supervisor detects the exit and decides whether to
    restart.
    """
    logging.basicConfig(level=log_level)
    try:
        _run(config, audio_queue, event_queue, stop_event)
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
) -> None:
    global _current_backend
    backend = WhisperBackend(config)
    _current_backend = backend
    vad = SileroVad(threshold=config.vad_threshold)
    decoder = StreamingDecoder(backend, vad, config, on_event=event_queue.put)

    while not stop_event.is_set():
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
        decoder.tick()
