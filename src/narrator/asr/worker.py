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


def run_asr_worker(
    config: AsrConfig,
    audio_queue: "mp.Queue[np.ndarray]",
    event_queue: "mp.Queue[CaptionEvent]",
    stop_event: "mp.synchronize.Event",
    log_level: str = "INFO",
) -> None:
    """Blocking loop; intended to be the target of a multiprocessing.Process.
    Exits when `stop_event` is set. Any uncaught exception propagates and
    ends the process -- app_controller.py's supervisor detects the exit and
    decides whether to restart.
    """
    logging.basicConfig(level=log_level)
    backend = WhisperBackend(config)
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
