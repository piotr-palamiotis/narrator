"""Continuous microphone capture via sounddevice (PortAudio), cross-platform.

The sounddevice callback runs on PortAudio's own audio thread; it must never
block. Frames are pushed onto a bounded queue.Queue and consumed elsewhere
(the ASR worker feed thread in app_controller.py). If the consumer falls
behind, the oldest queued frame is dropped rather than blocking the audio
callback -- a dropped frame occasionally is far less harmful than stalling
live capture.

Whisper requires 16kHz mono. We ask PortAudio for 16kHz directly; if the
device can't honor that, we open at its default rate and resample every
frame with scipy.signal.resample_poly before it reaches the queue, so
everything downstream always sees a clean 16kHz stream.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from narrator.audio.devices import resolve_device

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


class MicCapture:
    def __init__(
        self,
        device_spec: str = "default",
        frame_ms: int = 30,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        max_queued_frames: int = 200,
    ) -> None:
        self._device_index = resolve_device(device_spec)
        self._target_rate = target_sample_rate
        self._frame_ms = frame_ms
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queued_frames)
        self._stream: sd.InputStream | None = None
        self._native_rate: int | None = None

    def start(self) -> None:
        blocksize = int(self._target_rate * self._frame_ms / 1000)
        try:
            self._stream = sd.InputStream(
                device=self._device_index,
                channels=1,
                samplerate=self._target_rate,
                blocksize=blocksize,
                dtype="float32",
                callback=self._on_audio,
            )
            self._stream.start()
            self._native_rate = self._target_rate
            logger.info("Mic opened directly at %d Hz", self._target_rate)
        except sd.PortAudioError:
            logger.info("Device rejected %d Hz; opening at device default and resampling", self._target_rate)
            device_info = sd.query_devices(self._device_index, "input")
            native_rate = int(device_info["default_samplerate"])
            native_blocksize = int(native_rate * self._frame_ms / 1000)
            self._native_rate = native_rate
            self._stream = sd.InputStream(
                device=self._device_index,
                channels=1,
                samplerate=native_rate,
                blocksize=native_blocksize,
                dtype="float32",
                callback=self._on_audio,
            )
            self._stream.start()

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug("sounddevice status: %s", status)
        mono = indata[:, 0].copy()
        if self._native_rate != self._target_rate:
            mono = resample_poly(mono, self._target_rate, self._native_rate).astype(np.float32)
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(mono)
            except queue.Full:
                pass  # extremely unlikely race; drop this frame

    def frames(self, timeout: float = 1.0, stop_event: threading.Event | None = None) -> Iterator[np.ndarray]:
        """Blocking generator yielding mono float32 frames at `target_sample_rate`.

        `stop_event` is re-checked every `timeout` seconds even while the
        queue stays empty (e.g. after `stop()` closes the stream) -- without
        it, a permanently empty queue makes this loop back only on `continue`
        and never yield, so a caller's own stop check (reached only in the
        loop body, after a yield) never runs and the consuming thread never exits.
        """
        while stop_event is None or not stop_event.is_set():
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                continue

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
