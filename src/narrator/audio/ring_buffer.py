"""Growing/trimmable float32 audio window, used by the ASR worker to hold the
current sliding decode window (see narrator.asr.streaming_decoder).

This is a plain in-process numpy buffer, not a cross-process shared-memory
structure: audio frames reach the ASR worker process via a
multiprocessing.Queue (see app_controller.py), which is simple, robust, and
negligible overhead at 16kHz mono (~32KB/s) -- shared memory would only pay
for itself at much higher bandwidths than a single mono voice channel.
"""
from __future__ import annotations

import numpy as np


class AudioWindow:
    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._samples = np.zeros(0, dtype=np.float32)

    def append(self, frame: np.ndarray) -> None:
        self._samples = np.concatenate([self._samples, frame])

    def trim_to_last_seconds(self, seconds: float) -> None:
        max_samples = int(seconds * self._sample_rate)
        if len(self._samples) > max_samples:
            self._samples = self._samples[-max_samples:]

    def drop_first_seconds(self, seconds: float, keep_overlap_seconds: float = 0.0) -> None:
        """Discard confirmed/committed audio from the front, retaining a small
        overlap for decode continuity (see streaming_decoder's local-agreement
        commit step).
        """
        drop_samples = int(max(seconds - keep_overlap_seconds, 0) * self._sample_rate)
        if drop_samples > 0:
            self._samples = self._samples[drop_samples:]

    def clear(self) -> None:
        self._samples = np.zeros(0, dtype=np.float32)

    @property
    def duration_seconds(self) -> float:
        return len(self._samples) / self._sample_rate

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)
