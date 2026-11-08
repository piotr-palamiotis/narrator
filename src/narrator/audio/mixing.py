"""Channel selection and streaming resampling shared by every audio source.

`audio.channels` lists the 1-based input channels averaged into the mono
signal ASR sees. Both MicCapture (PortAudio, interleaved frames x channels)
and NdiAudioCapture (NDI, planar channels x samples) funnel through
`mix_channels`, so the selection semantics are identical.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import gcd

import numpy as np
from scipy.signal import resample_poly

NDI_DEVICE_PREFIX = "ndi:"


class ChannelSelectionError(ValueError):
    """The configured `audio.channels` do not exist on the input."""


def required_channels(channels: Sequence[int]) -> int:
    """Number of input channels a stream must provide to cover `channels`."""
    return max(channels)


def check_channels_available(channels: Sequence[int], available: int, source: str) -> None:
    """Raise a clear error if `channels` asks for more than `available`."""
    needed = required_channels(channels)
    if needed > available:
        raise ChannelSelectionError(
            f"audio.channels {list(channels)} needs {needed} input channel(s) but "
            f"{source} provides only {available}. Pick channels between 1 and {available}."
        )


def mix_channels(data: np.ndarray, channels: Sequence[int]) -> np.ndarray:
    """Average the chosen 1-based `channels` of `data` (frames, n_channels)
    into a mono float32 vector. A single channel is copied, not viewed, so
    the caller may keep the result after the source buffer is reused."""
    idx = [ch - 1 for ch in channels]
    if len(idx) == 1:
        return np.array(data[:, idx[0]], dtype=np.float32)
    return np.mean(data[:, idx], axis=1, dtype=np.float32)


class StreamResampler:
    """Chunk-by-chunk `resample_poly` that stays continuous across chunk
    boundaries by carrying a short left context of the previous input (a
    stateless per-chunk resample starts each chunk from zero padding, which
    clicks at every boundary). The context is a multiple of the decimation
    factor, so its output length is an exact integer and can be dropped."""

    _CONTEXT_MS = 20

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        g = gcd(source_rate, target_rate)
        self._up = target_rate // g
        self._down = source_rate // g
        ctx = int(source_rate * self._CONTEXT_MS / 1000)
        self._context_len = max(self._down, ctx - ctx % self._down)
        self._context = np.zeros(0, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        if self._up == self._down:
            return chunk
        # Only whole multiples of `down` are resampled so every chunk starts
        # on the output sample grid; the < `down` leftover waits for the next
        # call. Otherwise odd chunk lengths (1024 @ 48k -> 16k) slip the phase.
        buf = np.concatenate([self._pending, chunk])
        usable = len(buf) - len(buf) % self._down
        self._pending = buf[usable:].copy()
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        joined = np.concatenate([self._context, buf[:usable]])
        skip = len(self._context) * self._up // self._down
        out = resample_poly(joined, self._up, self._down).astype(np.float32)
        self._context = joined[-self._context_len:].copy()
        return out[skip:]


class FrameChunker:
    """Re-slices an arbitrary stream of mono samples into fixed-size frames,
    keeping the remainder for the next push."""

    def __init__(self, frame_samples: int) -> None:
        self.frame_samples = frame_samples
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        buf = np.concatenate([self._pending, np.asarray(samples, dtype=np.float32)])
        n = len(buf) // self.frame_samples
        frames = [buf[i * self.frame_samples:(i + 1) * self.frame_samples].copy() for i in range(n)]
        self._pending = buf[n * self.frame_samples:].copy()
        return frames

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
