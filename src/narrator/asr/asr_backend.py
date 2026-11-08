"""The ASR backend interface (todo C6): what the decoders
(decoder_common, streaming_decoder, segmenter), stream_sim, bench and the
worker need from a speech recognizer. `WhisperBackend` (backend.py,
whisper.cpp through pywhispercpp) is today's only implementation; ONNX
(sherpa-onnx) or NeMo models would be others, picked by the model catalog
(narrator.asr.catalog) in `create_backend`.

Also the data types every backend returns (word times, segment results)
and its error types, so a decoder never imports a concrete backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


class DevicePolicyError(RuntimeError):
    """The configured processing unit could not be used or stopped working.

    Raised instead of silently falling back to CPU. The configured unit is
    final by policy: a transient failure (the previous worker generation not
    yet having released the GPU, a driver hiccup) must surface as a worker
    crash the supervisor can see -- not as an invisible downgrade that leaves
    the operator's "GPU" config running 10x slower on CPU with a dashboard
    indicator claiming all is well.
    """


class ModelLoadError(RuntimeError):
    """The configured model could not be loaded (not a local file, and not a
    downloadable model name, or the download failed)."""


@dataclass(frozen=True)
class WordTime:
    """One word of the latest decode, with its start/end in seconds of the
    decode input (AsrBackend.last_word_times)."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SegmentToken:
    """One decoded token: model token id, text, probability, and its
    start/end in seconds of the (padded) decode input when reported."""

    id: int
    text: str
    p: float
    t0: float = 0.0
    t1: float = 0.0
    # The token's raw UTF-8 bytes: a Polish letter can be split across two
    # tokens, so text is joined as bytes and decoded once.
    raw: bytes = field(default=b"", repr=False, compare=False)


@dataclass(frozen=True)
class SegmentResult:
    """One whole-segment decode (AsrBackend.decode_segment): the filtered
    caption text, the kept text tokens (with their confidences), the
    language it was decoded in, the wall time it took, and -- when a quality
    gate discarded it -- why (`text` is then empty)."""

    text: str
    tokens: tuple[SegmentToken, ...] = field(default=())
    language: str | None = None
    decode_seconds: float = 0.0
    rejected: str | None = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@runtime_checkable
class AsrBackend(Protocol):
    """A loaded speech recognizer on one processing unit.

    Decoders call only these members. Audio is mono float32 at 16 kHz. A
    failure of the unit itself raises DevicePolicyError (never a silent
    switch to another unit); `detect_language` returns None on failure so
    the caller can retry later.
    """

    # What it runs on, for the dashboard ("Vulkan (NVIDIA ... [discrete])").
    resolved_device_description: str
    # Words of the latest `transcribe` with their times in seconds of its
    # input; empty after a rejected or empty decode, or when the backend
    # has no word times.
    last_word_times: list[WordTime]
    # Decodes discarded by the quality gates, by gate (diagnostics).
    rejected_decodes: dict[str, int]

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[tuple[float, float, str]]:
        """One decode of a (growing) window: (start_s, end_s, text) segments.
        `language` forces the language (None = the backend decides from the
        whitelist); `prompt` is text said just before this audio."""
        ...

    def decode_segment(
        self,
        audio: np.ndarray,
        *,
        prompt: str | None = None,
        language: str | None = None,
        drop_trailing_period: bool = False,
        peek: bool = False,
        sample_rate: int = 16000,
    ) -> SegmentResult:
        """Decode one whole segment once, with token confidences/times."""
        ...

    def detect_language(self, audio: np.ndarray) -> tuple[str, float, float] | None:
        """(language, confidence, margin to the runner-up) over the
        configured whitelist, None on failure."""
        ...

    def set_vocabulary(self, vocabulary: list[str]) -> None:
        """Replace the event vocabulary (prompt / biasing), live."""
        ...

    def close(self) -> None:
        """Release the model and its device memory deterministically."""
        ...


def create_backend(config) -> AsrBackend:
    """The backend for `config` (AsrConfig): the model catalog says which
    implementation runs the model; only whisper.cpp exists today."""
    from narrator.asr.catalog import AsrBackendKind, entry_for

    entry = entry_for(config.model)
    if entry.backend == AsrBackendKind.WHISPER_CPP:
        from narrator.asr.backend import WhisperBackend

        return WhisperBackend(config)
    raise ModelLoadError(f"model {config.model!r} needs the {entry.backend.value} backend, which is not installed")
