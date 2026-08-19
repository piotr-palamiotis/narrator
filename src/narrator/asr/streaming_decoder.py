"""Sliding-window + local-agreement incremental decode loop.

Mirrors the well-known "whisper_streaming" pattern: whisper.cpp is re-run on
a growing audio window every `step_ms`; the new hypothesis is compared
word-by-word against the previous one, and a prefix is only committed
(emitted as a FINAL CaptionEvent) once it has stayed stable for
`agreement_passes` consecutive decode steps. This trades a little latency
(bounded by step_ms * agreement_passes) for captions that don't visibly
backtrack -- appropriate for a broadcast lower-third. On VAD-detected silence
past `end_silence_ms`, whatever remains is force-finalized regardless of
agreement, bounding worst-case latency for short utterances.

Simplification, called out explicitly: the audio window is bounded by
unconditionally trimming to `max_context_seconds` rather than trimming
precisely at committed-word audio boundaries (which would require per-token
timestamps and materially more bookkeeping for a marginal-latency edge case
that only matters for unusually long continuous speech without a pause).
End-of-utterance silence detection is the primary bound in the common case.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from narrator.asr.backend import WhisperBackend
from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.vad import SileroVad
from narrator.audio.ring_buffer import AudioWindow
from narrator.config.schema import AsrConfig

logger = logging.getLogger(__name__)

_VAD_WINDOW_MS = 32.0  # fixed by the Silero graph's 512-sample-at-16kHz requirement
_MIN_DECODE_SECONDS = 0.3  # skip decode steps on near-empty windows


def _common_prefix_length(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _agreed_prefix_length(hypotheses: list[list[str]]) -> int:
    """Length of the prefix shared identically by every hypothesis in the list
    -- the standard "local agreement" measure: a word is confirmed once it has
    appeared at the same position in each of the last `agreement_passes`
    decode-step hypotheses.
    """
    common_len = len(hypotheses[0])
    for hypothesis in hypotheses[1:]:
        common_len = min(common_len, _common_prefix_length(hypothesis, hypotheses[0]))
    return common_len


class StreamingDecoder:
    def __init__(
        self,
        backend: WhisperBackend,
        vad: SileroVad,
        config: AsrConfig,
        on_event: Callable[[CaptionEvent], None],
        sample_rate: int = 16000,
        clock: Callable[[], float] = time.monotonic,
        min_decode_seconds: float = _MIN_DECODE_SECONDS,
    ) -> None:
        self._backend = backend
        self._vad = vad
        self._config = config
        self._on_event = on_event
        self._window = AudioWindow(sample_rate=sample_rate)
        self._clock = clock
        self._min_decode_seconds = min_decode_seconds

        self._in_speech = False
        self._silence_ms_accum = 0.0
        self._last_decode_monotonic = 0.0
        self._last_speech_monotonic: float | None = None
        self._cleared_since_silence = True

        self._recent_hypotheses: list[list[str]] = []  # last `agreement_passes` hypotheses
        self._committed_word_count = 0

        self._on_event(CaptionEvent(CaptionEventType.STATUS, backend.resolved_device_description))

    def feed(self, frame: np.ndarray) -> None:
        """Feed one mono float32 frame at the configured sample rate."""
        probabilities = self._vad.process(frame)
        now = self._clock()

        for probability in probabilities:
            if probability >= self._config.vad_threshold:
                if not self._in_speech:
                    self._start_utterance()
                self._silence_ms_accum = 0.0
                self._last_speech_monotonic = now
                self._cleared_since_silence = False
            else:
                self._silence_ms_accum += _VAD_WINDOW_MS

        if self._in_speech:
            self._window.append(frame)
            self._window.trim_to_last_seconds(self._config.max_context_seconds)

            if self._silence_ms_accum >= self._config.end_silence_ms:
                self._finalize_utterance()
                return

            if (now - self._last_decode_monotonic) * 1000.0 >= self._config.step_ms:
                self._run_decode_step(now)
        else:
            self._maybe_emit_clear(now)

    def _start_utterance(self) -> None:
        self._in_speech = True
        self._window.clear()
        self._recent_hypotheses = []
        self._committed_word_count = 0
        self._last_decode_monotonic = 0.0

    def _run_decode_step(self, now: float) -> None:
        self._last_decode_monotonic = now
        if self._window.duration_seconds < self._min_decode_seconds:
            return

        try:
            segments = self._backend.transcribe(self._window.samples)
        except Exception:
            logger.exception("ASR decode step failed; skipping this step")
            return

        text = " ".join(seg_text for _, _, seg_text in segments if seg_text).strip()
        words = text.split()

        self._recent_hypotheses.append(words)
        if len(self._recent_hypotheses) > self._config.agreement_passes:
            self._recent_hypotheses.pop(0)

        if len(self._recent_hypotheses) >= self._config.agreement_passes:
            agreed_len = _agreed_prefix_length(self._recent_hypotheses)
            if agreed_len > self._committed_word_count:
                new_words = words[self._committed_word_count : agreed_len]
                if new_words:
                    self._emit_final(" ".join(new_words))
                self._committed_word_count = agreed_len

        if self._config.show_partial_captions:
            partial_words = words[self._committed_word_count :]
            if partial_words:
                self._on_event(CaptionEvent(CaptionEventType.PARTIAL, " ".join(partial_words)))

    def _finalize_utterance(self) -> None:
        if self._window.duration_seconds >= self._min_decode_seconds:
            try:
                segments = self._backend.transcribe(self._window.samples)
                text = " ".join(seg_text for _, _, seg_text in segments if seg_text).strip()
                words = text.split()
                remaining = words[self._committed_word_count :]
                if remaining:
                    self._emit_final(" ".join(remaining))
            except Exception:
                logger.exception("Final decode pass failed for utterance")

        self._in_speech = False
        self._window.clear()
        self._recent_hypotheses = []
        self._committed_word_count = 0
        self._silence_ms_accum = 0.0

    def _emit_final(self, text: str) -> None:
        self._on_event(CaptionEvent(CaptionEventType.FINAL, text))

    def _maybe_emit_clear(self, now: float) -> None:
        if self._cleared_since_silence or self._last_speech_monotonic is None:
            return
        idle_seconds = now - self._last_speech_monotonic
        if idle_seconds >= self._config.clear_after_silence_seconds:
            self._on_event(CaptionEvent(CaptionEventType.CLEAR))
            self._cleared_since_silence = True
