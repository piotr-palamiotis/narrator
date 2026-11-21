"""Offline (best-effort, not live) transcription helper: span planning for
scripts/eval_transcript.py.

Long audio is decoded in bounded SPANS with a fresh whisper context each:
one monolithic decode lets a hallucination loop (music, applause) poison
everything after it through whisper's own context carry-over -- measured on
the reference library: 99% vs 36% WER. Spans end at silences where there
are any, so a cut rarely lands inside a word.
"""
from __future__ import annotations

from collections.abc import Sequence

MAX_SPAN_SECONDS = 60.0

_SPEECH_PAD_SECONDS = 0.3  # kept around speech so edges are not clipped
_MIN_SPAN_SECONDS = 0.5


def speech_spans(
    non_speech: Sequence[tuple[float, float]],
    duration_s: float,
    max_span_s: float = MAX_SPAN_SECONDS,
    pad: float = _SPEECH_PAD_SECONDS,
) -> list[tuple[float, float]]:
    """Complement of the non-speech windows, padded slightly into them (so
    speech at a music boundary isn't clipped), with long spans split at
    `max_span_s` so one poisoned stretch can only ruin one chunk."""
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for ns_start, ns_end in sorted(non_speech):
        if ns_start - pad > cursor:
            spans.append((cursor, ns_start - pad))
        cursor = max(cursor, ns_end + pad)
    if cursor < duration_s:
        spans.append((cursor, duration_s))

    bounded: list[tuple[float, float]] = []
    for s, e in spans:
        while e - s > max_span_s:
            bounded.append((s, s + max_span_s))
            s += max_span_s
        if e - s > _MIN_SPAN_SECONDS:
            bounded.append((s, e))
    return bounded
