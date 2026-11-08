"""Offline (best-effort, not live) transcription helpers: span planning and
segment -> cue shaping, shared by the recording draft pass
(narrator.recording.draft) and scripts/eval_transcript.py.

Long audio is decoded in bounded SPANS with a fresh whisper context each:
one monolithic decode lets a hallucination loop (music, applause) poison
everything after it through whisper's own context carry-over -- measured on
the reference library: 99% vs 36% WER. Spans end at silences where there
are any, so a cut rarely lands inside a word.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SAMPLE_RATE = 16000
MAX_SPAN_SECONDS = 60.0

# VAD span planning (planned_speech_spans)
_SPEECH_PAD_SECONDS = 0.3  # kept around speech so edges are not clipped
_MERGE_GAP_SECONDS = 1.0  # pauses shorter than this stay inside one span
_MIN_SPAN_SECONDS = 0.5
_CUT_SEARCH_FROM = 0.5  # a forced cut is searched in the later half of a span


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


def vad_probabilities(audio: np.ndarray, vad, chunk_samples: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Speech probability per VAD window over the whole recording, and the
    window length in seconds. `vad` is a narrator.asr.vad.SileroVad (or
    anything with `process(frame) -> list[float]` over 512-sample windows)."""
    probs: list[float] = []
    for start in range(0, len(audio), chunk_samples):
        probs.extend(vad.process(audio[start : start + chunk_samples]))
    window_s = 512 / SAMPLE_RATE
    return np.asarray(probs, dtype=np.float32), window_s


def planned_speech_spans(
    probs: np.ndarray,
    window_s: float,
    duration_s: float,
    threshold: float = 0.5,
    max_span_s: float = MAX_SPAN_SECONDS,
) -> list[tuple[float, float]]:
    """Decode spans from per-window speech probabilities: speech regions
    (pauses under _MERGE_GAP_SECONDS merged, padded), each at most
    `max_span_s` long. A longer region is cut at the longest pause in the
    later half of the allowed length, else at its least speech-like window,
    so cuts fall between words. Silence-only stretches are not decoded at all
    (whisper invents text over silence)."""
    speech = probs >= threshold
    regions: list[list[float]] = []
    for i in np.flatnonzero(speech):
        t0, t1 = i * window_s, (i + 1) * window_s
        if regions and t0 - regions[-1][1] < _MERGE_GAP_SECONDS:
            regions[-1][1] = t1
        else:
            regions.append([t0, t1])
    spans: list[tuple[float, float]] = []
    for r0, r1 in regions:
        start = max(0.0, r0 - _SPEECH_PAD_SECONDS)
        end = min(duration_s, r1 + _SPEECH_PAD_SECONDS)
        while end - start > max_span_s:
            cut = _best_cut(probs, speech, window_s, start + max_span_s * _CUT_SEARCH_FROM, start + max_span_s)
            spans.append((start, cut))
            start = cut
        if end - start > _MIN_SPAN_SECONDS:
            spans.append((start, end))
    return spans


def _best_cut(probs: np.ndarray, speech: np.ndarray, window_s: float, lo_s: float, hi_s: float) -> float:
    lo = max(0, int(lo_s / window_s))
    hi = min(len(probs), int(hi_s / window_s))
    if hi <= lo:
        return hi_s
    # The longest run of non-speech windows wins (its middle); with none,
    # the least speech-like window.
    best_len, best_mid, run_start = 0, None, None
    for i in range(lo, hi + 1):
        quiet = i < hi and not speech[i]
        if quiet and run_start is None:
            run_start = i
        elif not quiet and run_start is not None:
            if i - run_start > best_len:
                best_len, best_mid = i - run_start, (run_start + i) / 2
            run_start = None
    if best_mid is not None:
        return best_mid * window_s
    return (lo + int(np.argmin(probs[lo:hi]))) * window_s + window_s / 2


def collapse_segment_text(text: str) -> str:
    """Whisper's in-segment repetition loops collapsed with the live
    decoder's own rule (case/punctuation-blind, up to 32-word phrases)."""
    from narrator.asr.streaming_decoder import _collapse_repetitions

    return " ".join(_collapse_repetitions(text.split()))


def segments_to_cues(
    segments: Sequence[tuple[float, float, str]],
    max_chars: int = 84,
    min_cue_s: float = 0.5,
) -> list[tuple[float, float, str]]:
    """Whisper segments -> subtitle cues: repetition loops collapsed, a
    segment repeating the previous one dropped (loops across segments), long
    segments split into <= max_chars cues with proportional times, cues
    ordered and non-overlapping."""
    cues: list[tuple[float, float, str]] = []
    previous_key = None
    for t0, t1, raw in sorted(segments, key=lambda s: s[0]):
        text = collapse_segment_text(raw.strip())
        if not text:
            continue
        key = "".join(ch for ch in text.lower() if ch.isalnum())
        if key and key == previous_key:
            continue
        previous_key = key
        chunks = _split_text(text, max_chars)
        total = sum(len(c) for c in chunks) or 1
        cursor = t0
        for chunk in chunks:
            share = (t1 - t0) * len(chunk) / total
            cues.append((cursor, cursor + share, chunk))
            cursor += share
    ordered: list[tuple[float, float, str]] = []
    last_end = 0.0
    for start, end, text in cues:
        start = max(start, last_end)
        end = max(end, start + min_cue_s)
        ordered.append((start, end, text))
        last_end = end
    return ordered


def _split_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        if current and len(" ".join(current + [word])) > max_chars:
            chunks.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks
