"""Offline ASR accuracy evaluation against an SRT reference.

Transcribes an audio file with the configured whisper model (pywhispercpp,
same library the app uses) and scores the hypothesis against a reference
SRT transcript.

Two reporting levels, chosen to match what the inputs can actually support:

  * whole-recording WER -- the truest accuracy number. Reference text is
    collapsed out of live-caption ROLLUP repeats first (subtitle feeds that
    re-show the previous line plus new words double every word otherwise --
    see `_collapse_rollup`), and comparison is case- and punctuation-
    insensitive while keeping Polish diacritics meaningful.

  * per-window diagnosis -- the SRT's own cue windows CANNOT be used for
    scoring: live cues overlap each other by design (rollup), and whisper
    segment timestamps are segment-level, so words of a 10s segment share
    one timestamp and blur across short cues -- scoring per cue produced
    WERs far above 100% with perfectly good text. Instead the recording is
    split into fixed 60-second buckets; the WORST bucket is then shown
    word-aligned, which localizes errors without inheriting the cue
    overlap artifact.

Bracketed non-speech cues ("[Muzyka]", "[Aplauz]") are used twice: dropped
from the reference, and applied as a MASK to the hypothesis -- whisper
transcribes music as hallucination loops ("dźwięk piosenki" x30 measured),
so hypothesis words timed inside an annotated non-speech window (±1s) are
music output, not missed speech. Non-Polish SPEECH cues (songs in this
recording's first minutes) are excluded the same way: whisper forced to
`pl` renders them as a Polish TRANSLATION, which would score as 100%
error against the English reference words even though it is a correct
translation (and the live-caption English reference itself is garbled:
"behavi like a", "ey when you close your ey"). Both reference and
hypothesis then go through the app's own repetition collapse
(`_collapse_repetitions`, the streaming text-layer fix) so speaker
stammers are treated identically on both sides.

Usage:
  python scripts/eval_transcript.py <audio> <reference.srt> [--language pl]
      [--model large-v3] [--start 0] [--end 99999] [--json out.json]

`--start/--end` (seconds) evaluate a slice -- useful for a quick pass over a
long recording before committing to the full decode.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass

import jiwer

# Bucket size for the localized diagnosis. Small enough to point at a
# specific passage, large enough that whisper's segment-level timestamps
# stay inside their bucket.
_WINDOW_SECONDS = 60.0


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str


_NON_SPEECH_RE = re.compile(r"^[\[\(].*[\]\)]$")


def parse_srt_timestamp(stamp: str) -> float:
    """'HH:MM:SS,mmm' -> seconds."""
    hh, mm, rest = stamp.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt(path: str) -> list[Cue]:
    """Minimal SRT parser: index line, timing line, text lines (possibly multi)."""
    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", open(path, encoding="utf-8-sig").read().strip())
    timing = re.compile(r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)")
    for block in blocks:
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        match = timing.search(lines[1])
        if match is None:
            continue
        text = " ".join(lines[2:]).strip()
        cues.append(
            Cue(
                index=int(lines[0]) if lines[0].isdigit() else len(cues) + 1,
                start=parse_srt_timestamp(match.group(1)),
                end=parse_srt_timestamp(match.group(2)),
                text=text,
            )
        )
    return cues


def _strip_punct(word: str) -> str:
    """Punctuation removal by Unicode category (P*): Polish diacritics pass
    through untouched -- "gwiazdę" vs "gwiazde" stays a visible, counted
    error. Applied per already-split word so nothing is joined first."""
    return "".join(ch for ch in word.lower() if not unicodedata.category(ch).startswith("P"))


def _words(text: str) -> list[str]:
    return [w for w in (_strip_punct(tok) for tok in text.split()) if w]


def collapse_reference_cues(
    cues: list[Cue],
) -> tuple[list[str], list[tuple[float, float]], list[tuple[float, float]], int]:
    """Flatten SPEECH, Polish-language cues into one word list with rollup
    echoes collapsed; also returns the non-speech windows AND the excluded
    out-of-scope speech windows (English songs -- both become hypothesis
    masks), plus the dropped-cue count."""
    kept: list[Cue] = []
    non_speech: list[tuple[float, float]] = []
    excluded_speech: list[tuple[float, float]] = []
    speech_cues: list[Cue] = []
    for c in cues:
        if _NON_SPEECH_RE.match(c.text.strip()):
            non_speech.append((c.start, c.end))
            continue
        speech_cues.append(c)

    for group in _split_cue_groups(speech_cues):
        if _is_foreign_group(group):
            excluded_speech.append((group[0].start, group[-1].end))
            continue
        kept.extend(group)
    words: list[str] = []
    for cue in kept:
        cue_words = _words(cue.text)
        if not cue_words:
            continue
        if words:
            # Drop the longest prefix of this cue that duplicates the tail
            # of the accumulated text (rollup echo of the previous line(s)).
            max_overlap = min(len(cue_words), len(words), 15)
            overlap = 0
            for k in range(max_overlap, 0, -1):
                if words[-k:] == cue_words[:k]:
                    overlap = k
                    break
            words.extend(cue_words[overlap:])
        else:
            words.extend(cue_words)
    return words, non_speech, excluded_speech, len(cues) - len(kept) - len(excluded_speech)


def _mask_music(
    hyp_timed: list[tuple[float, str]],
    mask_windows: list[tuple[float, float]],
    tolerance: float = 1.0,
) -> list[str]:
    """Keep only hypothesis words NOT timed inside a mask window (annotated
    non-speech stretches AND out-of-scope foreign-language speech -- see
    module docstring). Whisper sings along over music and translates over
    foreign speech; neither is a transcription error this script should
    count. A word counts as 'in' a window when its segment's start time
    falls within ±1s of it (segment timestamps are segment-level, so this
    is inherently coarse)."""
    if not mask_windows:
        return [w for _, w in hyp_timed]
    kept = []
    for t, w in hyp_timed:
        if any(s - tolerance <= t <= e + tolerance for s, e in mask_windows):
            continue
        kept.append(w)
    return kept


_MAX_SPAN_SECONDS = 60.0

# Out-of-scope detection: songs sung in English are excluded from scoring
# (see module docstring). The robust signal is DIACRITIC DENSITY, not word
# lists (a word list missed "cant remember color shining through" -- measured):
# Polish prose runs ~8-12% of words containing one of ą/ć/ę/ł/ń/ó/ś/ź/ż,
# English runs 0%. Consecutive speech cues are grouped (splits at caption
# gaps -- the [Aplauz] breaks around songs -- and at _FOREIGN_GROUP_MAX_S);
# a group with enough words and near-zero diacritics is foreign. Per-cue
# Polish diacritics still force a cue Polish regardless of its group.
_POLISH_CHARS = set("ąćęłńóśźż")
_FOREIGN_GROUP_MAX_S = 90.0
_FOREIGN_GROUP_SPLIT_GAP_S = 3.0
_FOREIGN_MIN_GROUP_WORDS = 12
_FOREIGN_MAX_DIACRITIC_FRACTION = 0.02


def _split_cue_groups(cues: list[Cue]) -> list[list[Cue]]:
    """Consecutive speech cues grouped for language detection: split when the
    gap between cues exceeds _FOREIGN_GROUP_SPLIT_GAP_S (captions pause around
    musical numbers) or when a group would run past _FOREIGN_GROUP_MAX_S."""
    groups: list[list[Cue]] = []
    current: list[Cue] = []
    for cue in cues:
        if current and (
            cue.start - current[-1].end > _FOREIGN_GROUP_SPLIT_GAP_S
            or cue.end - current[0].start > _FOREIGN_GROUP_MAX_S
        ):
            groups.append(current)
            current = []
        current.append(cue)
    if current:
        groups.append(current)
    return groups


def _is_foreign_group(group: list[Cue]) -> bool:
    """Diacritic-density test over the group's words; see the constants'
    comment. One Polish-diacritic word in a short group keeps it Polish."""
    words = [w for cue in group for w in _words(cue.text)]
    if len(words) < _FOREIGN_MIN_GROUP_WORDS:
        return False
    marked = sum(1 for w in words if any(ch in _POLISH_CHARS for ch in w))
    return marked / len(words) <= _FOREIGN_MAX_DIACRITIC_FRACTION


def _speech_spans(
    non_speech: list[tuple[float, float]], duration_s: float
) -> list[tuple[float, float]]:
    """Complement of the non-speech windows, split at _MAX_SPAN_SECONDS so one
    poisoned stretch can only ruin one chunk (narrator.asr.offline, shared
    with the recording draft pass)."""
    from narrator.asr.offline import speech_spans

    return speech_spans(non_speech, duration_s, max_span_s=_MAX_SPAN_SECONDS)


def load_audio_16k_mono(path: str) -> tuple["np.ndarray", float]:
    """Decode any GStreamer-supported audio file to mono 16kHz float32.

    Same route as the app's fixtures (there is no ffmpeg here): a
    GStreamer pipe, nothing written to disk. An earlier version wrote a
    temp WAV per call and never deleted it -- 41 leaked 170 MB files filled
    /tmp.
    """
    from pathlib import Path

    from narrator_tune.real_fixtures import SAMPLE_RATE, load_audio_16k_mono as decode

    return decode(Path(path)), float(SAMPLE_RATE)


def reference_cues(cues: list[Cue], start: float, end: float) -> list[Cue]:
    """All cues overlapping the evaluated window -- speech AND non-speech.
    The non-speech tags must survive this step: collapse_reference_cues()
    both drops them from the reference text and returns their windows as the
    hypothesis mask (earlier versions filtered here and silently lost the
    mask -- 0 windows for masking)."""
    kept = [c for c in cues if c.start < end and c.end > start]
    print(f"reference: {len(kept)} cues in window (non-speech split inside collapse_reference_cues)")
    return kept


def whole_recording_wer(ref_words: list[str], hyp_words: list[str]) -> float:
    return jiwer.wer(" ".join(ref_words), " ".join(hyp_words)) if ref_words else 0.0


def worst_window_diff(
    ref_cues: list[Cue], segments_plain: list[tuple[float, float, str]], window_seconds: float
) -> None:
    """Find the 60s window whose reference text matches worst, and show the
    word diff for it. Windows are time-based, not cue-based, so overlapping
    rollup cues and segment-level whisper timestamps cannot smear words
    across buckets; the ref text of a window uses the rollup-collapsed cue
    sequence (re-collapsed per window)."""
    if not ref_cues:
        return
    t0 = min(c.start for c in ref_cues)
    t1 = max(c.end for c in ref_cues)
    best = (-1.0, 0.0, [], [])
    for w_start in [t0 + i * window_seconds for i in range(int((t1 - t0) / window_seconds) + 1)]:
        w_end = w_start + window_seconds
        win_cues = [c for c in ref_cues if c.start < w_end and c.end > w_start]
        win_ref, _, _, _ = collapse_reference_cues(win_cues)
        if len(win_ref) < 10:
            continue
        win_hyp = [
            w
            for t0, _, text in segments_plain
            if w_start - 2 <= t0 <= w_end + 2
            for w in _words(text)
        ]
        score = jiwer.wer(" ".join(win_ref), " ".join(win_hyp)) if win_ref else 0.0
        if score > best[0]:
            best = (score, w_start, win_ref, win_hyp)

    score, w_start, win_ref, win_hyp = best
    print(f"\n=== worst {int(window_seconds)}s window (starts {int(w_start // 60)}:{int(w_start % 60):02d}) WER {score*100:.1f}% ===")
    ref_line = " ".join(win_ref)
    hyp_line = " ".join(win_hyp)
    matcher = difflib.SequenceMatcher(None, ref_line.split(), hyp_line.split())
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_part = " ".join(ref_line.split()[i1:i2])
        hyp_part = " ".join(hyp_line.split()[j1:j2])
        if tag == "equal":
            print(f"  = {ref_part}")
        elif tag == "replace":
            print(f"  - {ref_part}")
            print(f"  + {hyp_part}")
        elif tag == "delete":
            print(f"  - {ref_part}   (missed)")
        elif tag == "insert":
            print(f"  + {hyp_part}   (hallucinated/extra)")


def _decode_all(args, non_speech: list[tuple[float, float]], excluded_speech: list[tuple[float, float]]) -> list:
    """Decode every in-scope speech span with a fresh whisper context per span.

    Out-of-scope windows (music/applause annotations AND foreign-language
    speech -- see module docstring) are span boundaries too: a song decoded
    as part of a span poisons the whole span's context carryover (measured:
    the concert's first minutes turned every later chunk into "dźwięk
    piosenki" loops), while a fresh context confines the damage to the
    poisoned stretch.
    """
    from pywhispercpp.model import Model

    audio, rate = load_audio_16k_mono(args.audio)
    lo = int(args.start * rate)
    hi = int(min(args.end, len(audio) / rate) * rate)
    audio = audio[lo:hi]
    duration_s = len(audio) / rate
    print(f"  {duration_s / 60:.1f} min of audio; transcribing with {args.model} ({args.language}) ...", flush=True)

    # Mask windows arrive in absolute SRT time; shift into slice-local time
    # (clamped -- a window may start before the slice).
    offset = args.start
    windows = [
        (max(s - offset, 0.0), min(e - offset, duration_s))
        for s, e in non_speech + excluded_speech
        if e - offset > 0 and s - offset < duration_s
    ]
    from narrator.paths import model_load_args

    model_name, models_dir = model_load_args(args.model)
    model = Model(model_name, models_dir=models_dir, language=args.language)
    spans = _speech_spans(windows, duration_s)
    print(f"  decoding {len(spans)} speech spans (bounded at {_MAX_SPAN_SECONDS:.0f}s each)")

    segments_plain = []
    for span_start, span_end in spans:
        chunk = audio[int(span_start * rate) : int(span_end * rate)]
        if len(chunk) < rate // 10:
            continue
        try:
            segs = model.transcribe(chunk, language=args.language)
        except Exception as exc:
            print(f"  decode failed on span {span_start:.1f}-{span_end:.1f}s: {exc}", file=sys.stderr)
            continue
        for seg in segs:
            segments_plain.append(
                (seg.t0 / 100.0 + span_start, seg.t1 / 100.0 + span_start, seg.text)
            )
    return segments_plain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("reference_srt")
    parser.add_argument("--language", default="pl")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--start", type=float, default=0.0, help="evaluate from this offset (s)")
    parser.add_argument("--end", type=float, default=1e9, help="evaluate up to this offset (s)")
    parser.add_argument("--json", default=None, help="write hypothesis + scores to this file")
    parser.add_argument(
        "--reuse-segments", default=None,
        help="skip decoding: load the 'segments' array from a previous --json run and rescore only",
    )
    args = parser.parse_args()

    cues = parse_srt(args.reference_srt)
    ref_cues = reference_cues(cues, args.start, args.end)
    if not ref_cues:
        print("no speech cues in the requested window", file=sys.stderr)
        return 2

    ref_words, non_speech, excluded_speech, _ = collapse_reference_cues(ref_cues)
    mask_windows = non_speech + excluded_speech
    print(
        f"reference text after rollup collapse: {len(ref_words)} words; "
        f"{len(non_speech)} non-speech + {len(excluded_speech)} foreign-speech windows for masking"
    )

    if args.reuse_segments:
        with open(args.reuse_segments, encoding="utf-8") as f:
            all_segments = [tuple(seg) for seg in json.load(f)["segments"]]
        # Keep only segments overlapping the evaluated window: a --json from a
        # full run covers far more than a --start/--end slice would.
        segments_plain = [
            (t0, t1, text)
            for t0, t1, text in all_segments
            if t1 > args.start and t0 < args.end
        ]
        print(f"reusing {len(segments_plain)}/{len(all_segments)} decoded segments from {args.reuse_segments}")
    else:
        segments_plain = _decode_all(args, non_speech, mask_windows)
    hyp_timed = [(t0, w) for t0, _, text in segments_plain for w in _words(text)]

    from narrator.asr.streaming_decoder import _collapse_repetitions

    raw_count = len(hyp_timed)
    hyp_words = _mask_music(hyp_timed, mask_windows)
    masked_count = raw_count - len(hyp_words)
    hyp_words = _collapse_repetitions(hyp_words)
    ref_words = _collapse_repetitions(ref_words)  # same normalization on both sides
    print(f"  {raw_count} hypothesis words; {masked_count} dropped by the non-speech/foreign mask; "
          f"{len(hyp_words)} after repetition collapse (ref: {len(ref_words)})")

    wer = whole_recording_wer(ref_words, hyp_words)
    print("\n=== whole recording (rollup-collapsed reference) ===")
    print(f"WER: {wer * 100:.1f}%   (ref {len(ref_words)} words, hyp {len(hyp_words)} words)")

    worst_window_diff(ref_cues, segments_plain, _WINDOW_SECONDS)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "whole_file_wer": wer,
                    "reference_words": len(ref_words),
                    "hypothesis_words": len(hyp_words),
                    "hypothesis_text": " ".join(hyp_words),
                    "segments": segments_plain,
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        print(f"\ndetail written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
