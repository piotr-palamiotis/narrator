"""Shared transcript scoring: normalization, WER, word-doubling detection,
word alignment with an edit breakdown, plus the 16kHz mono WAV loader used
by the autoconfigure bench/probe subprocesses (piper fixtures are 22.05kHz;
the live pipeline resamples in the capture path, but direct file loading
must resample itself).
"""
from __future__ import annotations

import re
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_DOUBLED_WORD_RE = re.compile(r"\b(\w+) \1\b", re.UNICODE | re.IGNORECASE)


def load_wav_16k_mono(path: Path, target_rate: int = 16000) -> np.ndarray:
    """Load any s16 WAV as float32 mono at `target_rate`, resampling if needed."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        from math import gcd

        g = gcd(int(rate), int(target_rate))
        audio = resample_poly(audio, target_rate // g, rate // g).astype(np.float32)
    return audio


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (NFC first)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein distance on normalized word lists."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[len(hyp)] / len(ref)


def has_word_doubling(text: str) -> bool:
    """True if the text contains an immediate repeated word ("niż niż")."""
    return _DOUBLED_WORD_RE.search(text) is not None


@dataclass(frozen=True)
class Alignment:
    """Word-level Levenshtein alignment of a hypothesis to a reference."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    # (reference index, hypothesis index) of every matched or substituted pair
    pairs: tuple[tuple[int, int], ...]
    # hypothesis indices that are insertions
    inserted: tuple[int, ...]

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def align(reference: list[str], hypothesis: list[str]) -> Alignment:
    """Minimum-edit alignment of two word lists, with the edit breakdown.

    Plain Wagner-Fischer with a backtrace -- O(len(ref) * len(hyp)), which is
    fine for excerpt-sized texts (a few hundred words). Ties prefer the
    diagonal, so a substitution is never reported as a deletion plus an
    insertion.
    """
    n, m = len(reference), len(hypothesis)
    # Integer ids make the inner comparison cheap.
    ids: dict[str, int] = {}
    ref = [ids.setdefault(w, len(ids)) for w in reference]
    hyp = [ids.setdefault(w, len(ids)) for w in hypothesis]
    dist = [list(range(m + 1))]
    for i in range(1, n + 1):
        prev = dist[-1]
        cur = [i] + [0] * m
        r = ref[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (r != hyp[j - 1])
            up = prev[j] + 1
            left = cur[j - 1] + 1
            cur[j] = diag if diag <= up and diag <= left else (up if up <= left else left)
        dist.append(cur)

    subs = dels = ins = 0
    pairs: list[tuple[int, int]] = []
    inserted: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                subs += 1
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            inserted.append(j - 1)
            j -= 1
    pairs.reverse()
    inserted.reverse()
    return Alignment(subs, dels, ins, n, tuple(pairs), tuple(inserted))
