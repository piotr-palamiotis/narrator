"""Shared transcript scoring: normalization, WER, word-doubling detection,
plus the 16kHz mono WAV loader used by the autoconfigure bench/probe
subprocesses (piper fixtures are 22.05kHz; the live pipeline resamples in the
capture path, but direct file loading must resample itself).
"""
from __future__ import annotations

import re
import unicodedata
import wave
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
