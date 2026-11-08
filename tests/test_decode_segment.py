"""Phase 1 decode core (todo B): WhisperBackend.decode_segment and its
helpers -- padding, the token filter (special tokens by the model's own
eot, the trailing period of a mid-speech cut), byte-joined text, the
quality gates -- with a fake whisper context instead of the GPU."""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from narrator.asr.backend import (
    SegmentToken,
    WhisperBackend,
    filter_segment_tokens,
    pad_segment,
    segment_text,
)
from narrator.config.schema import AsrConfig

EOT = 51865  # large-v3-like: NOT the 50256 of the English GPT-2 vocabulary
BEG = EOT + 100  # first timestamp token


def tok(text, id_=100, p=0.9):
    return SegmentToken(id=id_, text=text, p=p, raw=text.encode("utf-8"))


# -- padding -----------------------------------------------------------------


def test_segments_under_50ms_are_not_decoded():
    assert pad_segment(np.zeros(799, dtype=np.float32)) is None
    assert pad_segment(np.zeros(800, dtype=np.float32)) is not None


def test_short_segments_are_centred_in_1_01s_of_faint_noise_plus_edges():
    audio = np.full(8000, 0.5, dtype=np.float32)  # 0.5 s
    padded = pad_segment(audio)
    assert len(padded) == 16160 + 2 * 160  # 1.01 s + 10 ms each side
    edge = padded[:160]
    assert not edge.any() and not padded[-160:].any()
    body = padded[160:-160]
    start = (16160 - 8000) // 2
    assert (body[start : start + 8000] == 0.5).all()
    noise = np.concatenate([body[:start], body[start + 8000 :]])
    assert np.abs(noise).max() <= 0.01 and np.abs(noise).max() > 0
    assert np.array_equal(padded, pad_segment(audio))  # a replay pads identically


def test_long_segments_get_only_the_silent_edges():
    audio = np.full(32000, 0.25, dtype=np.float32)
    padded = pad_segment(audio)
    assert len(padded) == 32000 + 320
    assert (padded[160:-160] == 0.25).all()


# -- token filter --------------------------------------------------------------


def test_special_tokens_dropped_by_the_models_eot():
    tokens = [tok(" Ala"), tok(" ma"), SegmentToken(id=EOT, text="", p=1.0), SegmentToken(id=BEG + 5, text="", p=1.0),
              SegmentToken(id=50256, text=" kota", p=0.8, raw=b" kota")]
    kept = filter_segment_tokens(tokens, eot=EOT, drop_trailing_period=False)
    # 50256 is an ordinary text token in large-v3's vocabulary: kept
    assert [t.text for t in kept] == [" Ala", " ma", " kota"]


@pytest.mark.parametrize("last", [".", " .", "...", "…"])
def test_trailing_period_dropped_only_when_asked(last):
    tokens = [tok(" Ala"), tok(" ma"), tok(last), SegmentToken(id=EOT, text="", p=1.0)]
    assert [t.text for t in filter_segment_tokens(tokens, eot=EOT, drop_trailing_period=True)] == [" Ala", " ma"]
    assert len(filter_segment_tokens(tokens, eot=EOT, drop_trailing_period=False)) == 3


def test_inner_periods_and_other_punctuation_stay():
    tokens = [tok(" Tak"), tok("."), tok(" Nie"), tok("?")]
    kept = filter_segment_tokens(tokens, eot=EOT, drop_trailing_period=True)
    assert [t.text for t in kept] == [" Tak", ".", " Nie", "?"]


def test_text_joins_bytes_so_split_polish_letters_survive():
    # "ż" (c5 bc) split across two tokens
    tokens = [SegmentToken(id=1, text="", p=0.9, raw=b" mo\xc5"), SegmentToken(id=2, text="", p=0.9, raw=b"\xbce")]
    assert segment_text(tokens, strip_brackets=True) == "może"


def test_text_strips_bracketed_annotations_when_asked():
    tokens = [tok(" [oklaski]"), tok(" Dzię"), tok("kuję"), tok("  [muzyka] ")]
    assert segment_text(tokens, strip_brackets=True) == "Dziękuję"
    assert "[oklaski]" in segment_text(tokens, strip_brackets=False)


# -- decode_segment end to end over a fake whisper context ---------------------


@dataclass
class _Data:
    id: int
    p: float
    t0: int = 0
    t1: int = 0


class _FakeModel:
    def __init__(self, segments):
        self.segments = segments  # list of lists of (id, bytes, p)
        self.calls = []
        self._ctx = object()

    def transcribe(self, audio, **kwargs):
        self.calls.append((len(audio), kwargs))
        return []


def _install_fake_pw(monkeypatch, model):
    vocab = {}
    for segment in model.segments:
        for id_, piece, _ in segment:
            vocab[id_] = piece
    pw = types.SimpleNamespace(
        whisper_token_eot=lambda ctx: EOT,
        whisper_token_beg=lambda ctx: BEG,
        whisper_full_n_segments=lambda ctx: len(model.segments),
        whisper_full_n_tokens=lambda ctx, i: len(model.segments[i]),
        whisper_full_get_token_data=lambda ctx, i, j: _Data(model.segments[i][j][0], model.segments[i][j][2], 10 * j, 10 * j + 10),
        whisper_token_to_bytes=lambda ctx, id_: vocab[id_],
    )
    monkeypatch.setitem(sys.modules, "_pywhispercpp", pw)


def _backend(config, model):
    backend = WhisperBackend.__new__(WhisperBackend)
    backend._config = config
    backend._model = model
    backend._vocabulary = []
    backend.rejected_decodes = {}
    backend._dtw = False
    return backend


def _segments(*words, p=0.9):
    return [[(100 + i, w.encode(), p) for i, w in enumerate(words)] + [(EOT, b"", 1.0)]]


def test_decode_segment_pads_decodes_and_filters(monkeypatch):
    model = _FakeModel(_segments(" Ala", " ma", " kota", "."))
    _install_fake_pw(monkeypatch, model)
    backend = _backend(AsrConfig(language_whitelist=["pl"]), model)
    audio = np.zeros(8000, dtype=np.float32)
    result = backend.decode_segment(audio, prompt="Wcześniej.", language="pl", drop_trailing_period=True)
    assert result.text == "Ala ma kota"
    assert result.language == "pl" and result.rejected is None
    assert [t.text for t in result.tokens] == [" Ala", " ma", " kota"]
    n, kwargs = model.calls[0]
    assert n == 16160 + 320  # padded to 1.01 s + edges
    assert kwargs["temperature_inc"] == 0.0 and kwargs["initial_prompt"] == "Wcześniej."
    assert kwargs["language"] == "pl" and "audio_ctx" in kwargs
    # the prompt is set on EVERY call (pywhispercpp keeps params between calls)
    backend.decode_segment(audio, language="pl")
    assert model.calls[1][1]["initial_prompt"] == ""


def test_decode_segment_keeps_a_sentence_end_at_a_pause(monkeypatch):
    model = _FakeModel(_segments(" Koniec", "."))
    _install_fake_pw(monkeypatch, model)
    backend = _backend(AsrConfig(language_whitelist=["pl"]), model)
    assert backend.decode_segment(np.zeros(16000, np.float32), language="pl").text == "Koniec."


def test_decode_segment_too_short_is_not_decoded(monkeypatch):
    model = _FakeModel(_segments(" x"))
    _install_fake_pw(monkeypatch, model)
    backend = _backend(AsrConfig(language_whitelist=["pl"]), model)
    assert backend.decode_segment(np.zeros(400, np.float32), language="pl").text == ""
    assert model.calls == []


def test_decode_segment_confidence_gate(monkeypatch):
    model = _FakeModel(_segments(" mamrot", " mamrot", p=0.2))
    _install_fake_pw(monkeypatch, model)
    backend = _backend(AsrConfig(language_whitelist=["pl"], min_mean_token_p=0.4), model)
    result = backend.decode_segment(np.zeros(16000, np.float32), language="pl")
    assert result.text == "" and result.rejected.startswith("mean token p")
    assert backend.rejected_decodes == {"mean": 1}


def test_decode_segment_hallucination_guard(monkeypatch):
    # a timestamp token at 6 s inside 2 s of audio: ratio 3 > 2.25
    model = _FakeModel([[(100, b" loop", 0.9), (BEG + 300, b"", 1.0), (EOT, b"", 1.0)]])
    _install_fake_pw(monkeypatch, model)
    backend = _backend(AsrConfig(language_whitelist=["pl"], max_timestamp_ratio=2.25), model)
    result = backend.decode_segment(np.zeros(32000, np.float32), language="pl")
    assert result.text == "" and result.rejected is not None
