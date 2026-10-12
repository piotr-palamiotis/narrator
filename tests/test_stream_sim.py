"""Virtual-clock replay (narrator.asr.stream_sim): live scheduling semantics
without a GPU -- fake VAD/backend, constant decode cost model."""
from __future__ import annotations

import numpy as np
import pytest

from narrator.asr.stream_sim import DecodeCostModel, SimResult, simulate_stream
from narrator.config.schema import AsrConfig

FRAME = 480  # 30 ms @ 16 kHz


class ScheduledVad:
    """One probability per feed() call: speech for the first `speech_frames`
    calls, silence afterwards."""

    def __init__(self, speech_frames: int) -> None:
        self.speech_frames = speech_frames
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        return [1.0 if self.calls <= self.speech_frames else 0.0]


class EchoBackend:
    """Transcribes a window as one word per started second of audio, so the
    hypothesis is stable and grows with the window."""

    resolved_device_description = "fake"

    def __init__(self) -> None:
        self.transcribe_calls = 0

    def transcribe(self, audio, language=None, **kwargs):
        self.transcribe_calls += 1
        seconds = int(len(audio) / 16000)
        return [(0.0, float(seconds), " ".join(f"w{i}" for i in range(seconds)))] if seconds else []

    def detect_language(self, audio):
        return ("pl", 0.99, 0.9)


def config(**overrides) -> AsrConfig:
    defaults = dict(
        language_whitelist=["pl"],
        step_ms=300,
        agreement_passes=2,
        end_silence_ms=300,
        max_utterance_ms=25000,
        max_context_seconds=20.0,
    )
    defaults.update(overrides)
    return AsrConfig(**defaults)


def speech(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def run(cost: DecodeCostModel, speech_s: float = 6.0, total_s: float | None = None, **cfg):
    audio = speech(total_s if total_s is not None else speech_s)
    backend = EchoBackend()
    vad = ScheduledVad(int(speech_s * 1000 / 30))
    result = simulate_stream(backend, config(**cfg), audio, cost, vad=vad)
    return result, backend


def test_decode_cadence_follows_step_when_decodes_are_fast():
    result, _ = run(DecodeCostModel(base_s=0.05, per_audio_s=0.0), step_ms=300)
    # ~6 s of speech at one decode per 300 ms (the first waits for 0.3 s of
    # audio), plus the finalize decode
    assert 17 <= result.decode_calls <= 22


def test_decode_cadence_follows_decode_time_when_decodes_are_slow():
    result, _ = run(DecodeCostModel(base_s=0.6, per_audio_s=0.0), step_ms=300)
    # the worker is busy 0.6 s per decode, so it can decode ~every 0.6 s
    assert 8 <= result.decode_calls <= 12


def test_busy_fraction_reflects_cost():
    result, _ = run(DecodeCostModel(base_s=0.6, per_audio_s=0.0), step_ms=300)
    assert result.busy_seconds == pytest.approx(0.6 * result.decode_calls)
    assert 0.8 < result.busy_fraction <= 1.1


def test_finals_are_stamped_with_decode_completion_time():
    cost = DecodeCostModel(base_s=0.5, per_audio_s=0.0)
    result, _ = run(cost, speech_s=3.0, total_s=5.0, step_ms=300)
    assert result.finals, "the finalize decode must emit the utterance"
    # nothing can be shown before the first decode could have finished
    first_emit = result.finals[0][0]
    assert first_emit >= 0.3 + cost.base_s - 1e-9
    # emit times never decrease
    times = [t for t, _ in result.finals]
    assert times == sorted(times)


def test_finalize_decode_is_charged_after_silence():
    cost = DecodeCostModel(base_s=0.2, per_audio_s=0.0)
    result, _ = run(cost, speech_s=3.0, total_s=6.0, step_ms=300)
    # speech ends at 3.0 s and the utterance finalizes 300 ms of silence
    # later: its last words appear no earlier than one decode after speech
    # ends, and no later than the finalize decode (which may first wait for
    # a decode already in flight)
    last_emit = result.finals[-1][0]
    assert 3.0 + cost.base_s - 0.05 <= last_emit <= 3.3 + 2 * cost.base_s + 0.05


def test_trailing_utterance_is_flushed_without_explicit_silence():
    result, _ = run(DecodeCostModel(base_s=0.05, per_audio_s=0.0), speech_s=4.0, total_s=4.0)
    words = result.text.split()
    assert words, "speech running to the end of the audio must still be captioned"
    assert words[-1] == "w3"


def test_replay_is_deterministic():
    cost = DecodeCostModel(base_s=0.3, per_audio_s=0.02)
    first, _ = run(cost)
    second, _ = run(cost)
    assert first.finals == second.finals
    assert first.decode_calls == second.decode_calls


def test_cost_model_required_unless_measured():
    with pytest.raises(ValueError):
        simulate_stream(EchoBackend(), config(), speech(1.0), None, vad=ScheduledVad(10))
    result = simulate_stream(EchoBackend(), config(), speech(1.0), None, measured=True, vad=ScheduledVad(10))
    assert isinstance(result, SimResult)


def test_cost_model_fit_recovers_a_linear_cost():
    timings = {"2.0s": 0.1 + 0.05 * 2, "5.0s": 0.1 + 0.05 * 5, "10.0s": 0.1 + 0.05 * 10, "20.0s": 0.1 + 0.05 * 20}
    model = DecodeCostModel.fit(timings, detect_s=0.18)
    assert model.base_s == pytest.approx(0.1, abs=1e-3)
    assert model.per_audio_s == pytest.approx(0.05, abs=1e-4)
    assert model.detect_s == pytest.approx(0.18)
    assert model.decode_seconds(10.0) == pytest.approx(0.6, abs=1e-3)


def test_cost_model_fit_never_goes_negative():
    # noisy short windows can produce a negative intercept: refit through 0
    model = DecodeCostModel.fit({"2.0s": 0.01, "10.0s": 1.0})
    assert model.base_s >= 0.0
    assert model.per_audio_s > 0.0


def test_sim_result_round_trips_through_dict():
    result, _ = run(DecodeCostModel(base_s=0.05, per_audio_s=0.0))
    again = SimResult.from_dict(result.to_dict())
    assert again.text == result.text
    assert again.decode_calls == result.decode_calls


def test_cost_model_fit_calls_recovers_window_and_word_terms():
    calls = [
        (w, n, 0.05 + 0.02 * w + 0.01 * n)
        for w in (2.0, 5.0, 10.0, 15.0, 20.0)
        for n in (0, 10, 30, 50)
    ]
    model = DecodeCostModel.fit_calls(calls, detect_s=0.3)
    assert model.base_s == pytest.approx(0.05, abs=1e-3)
    assert model.per_audio_s == pytest.approx(0.02, abs=1e-4)
    assert model.per_word_s == pytest.approx(0.01, abs=1e-4)
    assert model.decode_seconds(10.0, 20) == pytest.approx(0.45, abs=1e-3)


def test_cost_model_fit_calls_keeps_coefficients_non_negative():
    # longer windows here emit fewer words, so an unconstrained fit would
    # give the word term a negative weight
    calls = [(2.0, 40, 0.2), (5.0, 30, 0.35), (10.0, 20, 0.6), (20.0, 5, 1.1)]
    model = DecodeCostModel.fit_calls(calls, detect_s=0.0)
    assert model.base_s >= 0 and model.per_audio_s >= 0 and model.per_word_s >= 0
    assert model.decode_seconds(10.0, 20) == pytest.approx(0.6, abs=0.1)


def test_measured_mode_records_per_call_timings():
    result = simulate_stream(EchoBackend(), config(), speech(2.0), None, measured=True, vad=ScheduledVad(60))
    assert len(result.calls) == result.decode_calls
    assert all(window > 0 and seconds >= 0 for window, _, seconds in result.calls)
    again = SimResult.from_dict(result.to_dict())
    assert again.calls == result.calls
