"""obs-localvocal's decode quality gates and annotation stripping (backend.py)."""
from narrator.asr.backend import decode_reject_reason, strip_bracketed_annotations

EOT = 50257
BEG = 50365  # large-v3's first timestamp token, <|0.00|>


def ts(seconds: float) -> tuple[int, float]:
    return (BEG + int(round(seconds / 0.02)), 1.0)


def reason(tokens, audio_seconds=4.0, min_p=0.0, max_ratio=0.0):
    return decode_reject_reason(
        tokens, eot=EOT, timestamp_begin=BEG, audio_seconds=audio_seconds,
        min_mean_token_p=min_p, max_timestamp_ratio=max_ratio,
    )


def test_gates_off_keep_everything():
    assert reason([ts(0.0), (100, 0.01), ts(60.0)]) is None


def test_timestamp_past_the_audio_is_a_hallucination():
    tokens = [ts(0.0), (100, 0.9), (101, 0.9), ts(10.0)]
    assert reason(tokens, audio_seconds=4.0, max_ratio=2.25).startswith("timestamp")
    assert reason(tokens, audio_seconds=5.0, max_ratio=2.25) is None


def test_timestamp_ratio_measured_against_at_least_one_second():
    # The encoder context never shrinks below ~5s, so a short window can end
    # on a timestamp well past its own length.
    tokens = [ts(0.0), (100, 0.9), ts(2.0)]
    assert reason(tokens, audio_seconds=0.5, max_ratio=2.25) is None


def test_low_mean_token_probability_is_rejected():
    tokens = [ts(0.0), (100, 0.2), (101, 0.3), ts(1.0)]
    assert reason(tokens, min_p=0.4).startswith("mean")
    assert reason(tokens, min_p=0.2) is None


def test_special_and_timestamp_tokens_do_not_count_toward_confidence():
    # Specials (between eot and the timestamps) and timestamps carry their
    # own probabilities; only text tokens are averaged.
    tokens = [(EOT + 1, 0.01), ts(0.0), (100, 0.9), ts(1.0), (EOT, 0.01)]
    assert reason(tokens, min_p=0.5) is None


def test_empty_hypothesis_is_never_rejected():
    assert reason([], min_p=0.9, max_ratio=1.0) is None


def test_strip_bracketed_annotations():
    assert strip_bracketed_annotations("[oklaski]") == ""
    assert strip_bracketed_annotations(" Dzień dobry [muzyka] wszystkim. ") == "Dzień dobry wszystkim."
    assert strip_bracketed_annotations("[wypowiedź w języku ukraińskim] Tak.") == "Tak."
    assert strip_bracketed_annotations("bez nawiasów") == "bez nawiasów"
