"""Unit tests for autoconfigure's pure decision logic (no GPU, no subprocess)."""
from __future__ import annotations

import pytest

from narrator.asr.autoconfigure import (
    SLOW_FACTOR,
    STEP_MAX_MS,
    STEP_MIN_MS,
    SPEED_REFERENCE_S,
    WER_FLOOR,
    BenchResult,
    apply_decision,
    decide,
    derive_step_ms,
    grade_candidate,
)
from narrator.config.loader import load_config, save_config


def make_result(model: str, s10: float | None, wer: float | None = None, **kwargs) -> BenchResult:
    """Build a BenchResult with decode times scaling proportionally to window
    length (2s = s10/5, 5s = s10/2), the shape real models exhibit."""
    windows: dict[str, float] = {}
    if s10 is not None:
        windows = {"2.0s": round(s10 * 0.2, 3), "5.0s": round(s10 * 0.5, 3), "10.0s": s10}
    return BenchResult(
        model=model,
        device=kwargs.pop("device", "auto"),
        resolved_device=kwargs.pop("resolved_device", "Vulkan (Test GPU)"),
        decode_seconds_by_window=windows,
        wer=wer,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# grade_candidate
# ---------------------------------------------------------------------------


def test_qualified_fast_accurate_scores_highest():
    fast = make_result("fast", s10=0.3, wer=0.05)
    grade_candidate(fast)
    assert fast.qualified
    # speed term: 0.6/0.3 = 1.0 (capped); accuracy: 1 - 0.05/0.30 = 0.833
    assert fast.score == pytest.approx(0.7 * 1.0 + 0.3 * (1 - 0.05 / 0.30))


def test_speed_term_scales_with_decode_time():
    at_reference = make_result("ref", s10=SPEED_REFERENCE_S, wer=0.0)
    double = make_result("slow2x", s10=SPEED_REFERENCE_S * 2, wer=0.0)
    grade_candidate(at_reference)
    grade_candidate(double)
    assert at_reference.score > double.score
    # exactly at the reference speed, the speed term is 1.0
    assert at_reference.score == pytest.approx(0.7 + 0.3)


def test_disqualified_when_decode_overruns_cadence():
    # 10s window decoded in 20s -> ratio 2.0x > 1.5x
    slow = make_result("slow", s10=20.0, wer=0.05)
    grade_candidate(slow)
    assert not slow.qualified
    assert slow.score == float("-inf")


def test_ratio_boundary_at_slow_factor_qualifies():
    exactly = make_result("edge", s10=SLOW_FACTOR * 10.0, wer=0.05)
    grade_candidate(exactly)
    assert exactly.qualified  # boundary is inclusive


def test_disqualified_when_wer_over_floor():
    bad = make_result("garbled", s10=0.3, wer=WER_FLOOR * 2)
    grade_candidate(bad)
    assert not bad.qualified


def test_doubled_words_disqualify_even_with_good_speed_and_wer():
    """Doubled words in the probe's FINAL text mean the streaming decoder's
    repetition collapse is absent or broken -- the measurement does not
    describe the shipping decoder, so the candidate cannot be ranked."""
    doubled = make_result("loopy", s10=0.2, wer=0.02, doubled=True)
    grade_candidate(doubled)
    assert not doubled.qualified
    assert doubled.score == float("-inf")


def test_wer_at_floor_scores_zero_accuracy_but_qualifies():
    edge = make_result("edge", s10=0.3, wer=WER_FLOOR)
    grade_candidate(edge)
    assert edge.qualified
    assert edge.score == pytest.approx(0.7 * 1.0 + 0.3 * 0.0)


def test_missing_wer_penalized_but_can_qualify():
    # Unmeasured WER is treated as at-the-floor: speed can still carry it.
    unknown = make_result("unknown", s10=0.2, wer=None)
    grade_candidate(unknown)
    assert unknown.qualified
    assert unknown.score == pytest.approx(0.7 * 1.0 + 0.3 * 0.0)


def test_error_result_never_qualifies():
    broken = make_result("broken", s10=None)
    broken.error = "boom"
    grade_candidate(broken)
    assert not broken.qualified


def test_slowest_ratio_uses_worst_window():
    r = make_result("mixed", s10=0.5)
    r.decode_seconds_by_window["10.0s"] = 1.0  # 0.1x real time
    r.decode_seconds_by_window["5.0s"] = 8.0  # 1.6x real time -- worst
    assert r.slowest_decode_ratio == pytest.approx(8.0 / 5.0)


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------


def test_decide_prefers_faster_qualifier():
    medium = make_result("medium", s10=0.55, wer=0.08)
    large = make_result("large-v3", s10=0.7, wer=0.02)
    for r in (medium, large):
        grade_candidate(r)
    decision = decide([large, medium])
    assert decision.model == "medium"


def test_decide_prefers_more_accurate_at_equal_speed():
    a = make_result("a", s10=0.5, wer=0.05)
    b = make_result("b", s10=0.5, wer=0.10)
    for r in (a, b):
        grade_candidate(r)
    assert decide([a, b]).model == "a"


def test_decide_skips_disqualified_even_if_accurate():
    # A fast, accurate candidate that overruns the cadence on some window
    # must not be chosen: that is the queue-saturation regime.
    large = make_result("large-v3", s10=12.0, wer=0.02)  # ratio 1.2x... use worse
    large.decode_seconds_by_window = {"10.0s": 20.0, "5.0s": 8.0, "2.0s": 0.3}
    medium = make_result("medium", s10=0.55, wer=0.08)
    grade_candidate(large)
    grade_candidate(medium)
    assert not large.qualified
    assert decide([large, medium]).model == "medium"


def test_decide_prefers_more_accurate_when_both_at_speed_cap():
    """On capable hardware both models cap the speed term (both decode a 10s
    window well under the 0.6s reference), so accuracy decides -- the
    post-collapse-fix regime, where large-v3 can legitimately win."""
    large = make_result("large-v3", s10=0.58, wer=0.02)
    medium = make_result("medium", s10=0.33, wer=0.08)
    for r in (large, medium):
        grade_candidate(r)
    assert large.qualified and medium.qualified
    assert decide([medium, large]).model == "large-v3"


def test_decide_raises_when_nothing_qualifies():
    bad = make_result("bad", s10=30.0, wer=0.9)
    grade_candidate(bad)
    with pytest.raises(RuntimeError, match="no ASR model qualified"):
        decide([bad])


def test_decide_carries_candidates_for_reporting():
    medium = make_result("medium", s10=0.55, wer=0.08)
    grade_candidate(medium)
    decision = decide([medium])
    assert decision.candidates == [medium]


# ---------------------------------------------------------------------------
# derive_step_ms
# ---------------------------------------------------------------------------


def test_step_ms_scales_with_decode_and_clamps():
    fast = derive_step_ms({"10.0s": 0.2})  # 0.2*1000*1.35 = 270 -> 250 floor region
    typical = derive_step_ms({"10.0s": 0.55})  # 742 -> 750
    slow = derive_step_ms({"10.0s": 5.0})  # far over ceiling
    assert STEP_MIN_MS <= fast <= STEP_MAX_MS
    assert typical == 750
    assert slow == STEP_MAX_MS


def test_step_ms_floor_when_window_missing():
    assert derive_step_ms({}) == STEP_MIN_MS


def test_step_ms_rounded_to_50ms():
    value = derive_step_ms({"10.0s": 0.51})  # 688.5 -> rounds to 700
    assert value % 50 == 0


# ---------------------------------------------------------------------------
# apply_decision
# ---------------------------------------------------------------------------


def test_apply_decision_updates_only_asr_fields(tmp_path):
    from narrator.config.schema import RootConfig

    config_path = tmp_path / "config.yaml"
    save_config(RootConfig(), config_path)

    medium = make_result("medium", s10=0.55, wer=0.08)
    grade_candidate(medium)
    decision = decide([medium])
    updated = apply_decision(config_path, decision)

    assert updated.asr.model == "medium"
    assert updated.asr.step_ms == decision.step_ms
    # untouched fields survive the round-trip
    assert updated.canvas.width_px == 1920
    assert updated.layout.max_lines == 2

    reloaded = load_config(config_path)
    assert reloaded.asr.model == "medium"
    assert reloaded.asr.step_ms == decision.step_ms
