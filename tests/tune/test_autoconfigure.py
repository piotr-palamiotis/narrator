"""Autoconfigure's decision logic and staged search, without GPU or
subprocesses: a fake evaluator scores configurations from a formula with a
known optimum."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from narrator_tune import autoconfigure as ac
from narrator_tune.config import load_tune_config
from narrator_tune.library_eval import ConfigScore, ModelProfile
from narrator.asr.stream_sim import DecodeCostModel
from narrator.config.loader import load_config, save_config
from narrator.config.schema import AsrConfig, RootConfig


def score(errors_per_excerpt: list[int], latency: float | None = 2.0, busy: float = 0.5,
          missing: int = 0, doubled: int = 0) -> ConfigScore:
    s = ConfigScore(missing=missing)
    for i, errors in enumerate(errors_per_excerpt):
        s.per_excerpt[f"x{i}"] = (errors, 100)
        s.substitutions += errors
        s.reference_words += 100
        s.hypothesis_words += 100
    s.excerpts = len(errors_per_excerpt)
    s.latency_p50 = latency
    s.audio_seconds = 100.0
    s.busy_seconds = busy * 100.0
    s.doubled = doubled
    return s


# ---------------------------------------------------------------------------
# small pure pieces
# ---------------------------------------------------------------------------


def test_with_setting_moves_the_rollover_as_one_setting():
    config = AsrConfig(max_context_seconds=20, max_utterance_ms=20000)
    changed = ac.with_setting(config, "max_context_seconds", 10.0)
    assert changed.max_context_seconds == 10.0 and changed.max_utterance_ms == 10000
    assert ac.with_setting(config, "step_ms", 300).step_ms == 300


def test_feasibility_rules():
    assert ac.feasible(score([10] * 4), max_latency_s=3.0)
    assert not ac.feasible(score([10] * 4, latency=3.5), max_latency_s=3.0)
    assert ac.feasible(score([10] * 4, latency=3.5), max_latency_s=None)
    assert not ac.feasible(score([10] * 4, busy=0.95), max_latency_s=3.0)
    assert ac.feasible(score([10] * 20, missing=1), max_latency_s=3.0)  # one crashed excerpt is tolerated
    assert not ac.feasible(score([10] * 4, missing=3), max_latency_s=3.0)
    assert not ac.feasible(score([10] * 4, doubled=10), max_latency_s=3.0)  # 2.5% of words
    assert not ac.feasible(ConfigScore(), max_latency_s=3.0)


def test_significance_needs_both_words_and_noise_margin():
    assert ac.significant(3, 1.0)
    assert not ac.significant(2, 0.0)  # too few words
    assert not ac.significant(5, 6.0)  # within noise


def test_pick_setting_adopts_the_largest_significant_saving():
    incumbent = score([10] * 8)
    candidates = [("a", score([9] * 8)), ("b", score([7] * 8)), ("c", score([10] * 8))]
    assert ac.pick_setting(incumbent, candidates, 3.0)[0] == "b"


def test_pick_setting_keeps_incumbent_when_gain_is_noise():
    incumbent = score([10, 10, 10, 10])
    noisy = score([0, 20, 2, 17])  # 1 error saved overall, huge spread
    assert ac.pick_setting(incumbent, [("n", noisy)], 3.0) is None


def test_pick_setting_never_adopts_an_infeasible_value():
    incumbent = score([10] * 8)
    assert ac.pick_setting(incumbent, [("slow", score([2] * 8, latency=4.0))], 3.0) is None


def test_pick_setting_fixes_an_infeasible_incumbent_even_without_gain():
    incumbent = score([10] * 8, latency=5.0)
    picked = ac.pick_setting(incumbent, [("ok", score([11] * 8)), ("worse", score([12] * 8))], 3.0)
    assert picked[0] == "ok"


def test_rank_models_by_wer_ignoring_delay_but_not_load():
    ranked = ac.rank_models(
        [
            ("over_cap", score([5] * 4, latency=4.0)),  # delay is tunable: still ranked by WER
            ("b", score([8] * 4)),
            ("overloaded", score([2] * 4, busy=0.95)),  # the hardware cannot carry it
            ("c", score([9] * 4)),
            ("no_results", ConfigScore()),
        ]
    )
    assert ranked == ["over_cap", "b", "c", "overloaded"]


def test_choose_winner_prefers_faster_model_within_noise():
    big = ("tuned big", AsrConfig(model="big"), score([10] * 8), 1.0)
    fast = ("tuned fast", AsrConfig(model="fast"), score([10] * 7 + [11]), 0.3)  # 1 error worse
    label, config, _ = ac.choose_winner([big, fast], 3.0)
    assert config.model == "fast"
    clearly_better = ("tuned big", AsrConfig(model="big"), score([5] * 8), 1.0)
    label, config, _ = ac.choose_winner([clearly_better, fast], 3.0)
    assert config.model == "big"


def test_holdout_gate():
    assert ac.holdout_gate(score([10] * 4), score([10] * 4))
    assert ac.holdout_gate(score([10] * 4), None)
    assert not ac.holdout_gate(score([12] * 4), score([10] * 4))  # 2 pt worse


def test_settings_diff_lists_model_and_decode_settings_only():
    old = AsrConfig(model="large-v3", step_ms=800, show_partial_captions=False)
    new = old.model_copy(update={"model": "large-v3-turbo", "step_ms": 500, "show_partial_captions": True})
    assert ac.settings_diff(old, new) == {"model": ("large-v3", "large-v3-turbo"), "step_ms": (800, 500)}


def test_slowest_decode_ratio():
    assert ac.slowest_decode_ratio({"2.0s": 0.4, "10.0s": 1.0}) == pytest.approx(0.2)
    assert ac.slowest_decode_ratio({}) == float("inf")


def test_quick_tuning_shrinks_samples_and_passes():
    quick = ac.Tuning().quick()
    assert quick.screen_excerpts <= 6 and quick.dev_excerpts <= 10 and quick.search_passes == 1
    assert quick.time_budget_minutes <= 60


def test_tuning_reads_tune_config(tmp_path):
    path = tmp_path / "tune.yaml"
    path.write_text(
        yaml.safe_dump({"max_latency_s": 0, "search_space": {"step_ms": [400, 600]}}),
        encoding="utf-8",
    )
    tuning = ac.Tuning.from_config(load_tune_config(path))
    assert tuning.max_latency_s is None  # 0 = no cap
    assert tuning.search_space["step_ms"] == [400, 600]
    assert tuning.search_space["agreement_passes"] == [2, 3]  # untouched grids keep defaults


# ---------------------------------------------------------------------------
# apply_settings
# ---------------------------------------------------------------------------


def write_config(tmp_path: Path, **asr) -> Path:
    path = tmp_path / "config.yaml"
    base = RootConfig()
    config = base.model_copy(update={"asr": AsrConfig.model_validate({**base.asr.model_dump(), **asr})})
    save_config(config, path)
    return path


def test_apply_settings_writes_model_and_decode_settings_only(tmp_path):
    path = write_config(tmp_path, model="large-v3", language_whitelist=["pl", "en"], unit="cpu", step_ms=800)
    winner = AsrConfig(model="large-v3-turbo", language_whitelist=["pl"], step_ms=500, max_context_seconds=15, max_utterance_ms=15000)
    ac.apply_settings(path, winner)
    saved = load_config(path).asr
    assert saved.model == "large-v3-turbo" and saved.step_ms == 500
    assert saved.max_context_seconds == 15 and saved.max_utterance_ms == 15000
    assert saved.language_whitelist == ["pl", "en"]  # operator's choice, untouched
    assert saved.unit == "cpu"  # the unit is never changed
    backup = load_config(path.with_name("config.yaml.bak")).asr
    assert backup.model == "large-v3" and backup.step_ms == 800


# ---------------------------------------------------------------------------
# bench / profiles
# ---------------------------------------------------------------------------


def test_profile_model_gates_slow_models_and_persists_bench(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "COST_MODELS_PATH", tmp_path / "cost_models.json")
    calls = []

    def fake_bench(module, payload, timeout):
        calls.append(payload["model"])
        if payload["model"] == "slow":
            return {"resolved_device": "CPU", "decode_seconds_by_window": {"2.0s": 4.0, "10.0s": 20.0}}
        return {"resolved_device": "GPU", "decode_seconds_by_window": {"2.0s": 0.2, "10.0s": 0.6}, "detect_seconds": 0.1}

    monkeypatch.setattr(ac, "run_probe_subprocess", fake_bench)
    tuning = ac.Tuning()
    fast = ac.profile_model("fast", "auto", tuning, None, rebench=False)
    assert fast.error is None and fast.profile.cost.detect_s == pytest.approx(0.1)
    slow = ac.profile_model("slow", "auto", tuning, None, rebench=False)
    assert slow.profile is None and "too slow" in slow.error
    ac.profile_model("fast", "auto", tuning, None, rebench=False)
    assert calls == ["fast", "slow"]  # the stored bench was reused
    ac.profile_model("fast", "auto", tuning, None, rebench=True)
    assert calls == ["fast", "slow", "fast"]


def test_profile_model_reports_bench_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "COST_MODELS_PATH", tmp_path / "cost_models.json")
    monkeypatch.setattr(ac, "run_probe_subprocess", lambda *a: {"error": "boom"})
    candidate = ac.profile_model("m", "auto", ac.Tuning(), None, rebench=False)
    assert candidate.profile is None and "boom" in candidate.error


# ---------------------------------------------------------------------------
# the staged search, end to end, on a fake evaluator
# ---------------------------------------------------------------------------


class FakeExcerpt:
    def __init__(self, i: int, kind: str = "synthetic") -> None:
        self.id = f"x{i}"
        self.kind = kind


class FakeEvaluator:
    """Errors per excerpt from a formula whose optimum is known:
    model 'turbo' is best, step_ms 500, max_context 15, vad 0.5; long steps
    with 3 agreement passes blow the delay cap."""

    def __init__(self, cost_per_eval: float = 0.0, holdout_penalty: dict[str, int] | None = None,
                 busy_by_step: bool = False) -> None:
        self.cost_per_eval = cost_per_eval
        # worker load grows as the step shrinks (step 300 ms overloads it)
        self.busy_by_step = busy_by_step
        # extra errors per holdout excerpt, per model: a model that only
        # looked good on the excerpts it was tuned on (overfitting)
        self.holdout_penalty = holdout_penalty or {}
        self.evaluations: list[tuple[str, str, int]] = []
        self.decode_seconds_spent = 0.0

        class _Cache:
            directory = Path("/nonexistent")

            def prepare(self, excerpts, progress=None):
                return {}

        self.cache = _Cache()

    @staticmethod
    def errors(config: AsrConfig) -> int:
        e = {"turbo": 20, "big": 22, "small": 30}.get(config.model, 40)
        e += abs(config.step_ms - 500) // 100
        e += int(abs(config.max_context_seconds - 15.0) // 5) * 2
        e += 0 if config.vad_threshold == 0.5 else 2
        return e

    def evaluate(self, profile, configs, excerpts, mode="stream", language=None):
        out = []
        for config in configs:
            self.evaluations.append((config.model, mode, len(excerpts), config.step_ms))
            on_holdout = bool(excerpts) and excerpts[0].kind == "holdout"
            per = self.errors(config) + (self.holdout_penalty.get(config.model, 0) if on_holdout else 0)
            latency = config.step_ms / 1000 * config.agreement_passes + 0.5
            busy = 0.33 / (config.step_ms / 1000) if self.busy_by_step else 0.5
            out.append(score([per] * len(excerpts), latency=latency, busy=busy))
        return out

    def estimate_seconds(self, profile, configs, excerpts, mode="stream"):
        return self.cost_per_eval * len(configs)


def make_run(tmp_path, evaluator, budget_minutes=600.0, holdout=True, **asr) -> ac._Run:
    path = write_config(tmp_path, model="big", step_ms=800, max_context_seconds=20, max_utterance_ms=20000, **asr)
    tuning = ac.Tuning(
        candidate_models=["big", "turbo", "small"],
        search_space={"max_context_seconds": [10.0, 15.0, 20.0], "vad_threshold": [0.35, 0.5], "step_ms": [300, 500, 800, 1200], "agreement_passes": [2, 3]},
    )
    cost = DecodeCostModel(0.1, 0.05)
    profiles = {m: ModelProfile(m, "auto", "fake", DecodeCostModel(0.05 if m == "turbo" else 0.1, 0.05)) for m in ("big", "turbo", "small")}
    run = ac._Run(path, tuning, ac.Budget(budget_minutes), evaluator, profiles)
    screen = [FakeExcerpt(i) for i in range(8)]
    dev = screen + [FakeExcerpt(i + 8) for i in range(8)]
    hold = [FakeExcerpt(100 + i, kind="holdout") for i in range(6)] if holdout else []
    run.excerpts = lambda: (screen, dev, hold, [])
    return run


def test_search_finds_the_known_optimum_and_writes_it(tmp_path):
    run = make_run(tmp_path, FakeEvaluator())
    outcome = run.run(["big", "turbo", "small"])
    assert outcome.written, outcome.reason
    winner = outcome.winner
    assert winner.model == "turbo"
    assert winner.step_ms == 500 and winner.max_context_seconds == 15.0 and winner.vad_threshold == 0.5
    assert winner.agreement_passes == 2
    assert outcome.changes["model"] == ("big", "turbo")
    assert {"model_screen", "search", "confirm", "holdout"} <= set(run.report["stages"])


def test_search_respects_the_delay_cap(tmp_path):
    run = make_run(tmp_path, FakeEvaluator())
    run.tuning.max_latency_s = 1.2  # step 500 x 2 passes + 0.5 = 1.5 s is too slow
    outcome = run.run(["big", "turbo", "small"])
    assert outcome.winner.step_ms * outcome.winner.agreement_passes / 1000 + 0.5 <= 1.2


def test_holdout_regression_blocks_the_write(tmp_path):
    run = make_run(tmp_path, FakeEvaluator(holdout_penalty={"turbo": 30}))
    outcome = run.run(["big", "turbo", "small"])
    assert not outcome.written
    assert "holdout" in outcome.reason


def test_budget_stops_the_search_early_but_still_decides(tmp_path):
    evaluator = FakeEvaluator(cost_per_eval=60.0)
    run = make_run(tmp_path, evaluator, budget_minutes=6.0)
    outcome = run.run(["big", "turbo", "small"])
    searched = [e for e in evaluator.evaluations if e[2] == 8]
    assert len(searched) < 20  # far fewer than a full search
    assert outcome.winner is not None


def test_current_settings_already_best_are_not_rewritten(tmp_path):
    run = make_run(tmp_path, FakeEvaluator())
    # make the operator's config the optimum
    config = load_config(run.config_path)
    optimum = config.asr.model_copy(update={"model": "turbo", "step_ms": 500, "max_context_seconds": 15.0, "max_utterance_ms": 15000})
    save_config(config.model_copy(update={"asr": optimum}), run.config_path)
    run.config = load_config(run.config_path)
    run.baseline = run.config.asr
    outcome = run.run(["big", "turbo", "small"])
    assert not outcome.written


def test_calibration_fits_measured_decodes_and_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "COST_MODELS_PATH", tmp_path / "cost_models.json")
    calls = [[w, n, 0.1 + 0.05 * w + 0.01 * n] for w in (2.0, 6.0, 12.0, 20.0) for n in (5, 20, 40)]

    class MeasuringEvaluator:
        def __init__(self) -> None:
            self.runs = 0

        def evaluate(self, profile, configs, excerpts, mode="stream", language=None):
            assert mode == "measured"
            self.runs += 1

        def record(self, excerpt, profile, config, mode):
            return {"result": {"calls": calls, "detects": [0.3, 0.32, 0.28]}}

    evaluator = MeasuringEvaluator()
    excerpts = [FakeExcerpt(i, kind="asr") for i in range(3)]
    profile = ModelProfile("m", "auto", "fake", DecodeCostModel(0.1, 0.01))
    assert ac.calibrate_profile(profile, AsrConfig(), excerpts, evaluator)
    assert profile.cost.per_audio_s == pytest.approx(0.05, abs=1e-3)
    assert profile.cost.per_word_s == pytest.approx(0.01, abs=1e-3)
    assert profile.cost.detect_s == pytest.approx(0.3)

    again = ModelProfile("m", "auto", "fake", DecodeCostModel(0.1, 0.01))
    assert ac.calibrate_profile(again, AsrConfig(), excerpts, evaluator)
    assert evaluator.runs == 1  # the stored calibration was reused
    assert again.cost == profile.cost


def test_calibration_skipped_for_synthetic_clips(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "COST_MODELS_PATH", tmp_path / "cost_models.json")
    profile = ModelProfile("m", "auto", "fake", DecodeCostModel(0.1, 0.01))
    assert not ac.calibrate_profile(profile, AsrConfig(), [FakeExcerpt(0)], evaluator=None)
    assert profile.cost == DecodeCostModel(0.1, 0.01)


def test_whitelist_check_reports_without_changing_the_whitelist(tmp_path):
    evaluator = FakeEvaluator()
    run = make_run(tmp_path, evaluator, language_whitelist=["pl", "en"])
    outcome = run.run(["big", "turbo", "small"])
    assert outcome.written
    checks = run.report["stages"]["whitelist_check"]
    assert len(checks) == 1 and checks[0]["settings"]["language_whitelist"] == ["pl"]
    assert outcome.winner.language_whitelist == ["pl", "en"]  # never changed


def test_no_whitelist_check_for_a_single_language(tmp_path):
    run = make_run(tmp_path, FakeEvaluator(), language_whitelist=["pl"])
    run.run(["big", "turbo", "small"])
    assert "whitelist_check" not in run.report["stages"]


def test_pilot_prunes_settings_that_overload_the_worker(tmp_path):
    evaluator = FakeEvaluator(busy_by_step=True)
    run = make_run(tmp_path, evaluator)
    outcome = run.run(["big", "turbo", "small"])
    assert outcome.winner.step_ms == 500
    step_300_runs = [e for e in evaluator.evaluations if e[3] == 300]
    assert step_300_runs and all(e[2] == ac.PILOT_EXCERPTS for e in step_300_runs)
    pruned = [e["label"] for e in run.report["stages"]["search"] if "pruned" in e["label"]]
    assert "step_ms=300 (pruned by pilot)" in pruned


def test_pilot_rejection_rules():
    assert ac.pilot_rejects(score([10] * 4, busy=1.2), 3.0)
    assert ac.pilot_rejects(score([10] * 4, latency=5.0), 3.0)
    assert not ac.pilot_rejects(score([10] * 4, latency=4.0, busy=0.95), 3.0)  # within pilot noise
    assert not ac.pilot_rejects(score([10] * 4, latency=9.0), None)
    assert not ac.pilot_rejects(ConfigScore(), 3.0)


def test_search_tries_delay_levers_first_when_over_the_cap(tmp_path):
    evaluator = FakeEvaluator()
    run = make_run(tmp_path, evaluator)
    run.tuning.max_latency_s = 1.2  # the starting step 800 x 2 passes is over the cap
    run.run(["big", "turbo", "small"])
    searched = [e["label"].split("=")[0] for e in run.report["stages"]["search"] if "=" in e["label"]]
    assert searched[0] in ac.LATENCY_SETTINGS


def test_apply_settings_never_writes_operator_text_settings(tmp_path):
    path = write_config(
        tmp_path, text_filters=[{"pattern": "foo", "replacement": "bar"}], strip_bracketed_annotations=False,
    )
    winner = AsrConfig(min_mean_token_p=0.3, max_timestamp_ratio=2.25)
    ac.apply_settings(path, winner)
    saved = load_config(path).asr
    assert saved.min_mean_token_p == 0.3 and saved.max_timestamp_ratio == 2.25
    assert [f.pattern for f in saved.text_filters] == ["foo"]
    assert saved.strip_bracketed_annotations is False


def test_decode_settings_are_json_safe_and_key_on_text_filters():
    from narrator_tune.library_eval import config_signature, decode_settings

    filtered = AsrConfig(text_filters=[{"pattern": "foo"}])
    assert json.dumps(decode_settings(filtered))
    assert config_signature(filtered) != config_signature(AsrConfig())


def test_cost_model_key_carries_gpu_and_whispercpp_identity(monkeypatch):
    monkeypatch.setattr(ac.library_eval, "machine_identity", lambda device="vulkan": "GPU-A|1.5.1+whispercpp.1.9.5")
    key_a = ac.cost_model_key("small", "vulkan")
    monkeypatch.setattr(ac.library_eval, "machine_identity", lambda device="vulkan": "GPU-B|1.5.1+whispercpp.1.9.5")
    assert key_a != ac.cost_model_key("small", "vulkan")
    assert "GPU-A" in key_a


def test_segmenter_levers_apply_only_to_the_segmenter():
    la = AsrConfig(language_whitelist=["pl"])
    seg = AsrConfig(language_whitelist=["pl"], decoder="segmenter")
    seg_partials = seg.model_copy(update={"show_partial_captions": True})
    for name in ("max_segment_ms", "segmentation_mode", "context_sentences"):
        assert ac.setting_applies(seg, name) and not ac.setting_applies(la, name)
    for name in ("step_ms", "agreement_passes", "max_context_seconds", "vad_preroll_ms", "prompt_carryover_chars"):
        assert ac.setting_applies(la, name) and not ac.setting_applies(seg, name)
    assert not ac.setting_applies(seg, "partial_latency_ms")  # partials off: no peeks to tune
    assert ac.setting_applies(seg_partials, "partial_latency_ms")
    for name in ("end_silence_ms", "vad_threshold", "min_mean_token_p", "max_timestamp_ratio", "decoder"):
        assert ac.setting_applies(seg, name) and ac.setting_applies(la, name)
    assert set(ac.DECODER_SPECIFIC) <= set(ac.SEARCH_ORDER)
    assert "max_segment_ms" in ac.LATENCY_SETTINGS


def test_partials_key_segmenter_results_but_not_local_agreement_ones():
    from narrator_tune.library_eval import config_signature

    la = AsrConfig(language_whitelist=["pl"])
    seg = AsrConfig(language_whitelist=["pl"], decoder="segmenter")
    on = {"show_partial_captions": True}
    assert config_signature(la) == config_signature(la.model_copy(update=on))
    assert config_signature(seg) != config_signature(seg.model_copy(update=on))
    assert config_signature(la) != config_signature(seg)
