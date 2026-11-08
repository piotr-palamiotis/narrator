"""`narrator-tune presets`: per-tier search wiring and presets.yaml rows
(fake search; GPU-free)."""
from __future__ import annotations

import yaml

from narrator.config.presets import PRESETS_PATH, load_presets
from narrator.config.schema import AsrConfig
from narrator_tune.autoconfigure import Outcome
from narrator_tune.config import TuneConfig
from narrator_tune.presets import tier_tuning_overrides, tune_presets


def _outcome(written: bool, winner: AsrConfig | None, baseline: AsrConfig, table: list[dict]) -> Outcome:
    return Outcome(written=written, reason="fake", baseline=baseline, winner=winner, table=table)


def test_tier_overrides_search_only_covered_settings():
    cpu = load_presets()["cpu"]
    overrides = tier_tuning_overrides(cpu, {"step_ms": [300, 500], "agreement_passes": [2, 3], "vad_threshold": [0.5]})
    assert set(overrides["search_space"]) == {"step_ms", "agreement_passes"}
    assert overrides["search_space"]["step_ms"] == cpu.tune["search_space"]["step_ms"]
    assert overrides["max_busy_fraction"] == cpu.tune["max_load"]
    assert overrides["candidate_models"] == [cpu.model]


def test_tune_presets_writes_measured_rows(tmp_path):
    output = tmp_path / "presets.yaml"
    output.write_text(PRESETS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    narrator_config = tmp_path / "config.yaml"
    narrator_config.write_text(yaml.safe_dump({"asr": {"unit": "vulkan:0", "vad_threshold": 0.4}}), encoding="utf-8")
    calls = []

    def fake_search(config_path, **kwargs):
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))["asr"]
        calls.append((raw, kwargs))
        baseline = AsrConfig.model_validate(raw)
        dev_row = {"label": "", "set": "dev", "excerpts": 4, "wer": 0.2, "latency_p50": 2.0, "busy_fraction": 0.5}
        if raw["model"] == "medium":  # tuned to a new step
            winner = baseline.model_copy(update={"step_ms": 700})
            return _outcome(True, winner, baseline, [{**dev_row, "label": "tuned medium"}])
        if raw["model"] == "small":  # current settings best and feasible
            return _outcome(False, baseline, baseline, [{**dev_row, "label": "current settings"}])
        return _outcome(False, None, baseline, [])  # nothing feasible

    outcomes = tune_presets(narrator_config, output=output, tune_config=TuneConfig(), run_search=fake_search)
    rows = load_presets(output)
    assert rows["balanced"].measured and rows["balanced"].step_ms == 700
    assert rows["light"].measured and rows["light"].step_ms == load_presets()["light"].step_ms
    assert not rows["quality"].measured and not rows["cpu"].measured
    assert [o.name for o in outcomes] == list(rows)
    # each tier starts from its own row, preset custom, the tier's unit
    for raw, kwargs in calls:
        assert raw["preset"] == "custom" and raw["vad_threshold"] == 0.4
        assert kwargs["models"] == [raw["model"]]
    units = {raw["model"]: raw["unit"] for raw, _ in calls}
    assert units["base"] == "cpu" and units["large-v3"] == "vulkan:0"
    assert (tmp_path / "presets.yaml.bak").exists()


def test_tune_presets_dry_run_writes_nothing(tmp_path):
    output = tmp_path / "presets.yaml"
    text = PRESETS_PATH.read_text(encoding="utf-8")
    output.write_text(text, encoding="utf-8")
    narrator_config = tmp_path / "config.yaml"
    narrator_config.write_text("asr: {}\n", encoding="utf-8")

    def fake_search(config_path, **kwargs):
        baseline = AsrConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8"))["asr"])
        return _outcome(True, baseline.model_copy(update={"step_ms": 999}), baseline, [])

    outcomes = tune_presets(narrator_config, output=output, tiers=["light"], dry_run=True,
                            tune_config=TuneConfig(), run_search=fake_search)
    assert [o.name for o in outcomes] == ["light"] and outcomes[0].after.step_ms == 999
    assert output.read_text(encoding="utf-8") == text
