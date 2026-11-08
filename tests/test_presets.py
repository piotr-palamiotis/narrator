"""asr.preset: resolution/precedence, save round trip, and the
`narrator --pick-preset` decision (GPU-free: fake bench/probe runner)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from narrator.asr import preset_picker as pp
from narrator.asr.stream_sim import DecodeCostModel
from narrator.config.loader import DEFAULTS_PATH, load_config, save_config, write_asr_preset
from narrator.config.presets import (
    CUSTOM,
    PRESET_FIELDS,
    load_presets,
    parse_presets,
    render_presets_yaml,
)
from narrator.config.schema import AsrConfig, RootConfig

PRESETS = load_presets()
FIRST = next(iter(PRESETS.values()))
LAST = list(PRESETS.values())[-1]


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --- presets.yaml ----------------------------------------------------------


def test_shipped_presets_are_well_formed_and_unmeasured():
    assert {"quality", "balanced", "light", "cpu"} <= set(PRESETS)
    assert CUSTOM not in PRESETS
    for preset in PRESETS.values():
        assert set(preset.values()) == set(PRESET_FIELDS)
        assert preset.measured is False  # nothing has been searched yet


def test_render_round_trips():
    assert parse_presets(yaml.safe_load(render_presets_yaml(PRESETS))) == PRESETS


def test_preset_rows_may_not_carry_device_or_whitelist():
    with pytest.raises(ValueError):
        parse_presets({"tiers": {"x": {"model": "small", "step_ms": 500, "agreement_passes": 2, "device": "cpu"}}})
    with pytest.raises(ValueError):
        parse_presets({"tiers": {"custom": {"model": "small", "step_ms": 500, "agreement_passes": 2}}})


# --- resolution ------------------------------------------------------------


def test_preset_supplies_its_fields():
    asr = AsrConfig.model_validate({"preset": LAST.name})
    assert (asr.model, asr.step_ms, asr.agreement_passes) == (LAST.model, LAST.step_ms, LAST.agreement_passes)


def test_explicit_value_wins_over_preset():
    asr = AsrConfig.model_validate({"preset": LAST.name, "step_ms": 1234})
    assert asr.step_ms == 1234
    assert asr.model == LAST.model and asr.agreement_passes == LAST.agreement_passes


def test_preset_never_touches_unit_or_whitelist():
    asr = AsrConfig.model_validate({"preset": LAST.name, "unit": "vulkan:1", "language_whitelist": ["pl", "en"]})
    assert asr.unit == "vulkan:1"
    assert asr.language_whitelist == ["pl", "en"]
    assert AsrConfig.model_validate({"preset": LAST.name}).unit == "auto"


def test_custom_and_missing_preset_change_nothing():
    plain = AsrConfig.model_validate({"model": "tiny", "step_ms": 700})
    assert plain.preset == CUSTOM
    assert (plain.model, plain.step_ms, plain.agreement_passes) == ("tiny", 700, 2)
    assert AsrConfig.model_validate({"preset": CUSTOM}) == AsrConfig()


def test_unknown_preset_is_rejected():
    with pytest.raises(ValidationError, match="unknown asr.preset"):
        AsrConfig.model_validate({"preset": "qualty"})


def test_old_config_without_preset_keeps_its_values(tmp_path):
    path = _write(tmp_path / "c.yaml", {"asr": {"model": "medium", "step_ms": 800, "agreement_passes": 3}})
    asr = load_config(path).asr
    assert (asr.preset, asr.model, asr.step_ms, asr.agreement_passes) == (CUSTOM, "medium", 800, 3)


def test_defaults_keep_todays_values():
    asr = load_config(DEFAULTS_PATH).asr
    assert asr.preset != CUSTOM
    assert (asr.model, asr.step_ms, asr.agreement_passes) == ("large-v3", 500, 2)
    raw = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))["asr"]
    assert not set(PRESET_FIELDS) & set(raw)  # follows the preset, nothing frozen


# --- saving keeps preset semantics -------------------------------------------


def test_save_round_trip_does_not_freeze_preset_values(tmp_path):
    path = _write(tmp_path / "c.yaml", {"asr": {"preset": LAST.name, "device": "vulkan"}})
    config = load_config(path)
    # The GUI/web Save path: rebuild from a dump of the live config + the form,
    # which always carries model/step_ms/agreement_passes.
    form = AsrConfig.model_validate({**config.asr.model_dump(), "model": config.asr.model, "step_ms": config.asr.step_ms})
    save_config(RootConfig.model_validate({**config.model_dump(), "asr": form.model_dump()}), path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))["asr"]
    assert saved["preset"] == LAST.name
    assert not set(PRESET_FIELDS) & set(saved)
    assert load_config(path) == config


def test_save_keeps_an_explicit_override(tmp_path):
    path = _write(tmp_path / "c.yaml", {"asr": {"preset": LAST.name}})
    config = load_config(path)
    edited = config.model_copy(update={"asr": AsrConfig.model_validate({**config.asr.model_dump(), "step_ms": 4321})})
    save_config(edited, path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))["asr"]
    assert saved["step_ms"] == 4321 and "model" not in saved
    assert load_config(path).asr.step_ms == 4321


def test_save_custom_writes_everything(tmp_path):
    path = tmp_path / "c.yaml"
    save_config(RootConfig(), path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))["asr"]
    assert saved["preset"] == CUSTOM and set(PRESET_FIELDS) <= set(saved)


# --- write_asr_preset -------------------------------------------------------


def test_write_asr_preset_changes_only_the_preset(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(DEFAULTS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    before = yaml.safe_load(path.read_text(encoding="utf-8"))
    backup = write_asr_preset(path, LAST.name)
    after = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert after["asr"]["preset"] == LAST.name
    after["asr"]["preset"] = before["asr"]["preset"]
    assert after == before
    assert "# Performance preset" in path.read_text(encoding="utf-8")  # comments kept
    assert backup.read_text(encoding="utf-8") == DEFAULTS_PATH.read_text(encoding="utf-8")


def test_write_asr_preset_inserts_and_validates(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("# mine\nasr:\n    model: small  # pinned\napp:\n  log_level: INFO\n", encoding="utf-8")
    write_asr_preset(path, LAST.name)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw == {"asr": {"preset": LAST.name, "model": "small"}, "app": {"log_level": "INFO"}}
    assert "# pinned" in path.read_text(encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValidationError):
        write_asr_preset(path, "nope")
    assert path.read_text(encoding="utf-8") == before


# --- the picker -------------------------------------------------------------


def test_caption_delays_align_words():
    segments = [{"start_s": 0.0, "end_s": 2.0, "reference": "jeden dwa"}, {"start_s": 3.0, "end_s": 4.0, "reference": "trzy"}]
    finals = [(2.5, "Jeden, dwa."), (5.0, "trzy")]
    assert pp.caption_delays(finals, segments) == pytest.approx([2.0, 1.0, 1.5])
    assert pp.delay_p50([], segments) is None


def test_choose_preset_takes_the_first_tier_that_fits():
    rows = [
        pp.TierResult("a", "big", 500, 2, load=0.9, delay_p50=1.0),
        pp.TierResult("b", "mid", 500, 2, load=0.5, delay_p50=3.5),
        pp.TierResult("c", "small", 500, 2, error="not downloaded"),
        pp.TierResult("d", "tiny", 500, 2, load=0.3, delay_p50=1.2),
        pp.TierResult("e", "tinier", 500, 2, load=0.1, delay_p50=0.5),
    ]
    assert pp.choose_preset(rows).name == "d"
    assert pp.choose_preset(rows[:3]) is None


# Fake hardware: per-model cost models; "worker load" and delay follow from
# them the way the real probe's replay would (decode per step / step length).
FAKE_COSTS = {
    "large-v3": DecodeCostModel(base_s=0.2, per_audio_s=0.1, per_word_s=0.03),
    "medium": DecodeCostModel(base_s=0.03, per_audio_s=0.01, per_word_s=0.005),
    "small": DecodeCostModel(base_s=0.01, per_audio_s=0.005, per_word_s=0.002),
    "base": DecodeCostModel(base_s=0.005, per_audio_s=0.002, per_word_s=0.001),
}


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, module: str, payload: dict, timeout_s: float) -> dict:
        self.calls.append((module, payload))
        cost = FAKE_COSTS[payload["model"]]
        if module == "narrator.asr.bench":
            return {
                "decode_seconds_by_window": {f"{w:.1f}s": cost.decode_seconds(w, int(2.5 * w)) for w in payload["windows_s"]},
                "detect_seconds": 0.1,
                "resolved_device": payload["unit"],
            }
        tiers = []
        for tier in payload["tiers"]:
            step = tier["step_ms"] / 1000.0
            decode = cost.decode_seconds(5.0, 12)
            tiers.append(
                {
                    "name": tier["name"],
                    "busy_fraction": min(decode / step, 1.5),
                    "delay_p50": step * tier["agreement_passes"] + decode,
                    "wer": 0.05,
                }
            )
        return {"tiers": tiers}


def test_picker_chooses_most_accurate_fitting_preset(tmp_path):
    path = _write(tmp_path / "config.yaml", {"asr": {"device": "vulkan", "language_whitelist": ["pl"]}})
    runner = FakeRunner()
    outcome = pp.pick_preset(path, runner=runner, model_available=lambda m: True, log=lambda s: None)
    # quality: large-v3 decodes 1.06s per 0.5s step -> load > 2, over the limit;
    # balanced: medium 0.14s per 0.35s step -> load 0.4
    assert outcome.rows[0].name == "quality" and not outcome.rows[0].fits()
    assert outcome.chosen.name == "balanced"
    assert [r.name for r in outcome.rows] == ["quality", "balanced"]  # stops at the first fit
    assert outcome.written
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw == {"asr": {"device": "vulkan", "language_whitelist": ["pl"], "preset": "balanced"}}
    assert (tmp_path / "config.yaml.bak").exists()


def test_picker_never_changes_the_unit(tmp_path):
    # an old config: asr.device + asr.gpu_device migrate to asr.unit vulkan:1
    path = _write(tmp_path / "config.yaml", {"asr": {"device": "vulkan", "gpu_device": 1, "unit_identity": "GPU @ 0000:03:00.0"}})
    runner = FakeRunner()
    pp.pick_preset(path, runner=runner, model_available=lambda m: True, measure_all=True, dry_run=True, log=lambda s: None)
    assert {payload["model"] for _, payload in runner.calls} == set(FAKE_COSTS)  # the "cpu" tier too
    for module, payload in runner.calls:
        settings = payload if module == "narrator.asr.bench" else payload["asr"]
        assert settings["unit"] == "vulkan:1"
        assert settings["unit_identity"] == "GPU @ 0000:03:00.0"


def test_picker_dry_run_and_missing_models(tmp_path):
    path = _write(tmp_path / "config.yaml", {"asr": {"model": "large-v3"}})
    text = path.read_text(encoding="utf-8")
    runner = FakeRunner()
    outcome = pp.pick_preset(path, runner=runner, dry_run=True, model_available=lambda m: m == "small", log=lambda s: None)
    assert outcome.chosen.name == "light" and not outcome.written
    assert path.read_text(encoding="utf-8") == text and not (tmp_path / "config.yaml.bak").exists()
    assert {payload["model"] for _, payload in runner.calls} == {"small"}
    assert "not downloaded" in outcome.rows[0].error
    assert outcome.overrides == {"model": "large-v3"}  # reported: it overrides the preset


def test_picker_writes_nothing_when_no_preset_fits(tmp_path):
    path = _write(tmp_path / "config.yaml", {"asr": {"device": "cpu"}})
    outcome = pp.pick_preset(path, runner=FakeRunner(), model_available=lambda m: False, log=lambda s: None)
    assert outcome.chosen is None and not outcome.written
    assert not (tmp_path / "config.yaml.bak").exists()


def test_bundled_clip_matches_its_metadata():
    audio, segments = pp.load_clip()
    assert len(segments) == 6
    assert segments[-1]["end_s"] <= len(audio) / 16000
