"""narrator-tune's own settings (config/tune.yaml, narrator_tune.config)."""
import pytest
import yaml
from pydantic import ValidationError

from narrator_tune.autoconfigure import Tuning
from narrator_tune.config import DEFAULTS_PATH, TuneConfig, load_tune_config


def test_tune_config_defaults_and_bounds():
    tc = TuneConfig()
    assert tc.candidate_models[:2] == ["large-v3", "large-v3-turbo"]
    assert tc.time_budget_minutes == pytest.approx(720.0)
    assert tc.max_latency_s == pytest.approx(3.0)
    assert (tc.screen_excerpts, tc.dev_excerpts, tc.holdout_excerpts) == (32, 96, 48)
    assert tc.reference_languages == ["pl"]

    # the screen set is the first part of the dev set
    with pytest.raises(ValidationError):
        TuneConfig(screen_excerpts=50, dev_excerpts=40)
    # only searchable settings, each with at least one value
    with pytest.raises(ValidationError):
        TuneConfig(search_space={"font_size": [10]})
    with pytest.raises(ValidationError):
        TuneConfig(search_space={"step_ms": []})
    with pytest.raises(ValidationError):
        TuneConfig(reference_languages=["xx"])
    assert TuneConfig(search_space={"step_ms": [400, 600]}).search_space == {"step_ms": [400, 600]}


def test_shipped_template_matches_model_defaults():
    assert TuneConfig.model_validate(yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))) == TuneConfig()


def test_missing_tune_yaml_is_created_from_template(tmp_path):
    path = tmp_path / "config" / "tune.yaml"
    assert load_tune_config(path) == TuneConfig()
    assert path.read_text(encoding="utf-8") == DEFAULTS_PATH.read_text(encoding="utf-8")


def test_leftover_autoconfigure_section_is_migrated(tmp_path):
    """An existing narrator config.yaml's `autoconfigure:` section seeds
    tune.yaml on first use; keys of the previous autoconfigure design
    (wer_floor, speed weights, ...) load without error."""
    narrator_config = tmp_path / "config.yaml"
    narrator_config.write_text(
        yaml.safe_dump(
            {
                "autoconfigure": {
                    "max_latency_s": 2.0,
                    "time_budget_minutes": 90,
                    "wer_floor": 0.6,
                    "speed_weight": 0.5,
                    "real_excerpts": 4,
                },
                "asr": {"max_utterance_ms": 25000, "max_context_seconds": 20},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    path = tmp_path / "tune.yaml"
    loaded = load_tune_config(path, narrator_config_path=narrator_config)
    assert loaded.max_latency_s == pytest.approx(2.0)
    assert path.exists()
    # once created, tune.yaml is the source of truth
    assert load_tune_config(path) == loaded

    tuning = Tuning.from_config(loaded)
    assert tuning.max_latency_s == pytest.approx(2.0)
    assert tuning.time_budget_minutes == pytest.approx(90.0)
    assert tuning.slow_factor == pytest.approx(1.5)  # untouched fields keep defaults


def test_invalid_leftover_section_falls_back_to_template(tmp_path):
    narrator_config = tmp_path / "config.yaml"
    narrator_config.write_text(yaml.safe_dump({"autoconfigure": {"screen_excerpts": 50, "dev_excerpts": 4}}))
    assert load_tune_config(tmp_path / "tune.yaml", narrator_config_path=narrator_config) == TuneConfig()
