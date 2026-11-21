import pytest
from pydantic import ValidationError

from narrator.config.loader import load_config, save_config
from narrator.config.schema import AsrConfig, CanvasConfig, LayoutConfig, RootConfig


def test_defaults_are_valid():
    config = RootConfig()
    assert config.asr.model == "large-v3"
    assert config.asr.language_whitelist == []  # unrestricted auto by default
    assert config.style.overrides.background_band.enabled is True


def test_top_bottom_margin_sum_must_be_under_100():
    with pytest.raises(ValidationError):
        LayoutConfig(margin_top_pct=60, margin_bottom_pct=40)


def test_top_bottom_margin_sum_exactly_100_rejected():
    with pytest.raises(ValidationError):
        LayoutConfig(margin_top_pct=50, margin_bottom_pct=50)


def test_left_right_margin_sum_must_be_under_100():
    with pytest.raises(ValidationError):
        LayoutConfig(margin_left_pct=70, margin_right_pct=30)


def test_valid_margins_pass():
    layout = LayoutConfig(margin_top_pct=10, margin_bottom_pct=10, margin_left_pct=10, margin_right_pct=10)
    assert layout.margin_top_pct == 10


def test_max_lines_must_be_at_least_one():
    with pytest.raises(ValidationError):
        LayoutConfig(max_lines=0)


def test_interline_consuming_whole_box_rejected_at_root_level():
    with pytest.raises(ValidationError):
        RootConfig(
            canvas=CanvasConfig(width_px=1000, height_px=1000),
            layout=LayoutConfig(max_lines=10, margin_top_pct=0, margin_bottom_pct=0, interline_pct=20.0),
        )


def test_invalid_color_format_rejected():
    with pytest.raises(ValidationError):
        CanvasConfig(background_color="not-a-color")
    with pytest.raises(ValidationError):
        CanvasConfig(background_color="#FFFFFF")  # missing alpha channel


def test_valid_rgba_color_accepted():
    canvas = CanvasConfig(background_color="#00000000")
    assert canvas.background_color == "#00000000"


def test_no_font_size_field_exists_anywhere():
    dumped = RootConfig().model_dump()
    def find_key(d, key):
        if isinstance(d, dict):
            if key in d:
                return True
            return any(find_key(v, key) for v in d.values())
        return False
    assert not find_key(dumped, "font_size")
    assert not find_key(dumped, "font_size_px")


def test_asr_unit_defaults_to_auto():
    from narrator.config.schema import AsrConfig

    asr = AsrConfig()
    assert asr.unit == "auto" and asr.unit_identity == "" and asr.unit_kind == "auto"


@pytest.mark.parametrize(
    "old,unit",
    [
        ({"device": "cpu"}, "cpu"),
        ({"device": "cpu", "gpu_device": 1}, "cpu"),
        ({"device": "vulkan"}, "auto"),
        ({"device": "auto", "gpu_device": "auto"}, "auto"),
        ({"device": "vulkan", "gpu_device": 1}, "vulkan:1"),
        ({"gpu_device": 0}, "vulkan:0"),
        ({"device": "auto", "gpu_device": "2"}, "vulkan:2"),
    ],
)
def test_retired_device_keys_migrate_to_unit(old, unit, caplog):
    """asr.device + asr.gpu_device (before C6) become asr.unit, with a log
    line, so an existing config.yaml keeps its choice."""
    from narrator.config.schema import AsrConfig

    with caplog.at_level("INFO", logger="narrator.config.schema"):
        asr = AsrConfig.model_validate(old)
    assert asr.unit == unit
    assert "asr.unit" in caplog.text
    assert "device" not in asr.model_dump() and "gpu_device" not in asr.model_dump()


def test_explicit_unit_wins_over_retired_keys():
    from narrator.config.schema import AsrConfig

    assert AsrConfig.model_validate({"unit": "cpu", "device": "vulkan", "gpu_device": 1}).unit == "cpu"


def test_migrated_config_file_saves_only_the_unit(tmp_path):
    import yaml as yaml_mod

    path = tmp_path / "config.yaml"
    path.write_text(yaml_mod.safe_dump({"asr": {"device": "vulkan", "gpu_device": 1}}), encoding="utf-8")
    config = load_config(path)
    assert config.asr.unit == "vulkan:1"
    save_config(config, path)
    asr = yaml_mod.safe_load(path.read_text(encoding="utf-8"))["asr"]
    assert asr["unit"] == "vulkan:1" and "device" not in asr and "gpu_device" not in asr


def test_old_clear_after_silence_key_migrates_to_prune():
    """Configs written before the rename keep working: the retired
    clear_after_silence_seconds key maps onto the prune knob."""
    from narrator.config.schema import AsrConfig

    config = AsrConfig.model_validate({"clear_after_silence_seconds": 12.5})
    assert config.prune_history_after_silence_seconds == 12.5
    assert not hasattr(config, "clear_after_silence_seconds")  # popped, not kept


def test_new_prune_key_wins_over_old_alias():
    """When both keys are present, the explicit new one wins."""
    from narrator.config.schema import AsrConfig

    config = AsrConfig.model_validate(
        {"clear_after_silence_seconds": 12.5, "prune_history_after_silence_seconds": 3.0}
    )
    assert config.prune_history_after_silence_seconds == 3.0


def test_wipe_screen_after_silence_defaults_to_disabled():
    """0 = the screen never clears on its own; the operator decides."""
    from narrator.config.schema import AsrConfig

    config = AsrConfig()
    assert config.wipe_screen_after_silence_seconds == 0.0

    with pytest.raises(ValidationError):
        AsrConfig(wipe_screen_after_silence_seconds=-1.0)


def test_max_utterance_may_be_below_max_context():
    """The decoder rolls over at the smaller of the two, so any combination
    is valid (the old cross-field rule no longer applies)."""
    from narrator.config.schema import AsrConfig

    config = RootConfig(asr=AsrConfig(max_context_seconds=30, max_utterance_ms=25000))
    assert config.asr.max_utterance_ms == 25000


def test_preroll_and_prompt_carryover_bounds():
    from narrator.config.schema import AsrConfig

    config = AsrConfig()
    assert config.vad_preroll_ms == 0 and config.prompt_carryover_chars == 0
    assert AsrConfig(vad_preroll_ms=250, prompt_carryover_chars=200).vad_preroll_ms == 250
    with pytest.raises(ValidationError):
        AsrConfig(vad_preroll_ms=-1)
    with pytest.raises(ValidationError):
        AsrConfig(prompt_carryover_chars=5000)


def test_asr_unit_accepts_each_form_and_normalizes():
    from narrator.config.schema import AsrConfig

    assert AsrConfig(unit="Vulkan:1 ").unit == "vulkan:1"
    assert AsrConfig(unit="cuda:0").unit_kind == "cuda"
    assert AsrConfig(unit="vulkan:3").unit_index == 3
    assert AsrConfig(unit="cpu").unit_index is None
    assert AsrConfig(unit="vulkan:0", unit_identity=" NVIDIA @ 0000:01:00.0 ").unit_identity == "NVIDIA @ 0000:01:00.0"


def test_leftover_autoconfigure_section_is_dropped(tmp_path, caplog):
    """The `autoconfigure:` section moved to narrator-tune's own
    config/tune.yaml; an old config.yaml that still has it loads, and the
    section is gone after a save."""
    import yaml as yaml_mod

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml_mod.safe_dump(
            {"autoconfigure": {"max_latency_s": 2.0, "wer_floor": 0.6}, "asr": {"step_ms": 700}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with caplog.at_level("INFO", logger="narrator.config.schema"):
        loaded = load_config(config_path)
    assert loaded.asr.step_ms == 700
    assert not hasattr(loaded, "autoconfigure")
    assert "autoconfigure" in caplog.text
    save_config(loaded, config_path)
    assert "autoconfigure" not in yaml_mod.safe_load(config_path.read_text(encoding="utf-8"))


def test_asr_unit_rejects_nonsense():
    import pytest
    from pydantic import ValidationError

    from narrator.config.schema import AsrConfig

    for bad in ("second one", "gpu", "vulkan", "vulkan:-1", "rocm:0", "cuda:x"):
        with pytest.raises(ValidationError):
            AsrConfig(unit=bad)
    with pytest.raises(ValidationError):
        AsrConfig(gpu_device="second one")  # the retired key is still validated
    with pytest.raises(ValidationError):
        AsrConfig(device="opencl")


def test_model_unit_combination_is_checked_against_the_catalog(monkeypatch):
    """The model catalog decides which unit kinds a model runs on: every
    whisper.cpp model runs on cpu/cuda/vulkan; an ONNX model not on Vulkan,
    a NeMo model only on CUDA. `auto` fits any model."""
    import pytest
    from pydantic import ValidationError

    from narrator.asr import catalog
    from narrator.config.schema import AsrConfig

    for unit in ("cpu", "vulkan:0", "cuda:0", "auto"):
        assert AsrConfig(model="large-v3", unit=unit).unit == unit
    monkeypatch.setattr(catalog, "EXTRA_ENTRIES", [
        catalog.CatalogEntry("parakeet-test", catalog.AsrBackendKind.ONNX, ("parakeet.onnx",)),
        catalog.CatalogEntry("canary-test", catalog.AsrBackendKind.NEMO, ("canary.nemo",)),
    ])
    assert AsrConfig(model="parakeet-test", unit="cuda:0").unit == "cuda:0"
    assert AsrConfig(model="parakeet-test", unit="auto").unit == "auto"
    with pytest.raises(ValidationError, match="not on vulkan"):
        AsrConfig(model="parakeet-test", unit="vulkan:0")
    with pytest.raises(ValidationError, match="not on cpu"):
        AsrConfig(model="canary-test", unit="cpu")


# --- language whitelist -----------------------------------------------------


def test_language_whitelist_defaults_empty():
    from narrator.config.schema import AsrConfig

    assert AsrConfig().language_whitelist == []


def test_language_whitelist_accepts_known_codes():
    from narrator.config.schema import AsrConfig

    config = AsrConfig(language_whitelist=["pl", "en"])
    assert config.language_whitelist == ["pl", "en"]


def test_language_whitelist_rejects_unknown_code():
    from narrator.config.schema import AsrConfig

    with pytest.raises(ValidationError, match="not a language whisper supports"):
        AsrConfig(language_whitelist=["pl", "xx"])


def test_language_whitelist_deduplicates_preserving_order():
    from narrator.config.schema import AsrConfig

    config = AsrConfig(language_whitelist=["en", "pl", "en"])
    assert config.language_whitelist == ["en", "pl"]


# ---------------------------------------------------------------------------
# obs-localvocal-derived settings
# ---------------------------------------------------------------------------


def test_text_filter_pattern_must_be_a_valid_regex():
    from narrator.config.schema import AsrConfig

    with pytest.raises(ValidationError):
        AsrConfig(text_filters=[{"pattern": "(unclosed"}])
    with pytest.raises(ValidationError):
        AsrConfig(text_filters=[{"pattern": ""}])
    assert AsrConfig(text_filters=[{"pattern": r"\bfoo\b"}]).text_filters[0].replacement == ""


def test_quality_gates_default_off_and_validate_ranges():
    from narrator.config.schema import AsrConfig

    config = AsrConfig()
    assert config.min_mean_token_p == 0.0 and config.max_timestamp_ratio == 0.0
    assert config.strip_bracketed_annotations and not config.flash_attn
    with pytest.raises(ValidationError):
        AsrConfig(min_mean_token_p=1.5)
    with pytest.raises(ValidationError):
        AsrConfig(max_timestamp_ratio=-1)


def test_removed_transcript_and_recording_sections_are_dropped():
    from narrator.config.schema import RootConfig

    config = RootConfig.model_validate({
        "transcript": {"enabled": True, "format": "docx"},
        "recording": {"record_on_start": True},
        "web": {"shortcuts": {"recording_toggle": "F9", "clear": "F7"}},
    })
    assert not hasattr(config, "transcript") and not hasattr(config, "recording")
    assert config.web.shortcuts.clear == "F7"
    assert "recording_toggle" not in config.web.shortcuts.model_dump()


def test_layout_pacing_accepts_yaml_bool_false_as_off():
    import yaml
    from narrator.config.schema import CaptionPacing, LayoutConfig

    raw = yaml.safe_load("pacing: off")  # unquoted -> False
    assert raw["pacing"] is False
    assert LayoutConfig.model_validate(raw).pacing == CaptionPacing.OFF
    assert LayoutConfig.model_validate({"pacing": "word"}).pacing == CaptionPacing.WORD


# ---------------------------------------------------------------------------
# asr.decoder and the segmenter's settings (todo B Phase 5)
# ---------------------------------------------------------------------------


def test_decoder_defaults_to_local_agreement_and_defaults_yaml_agrees():
    import yaml as _yaml
    from importlib import resources

    assert AsrConfig().decoder.value == "local_agreement"
    shipped = _yaml.safe_load(resources.files("narrator.config").joinpath("defaults.yaml").read_text())
    asr = AsrConfig.model_validate(shipped["asr"])
    assert asr.decoder.value == "local_agreement"
    assert asr.segmentation_mode.value == "vad"
    assert (asr.partial_latency_ms, asr.segment_duration_ms, asr.max_segment_ms, asr.context_sentences) == (
        1100, 7000, 15000, 1)


@pytest.mark.parametrize(
    "raw, expected",
    [("localagreement", "local_agreement"), ("Local-Agreement", "local_agreement"),
     ("segment", "segmenter"), ("segmenter", "segmenter"), ("localvocal", "segmenter")],
)
def test_decoder_aliases_are_migrated(raw, expected):
    assert AsrConfig(decoder=raw).decoder.value == expected


def test_unknown_decoder_is_rejected():
    with pytest.raises(ValidationError):
        AsrConfig(decoder="simulstreaming_typo")


def test_localvocal_names_are_migrated():
    asr = AsrConfig.model_validate(
        {"decoder": "segmenter", "segmentation_mode": "disabled", "duration_filter_threshold": 2.25,
         "n_context_sentences": 3, "segment_duration_ms": 5000})
    assert asr.segmentation_mode.value == "fixed"
    assert asr.max_timestamp_ratio == 2.25
    assert asr.context_sentences == 3


def test_new_names_win_over_migrated_ones():
    asr = AsrConfig.model_validate({"duration_filter_threshold": 2.25, "max_timestamp_ratio": 3.0})
    assert asr.max_timestamp_ratio == 3.0


def test_segmenter_cap_taken_from_the_local_agreement_caps_when_unset():
    asr = AsrConfig.model_validate({"decoder": "segmenter", "max_utterance_ms": 12000, "max_context_seconds": 9})
    assert asr.max_segment_ms == 9000
    explicit = AsrConfig.model_validate({"decoder": "segmenter", "max_utterance_ms": 12000, "max_segment_ms": 8000})
    assert explicit.max_segment_ms == 8000
    la = AsrConfig.model_validate({"max_utterance_ms": 12000})
    assert la.max_segment_ms == 15000  # local agreement: untouched


def test_segmenter_lengths_are_cross_checked():
    with pytest.raises(ValidationError, match="partial_latency_ms"):
        AsrConfig(decoder="segmenter", show_partial_captions=True, partial_latency_ms=5000, max_segment_ms=4000)
    with pytest.raises(ValidationError, match="end_silence_ms"):
        AsrConfig(decoder="segmenter", end_silence_ms=4000, max_segment_ms=4000)
    with pytest.raises(ValidationError, match="segment_duration_ms"):
        AsrConfig(decoder="segmenter", segmentation_mode="hybrid", show_partial_captions=True,
                  partial_latency_ms=3000, segment_duration_ms=2000)
    # only what is read is checked: partials off, another mode, another decoder
    AsrConfig(decoder="segmenter", partial_latency_ms=5000, max_segment_ms=4000)
    AsrConfig(decoder="segmenter", segmentation_mode="hybrid", end_silence_ms=4000, max_segment_ms=4000)
    AsrConfig(end_silence_ms=4000, max_segment_ms=4000, show_partial_captions=True, partial_latency_ms=5000)


def test_presets_leave_the_decoder_alone():
    asr = AsrConfig(preset="balanced", decoder="segmenter")
    assert asr.decoder.value == "segmenter"
    assert AsrConfig(preset="balanced").decoder.value == "local_agreement"


def test_pacing_accepts_yaml_off():
    from narrator.config.schema import LayoutConfig

    assert LayoutConfig(pacing=False).pacing.value == "off"
    assert LayoutConfig(pacing="word").pacing.value == "word"
