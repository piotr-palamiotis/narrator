import pytest
from pydantic import ValidationError

from narrator.config.schema import CanvasConfig, LayoutConfig, RootConfig


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


def test_asr_gpu_device_defaults_to_auto():
    from narrator.config.schema import AsrConfig

    assert AsrConfig().gpu_device == "auto"


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


def test_asr_gpu_device_accepts_explicit_index():
    from narrator.config.schema import AsrConfig

    assert AsrConfig(gpu_device=1).gpu_device == 1


def test_asr_gpu_device_rejects_nonsense():
    import pytest
    from pydantic import ValidationError

    from narrator.config.schema import AsrConfig

    with pytest.raises(ValidationError):
        AsrConfig(gpu_device="second one")


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
