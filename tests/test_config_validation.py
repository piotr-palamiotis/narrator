import pytest
from pydantic import ValidationError

from narrator.config.schema import CanvasConfig, LayoutConfig, RootConfig


def test_defaults_are_valid():
    config = RootConfig()
    assert config.asr.model == "large-v3"
    assert config.asr.language == "pl"
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


def test_language_whitelist_rejects_auto():
    from narrator.config.schema import AsrConfig

    with pytest.raises(ValidationError, match="ISO codes"):
        AsrConfig(language_whitelist=["pl", "auto"])


def test_forced_language_is_implicit_whitelist_member():
    """language=pl with a whitelist that forgets pl must not strand the
    default language outside the allowed set -- the validator prepends it."""
    from narrator.config.schema import AsrConfig

    config = AsrConfig(language="pl", language_whitelist=["en"])
    assert config.language_whitelist == ["pl", "en"]


def test_auto_language_is_not_added_to_whitelist():
    from narrator.config.schema import AsrConfig

    config = AsrConfig(language="auto", language_whitelist=["en"])
    assert config.language_whitelist == ["en"]
