"""Pydantic config schema for narrator.

Note: there is deliberately no `font_size` field anywhere in this schema.
Font size is always derived at render time from canvas/margin/line-count/
interline geometry (see narrator.captions.layout) and must never be set
directly by the operator.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, model_validator

_HEX_RGBA_RE = re.compile(r"^#([0-9A-Fa-f]{8})$")


def parse_rgba(color: str) -> tuple[int, int, int, int]:
    """Parse a '#RRGGBBAA' string into an (r, g, b, a) tuple of ints 0-255."""
    match = _HEX_RGBA_RE.match(color)
    if not match:
        raise ValueError(f"color must be in '#RRGGBBAA' format, got {color!r}")
    hex_digits = match.group(1)
    r, g, b, a = (int(hex_digits[i : i + 2], 16) for i in (0, 2, 4, 6))
    return r, g, b, a


def _validate_rgba_str(value: str) -> str:
    parse_rgba(value)  # raises ValueError if invalid
    return value


RgbaColor = Annotated[str, AfterValidator(_validate_rgba_str)]


class AsrDevice(str, Enum):
    AUTO = "auto"
    VULKAN = "vulkan"
    CPU = "cpu"


class TextAlign(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class VerticalAlign(str, Enum):
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"


class BandShape(str, Enum):
    RECTANGLE = "rectangle"
    PILL = "pill"


class AppConfig(BaseModel):
    log_level: str = "INFO"


class AudioConfig(BaseModel):
    device: str = "default"
    sample_rate: int = Field(default=16000, gt=0)
    frame_ms: int = Field(default=30, gt=0)


class AsrConfig(BaseModel):
    model: str = "large-v3"
    language: str = "pl"
    # Dynamic language switching (only when `language` is a forced code, not
    # "auto"): the configured language is used for decoding until detection
    # reports a DIFFERENT language confidently this many consecutive times
    # (once per utterance finalize, plus periodic checks inside long
    # utterances), at which point the active language switches -- and switches
    # back the same way. 0 disables switching entirely (pure forced). The
    # special value language="auto" instead follows the speaker verbatim per
    # utterance with no default language.
    language_switch_threshold: int = Field(default=2, ge=0)
    # Optional restriction of the languages the decoder may use. Empty = no
    # restriction (whisper decides freely). When set, every language decision
    # (auto per-utterance lock and dynamic switching alike) re-argmaxes the
    # detection probabilities over this whitelist, so a known pl+en venue can
    # never produce German. The configured `language` (if not "auto") is
    # always an implicit member.
    language_whitelist: list[str] = Field(default_factory=list)
    device: AsrDevice = AsrDevice.AUTO
    # Which Vulkan GPU to use when more than one is present (a discrete card
    # alongside an integrated one is the common case). "auto" picks the first
    # discrete GPU, falling back to whatever is enumerated first -- without
    # this, whisper.cpp always takes GPU 0, which on a machine whose iGPU
    # enumerates first would silently run inference on the slow adapter.
    gpu_device: int | Literal["auto"] = "auto"
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    step_ms: int = Field(default=500, gt=0)
    agreement_passes: int = Field(default=2, ge=1)
    end_silence_ms: int = Field(default=650, gt=0)
    max_context_seconds: float = Field(default=20.0, gt=0)
    clear_after_silence_seconds: float = Field(default=8.0, gt=0)
    show_partial_captions: bool = False

    @model_validator(mode="after")
    def _check_language_whitelist(self) -> "AsrConfig":
        # Unknown codes would silently never match any detection -- an
        # operator typo like "pol" would shrink the effective whitelist
        # without any signal. Reject at config time instead. "auto" in the
        # list means 'follow the speaker verbatim', which the dynamic-switch
        # machinery cannot express; the per-utterance "auto" language mode is
        # the supported way to get that behavior.
        from narrator.asr.whisper_languages import whisper_language_codes

        valid = set(whisper_language_codes())
        for code in self.language_whitelist:
            if code == "auto":
                raise ValueError("language_whitelist must contain ISO codes, not 'auto'")
            if code not in valid:
                raise ValueError(
                    f"language_whitelist entry {code!r} is not a language whisper supports"
                )
        # An EMPTY whitelist means unrestricted -- never auto-populate it
        # with the default language, or `language: pl` + `whitelist: []`
        # would silently become pl-only and dynamic switching could never
        # fire. Prepending only applies to an explicit list that forgot its
        # own default language.
        if (
            self.language_whitelist
            and self.language != "auto"
            and self.language not in self.language_whitelist
        ):
            self.language_whitelist = [self.language, *self.language_whitelist]
        return self


class CanvasConfig(BaseModel):
    width_px: int = Field(default=1920, gt=0)
    height_px: int = Field(default=1080, gt=0)
    background_color: RgbaColor = "#00000000"


class LayoutConfig(BaseModel):
    max_lines: int = Field(default=2, ge=1)
    margin_top_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    margin_bottom_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    margin_left_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    margin_right_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    interline_pct: float = Field(default=1.0, ge=0.0)
    text_align: TextAlign = TextAlign.LEFT
    vertical_align_within_box: VerticalAlign = VerticalAlign.BOTTOM

    @model_validator(mode="after")
    def _check_margin_sums(self) -> "LayoutConfig":
        if self.margin_top_pct + self.margin_bottom_pct >= 100.0:
            raise ValueError(
                "layout.margin_top_pct + layout.margin_bottom_pct must be < 100 "
                f"(got {self.margin_top_pct} + {self.margin_bottom_pct})"
            )
        if self.margin_left_pct + self.margin_right_pct >= 100.0:
            raise ValueError(
                "layout.margin_left_pct + layout.margin_right_pct must be < 100 "
                f"(got {self.margin_left_pct} + {self.margin_right_pct})"
            )
        return self


class BackgroundBandStyle(BaseModel):
    enabled: bool = True
    shape: BandShape = BandShape.PILL
    color: RgbaColor = "#000000A0"
    corner_radius_pct: float = Field(default=2.0, ge=0.0)
    padding_pct: float = Field(default=1.0, ge=0.0)


class ShadowStyle(BaseModel):
    enabled: bool = False
    offset_x_pct: float = 0.3
    offset_y_pct: float = 0.3
    blur_px: int = Field(default=4, ge=0)
    color: RgbaColor = "#000000C0"


class StrokeStyle(BaseModel):
    enabled: bool = True
    width_px: int = Field(default=2, ge=0)
    color: RgbaColor = "#000000FF"


class StyleOverrides(BaseModel):
    background_band: BackgroundBandStyle = Field(default_factory=BackgroundBandStyle)
    shadow: ShadowStyle = Field(default_factory=ShadowStyle)
    stroke: StrokeStyle = Field(default_factory=StrokeStyle)


class StyleConfig(BaseModel):
    font_family: str = "assets/fonts/NotoSans-Regular.ttf"
    font_color: RgbaColor = "#FFFFFFFF"
    overrides: StyleOverrides = Field(default_factory=StyleOverrides)


class NdiConfig(BaseModel):
    source_name: str = "LiveCaption"
    fps: float = Field(default=30.0, gt=0)


class GuiConfig(BaseModel):
    enabled: bool = True


class RootConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    canvas: CanvasConfig = Field(default_factory=CanvasConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)
    ndi: NdiConfig = Field(default_factory=NdiConfig)
    gui: GuiConfig = Field(default_factory=GuiConfig)

    @model_validator(mode="after")
    def _check_interline_fits(self) -> "RootConfig":
        height = self.canvas.height_px
        layout = self.layout
        box_height = height * (
            1.0 - (layout.margin_top_pct + layout.margin_bottom_pct) / 100.0
        )
        gap_count = max(layout.max_lines - 1, 0)
        total_gap_px = height * layout.interline_pct / 100.0 * gap_count
        if total_gap_px >= box_height:
            raise ValueError(
                "layout.interline_pct combined with layout.max_lines consumes the "
                f"entire text box height (total gap {total_gap_px:.1f}px >= box height "
                f"{box_height:.1f}px); reduce interline_pct, max_lines, or margins"
            )
        return self
