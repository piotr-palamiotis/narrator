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
    # The language whitelist is the single language control. Empty = auto:
    # each utterance is decoded in whatever language detection reports (any
    # of whisper's languages). One entry = that language, permanently. Several
    # entries = each utterance is decoded in the highest-confidence allowed
    # language, with the decision gated on the detector's probability MARGIN
    # between the top two allowed candidates (see streaming_decoder.py):
    # ambiguous windows are not locked until the margin clears, so a genuine
    # language switch is never guessed on a coin flip. Unknown codes are
    # rejected by the validator below.
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
    # Continuous speech is force-finalized ("rolled over") at the SMALLER of
    # max_utterance_ms and max_context_seconds, and the next speech starts a
    # fresh utterance (fresh whisper context, cleared hypotheses). The
    # decode window is never trimmed mid-utterance: trimming shifted the
    # hypotheses under the committed-word count and silently dropped ~5s of
    # speech per long utterance (documentary narration runs past 20s
    # without a pause for 40-65% of its speech). Long single contexts also
    # accumulate errors on noisy audio (measured on a real 88-min assembly
    # recording, see scripts/stream_eval.py).
    max_utterance_ms: int = Field(default=20000, gt=0)
    max_context_seconds: float = Field(default=20.0, gt=0)
    # After this much continuous silence, the ASR worker emits a CLEAR event,
    # which prunes only OFF-SCREEN caption history. The visible lower-third
    # stays until the operator clears it (Ctrl+N / the GUI Clear button), a
    # wipe_screen_after_silence_seconds lull elapses, or enough new speech
    # scrolls it out under layout.max_lines.
    prune_history_after_silence_seconds: float = Field(default=8.0, gt=0)
    # After this much silence (and only this much -- a long lull where stale
    # captions would misattribute to unrelated new speech), the worker emits a
    # WIPE event: a true screen clear. 0 disables it entirely, which is the
    # default -- the operator decides when the screen goes blank.
    wipe_screen_after_silence_seconds: float = Field(default=0.0, ge=0)
    show_partial_captions: bool = False
    # Audio kept from just BEFORE the VAD fires, prepended to each utterance:
    # Silero needs a window or two to cross its threshold, so without it the
    # first syllable of an utterance can be clipped. 0 disables it.
    vad_preroll_ms: int = Field(default=0, ge=0, le=1000)
    # Characters of already-captioned text handed to whisper as the prompt
    # for the next utterance's decodes (context for names, spelling and
    # style); reset by a screen wipe and on a language change. 0 = each
    # utterance starts from a blank context.
    prompt_carryover_chars: int = Field(default=0, ge=0, le=1000)

    @model_validator(mode="before")
    @classmethod
    def _migrate_clear_after_silence(cls, data):
        """Accept the retired `clear_after_silence_seconds` key.

        That knob's meaning changed when CLEAR stopped wiping the screen
        (it now only prunes off-screen history), so it was renamed to make
        the behavior readable from the config file. Old keys are still
        honored -- mapped onto the prune knob -- so existing config.yaml
        files keep their value across the upgrade instead of silently
        reverting to the default. Explicit new keys always win over the old
        one (setdefault).
        """
        if isinstance(data, dict) and "clear_after_silence_seconds" in data:
            old_value = data.pop("clear_after_silence_seconds")
            data.setdefault("prune_history_after_silence_seconds", old_value)
        return data

    @model_validator(mode="after")
    def _check_language_whitelist(self) -> "AsrConfig":
        # Unknown codes would silently never match any detection -- an
        # operator typo like "pol" would shrink the effective whitelist
        # without any signal. Reject at config time instead.
        from narrator.asr.whisper_languages import whisper_language_codes

        valid = set(whisper_language_codes())
        for code in self.language_whitelist:
            if code not in valid:
                raise ValueError(
                    f"language_whitelist entry {code!r} is not a language whisper supports"
                )
        # De-duplicate, preserving order: a repeated entry would double-count
        # in the margin computation (harmless for argmax, confusing for
        # display) and adds nothing.
        deduped: list[str] = []
        for code in self.language_whitelist:
            if code not in deduped:
                deduped.append(code)
        self.language_whitelist = deduped
        return self


# Decode settings `narrator --autoconfigure` may search (asr.<name>), in
# search order; autoconfigure.search_space can override their value grids.
AUTOCONFIGURE_SEARCH_SETTINGS = (
    "max_context_seconds",
    "vad_threshold",
    "end_silence_ms",
    "step_ms",
    "agreement_passes",
    "vad_preroll_ms",
    "prompt_carryover_chars",
)


class AutoconfigureConfig(BaseModel):
    """Tuning for `narrator --autoconfigure`, which searches the model and
    decode settings (asr.*) under which live captions best match the
    reference transcripts in scripts/test_audio/, and writes the winner into
    this file. Read only by that CLI run -- the live pipeline never touches
    this section, so editing it in the GUI while captioning is harmless.
    """

    # Models considered. Ones missing from the local pywhispercpp cache are
    # skipped (no surprise multi-GB downloads unless --allow-download).
    # `narrator --models ...` overrides this list for one run.
    candidate_models: list[str] = Field(
        default_factory=lambda: ["large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"]
    )
    # A model whose benched decode is slower than this multiple of real time
    # at any window cannot caption live on this machine and is skipped.
    slow_factor: float = Field(default=1.5, gt=0.0)
    # Wall-clock budget for new decoding. Results are cached across runs, so
    # a rerun only spends time on what it has not decoded before.
    time_budget_minutes: float = Field(default=720.0, gt=0.0)
    # Median delay from a word being spoken to its caption appearing that a
    # configuration may have (seconds); 0 = no cap, pure accuracy.
    max_latency_s: float = Field(default=3.0, ge=0.0)
    # Evaluation sample: fixed-length excerpts, one per recording before any
    # recording gets a second. screen (the search) is the first part of dev
    # (confirmation); holdout recordings are never used for tuning.
    excerpt_seconds: float = Field(default=90.0, ge=20.0, le=600.0)
    screen_excerpts: int = Field(default=32, ge=1)
    dev_excerpts: int = Field(default=96, ge=1)
    holdout_excerpts: int = Field(default=48, ge=0)
    # Reference languages that are scored; recordings whose subtitle file is
    # tagged with another language (e.g. "(English).srt") are skipped.
    reference_languages: list[str] = Field(default_factory=lambda: ["pl"])
    # Excerpts that even offline large-v3 disagrees with by more than this
    # (WER) are dropped: their reference does not match the audio.
    max_ceiling_wer: float = Field(default=0.85, gt=0.0)
    # Disk guards for the decoded-excerpt cache (scripts/autoconfigure_cache).
    min_free_disk_mb: float = Field(default=1024.0, ge=0.0)
    cache_max_mb: float = Field(default=1024.0, gt=0.0)
    # Optional per-setting value grids replacing the built-in ones, e.g.
    # {"step_ms": [400, 600, 900]}.
    search_space: dict[str, list[int | float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_autoconfigure_bounds(self) -> "AutoconfigureConfig":
        if self.screen_excerpts > self.dev_excerpts:
            raise ValueError(
                "autoconfigure.screen_excerpts must be <= autoconfigure.dev_excerpts "
                "(the screen set is the first part of the dev set)"
            )
        unknown = sorted(set(self.search_space) - set(AUTOCONFIGURE_SEARCH_SETTINGS))
        if unknown:
            raise ValueError(
                f"autoconfigure.search_space has unknown settings {unknown}; "
                f"known: {list(AUTOCONFIGURE_SEARCH_SETTINGS)}"
            )
        empty = sorted(name for name, values in self.search_space.items() if not values)
        if empty:
            raise ValueError(f"autoconfigure.search_space entries need at least one value: {empty}")
        from narrator.asr.whisper_languages import whisper_language_codes

        valid = set(whisper_language_codes())
        for code in self.reference_languages:
            if code not in valid:
                raise ValueError(
                    f"autoconfigure.reference_languages entry {code!r} is not a language whisper supports"
                )
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
    autoconfigure: AutoconfigureConfig = Field(default_factory=AutoconfigureConfig)

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
