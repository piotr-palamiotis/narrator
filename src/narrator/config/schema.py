"""Pydantic config schema for narrator.

Note: there is deliberately no `font_size` field anywhere in this schema.
Font size is always derived at render time from canvas/margin/line-count/
interline geometry (see narrator.captions.layout) and must never be set
directly by the operator.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

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


class AsrDecoder(str, Enum):
    """The streaming strategy (narrator.asr.decoder_common.create_decoder)."""

    LOCAL_AGREEMENT = "local_agreement"  # growing window + local agreement (streaming_decoder.py)
    SEGMENTER = "segmenter"  # VAD segments decoded once + partial peeks (segmenter.py)


class SegmentationMode(str, Enum):
    """How the segmenter cuts audio into segments."""

    VAD = "vad"  # a segment = one stretch of speech (Silero VAD)
    HYBRID = "hybrid"  # fixed segment_duration_ms segments, partials only while VAD hears speech
    FIXED = "fixed"  # fixed segment_duration_ms segments, no VAD at all


class CaptionPacing(str, Enum):
    """Typewriter-style caption output (narrator.captions.pacer)."""

    OFF = "off"
    CHAR = "char"
    WORD = "word"
    SENTENCE = "sentence"
    SMOOTH = "smooth"


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
    # A PortAudio input device name ("default" or a substring), or
    # "ndi:<source name>" to receive audio from an NDI source.
    device: str = "default"
    sample_rate: int = Field(default=16000, gt=0)
    frame_ms: int = Field(default=30, gt=0)
    # 1-based input channels averaged into the mono signal ASR sees. [1] =
    # first channel only; [1, 2] = stereo downmix. Whether the device (or NDI
    # source) really has these channels is checked when capture starts.
    channels: list[int] = Field(default_factory=lambda: [1])

    @field_validator("channels")
    @classmethod
    def _check_channels(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("audio.channels must list at least one input channel")
        if any(ch < 1 for ch in value):
            raise ValueError("audio.channels are 1-based: every channel must be >= 1")
        if len(set(value)) != len(value):
            raise ValueError("audio.channels must not repeat a channel")
        return value


class TextFilter(BaseModel):
    """One find-and-replace rule applied to every decode hypothesis before it
    can reach the captions: a regular expression, matched case-insensitively.
    An empty replacement deletes the match."""

    pattern: str
    replacement: str = ""

    @field_validator("pattern")
    @classmethod
    def _check_pattern(cls, value: str) -> str:
        if not value:
            raise ValueError("text filter pattern must not be empty")
        try:
            re.compile(value, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"text filter pattern {value!r} is not a valid regular expression: {exc}") from exc
        return value


class AsrConfig(BaseModel):
    # Performance preset (narrator.config.presets, presets.yaml): a named
    # tier supplying model, step_ms and agreement_passes wherever config.yaml
    # does not set them itself; "custom" = no preset. Resolved before
    # validation (_apply_preset), so the app only sees plain values.
    preset: str = "custom"
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
    # The processing unit (narrator.asr.units): "cpu", "vulkan:<n>" (ggml's
    # Vulkan index), "cuda:<n>" (nvidia-smi index; needs a CUDA build:
    # make asr-backend ASR_GPU=cuda), or "auto" =
    # the recommended unit when the worker starts (CUDA > discrete Vulkan
    # GPU > CPU > integrated GPU). Replaces asr.device + asr.gpu_device (old
    # configs are migrated below). The app never changes it by itself: a
    # missing or failing unit is a problem in the dashboard, and only the
    # operator picks another one.
    unit: str = "auto"
    # Name + PCI bus id of the GPU the operator picked ("NVIDIA GeForce RTX
    # 3070 @ 0000:01:00.0"), set by the web UI's unit picker. Indices shift
    # when cards are added; a unit whose device no longer matches this is a
    # problem and the worker refuses it, never a silent re-map. "" = not
    # recorded (no check).
    unit_identity: str = ""
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
    # stays until the operator clears it (the web UI Clear button / shortcut), a
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
    # Decode quality gates (from obs-localvocal), applied to every decode's
    # hypothesis; a rejected hypothesis counts as silence for that step.
    # Off (0) by default -- narrator-tune measures them against the reference
    # library before they should be trusted on Polish large-v3.
    # Reject when the mean probability of the text tokens falls below this.
    min_mean_token_p: float = Field(default=0.0, ge=0.0, le=1.0)
    # Reject when whisper's timestamps run past this multiple of the audio's
    # real length -- the signature of a hallucination/repetition loop
    # (obs-localvocal uses 2.25).
    max_timestamp_ratio: float = Field(default=0.0, ge=0.0)
    # Remove whisper's [bracketed] sound annotations ("[oklaski]",
    # "[muzyka]", "[wypowiedź w języku ukraińskim]") -- on this pipeline they
    # are mostly hallucinated over noise, music and applause.
    strip_bracketed_annotations: bool = True
    # Find-and-replace rules applied to each hypothesis (see TextFilter).
    text_filters: list[TextFilter] = Field(default_factory=list)
    # Event vocabulary (names, places, jargon): put at the front of whisper's
    # prompt on every decode, ahead of the prompt carry-over (see
    # narrator.asr.vocabulary). Merged with the named list below, if any.
    vocabulary: list[str] = Field(default_factory=list)
    # Name of a saved list, <config dir>/vocabulary/<name>.txt ("" = none).
    vocabulary_list: str = ""
    # Replace near-miss spellings of vocabulary words with the list's own
    # spelling (before agreement, like text_filters). Off by default: it can
    # over-snap ordinary words that resemble a name.
    vocabulary_snap: bool = False
    # whisper.cpp flash attention. Off by default: its Vulkan kernel crashes
    # the worker (SIGILL) on some GPU/driver combinations. It works on the
    # RX 6700 XT/RADV (no meaningful speedup there) and on the RTX 3070
    # (~25% faster decodes, but measurably different text). whisper.cpp
    # turns dtw_word_timestamps off when this is on.
    flash_attn: bool = False
    # Word times of caption events (CaptionEvent.start/end) from whisper's
    # DTW token alignment (cross-attention of the model's alignment heads)
    # instead of its cheaper token-timestamp heuristic. Measured on
    # large-v3: word-boundary error 0.07 s vs 0.25 s mean, at ~20% more
    # decode time. Captions themselves are unchanged.
    dtw_word_timestamps: bool = False
    # Streaming strategy: local_agreement (re-decode a growing window every
    # step_ms, commit what agreement_passes hypotheses agree on) or segmenter
    # (decode each VAD speech segment once when it ends; optional whole-
    # segment peeks while it is open). The fields below are the segmenter's;
    # step_ms/agreement_passes/max_utterance_ms/max_context_seconds/
    # vad_preroll_ms/prompt_carryover_chars are local_agreement's.
    decoder: AsrDecoder = AsrDecoder.LOCAL_AGREEMENT
    segmentation_mode: SegmentationMode = SegmentationMode.VAD
    # While a segment is open and show_partial_captions is on, re-decode it
    # (a peek) for display after this much new audio; the interval grows
    # with the segment's length and peeks give way when the worker is behind.
    partial_latency_ms: int = Field(default=1100, gt=0)
    # Segment length in hybrid/fixed mode.
    segment_duration_ms: int = Field(default=7000, gt=0)
    # Longest segment in vad mode: continuous speech is cut here (at the
    # first >=160 ms pause once 60% full), and the next segment continues
    # at once -- nothing is lost.
    max_segment_ms: int = Field(default=15000, gt=0)
    # FINAL texts carried as whisper's prompt into the next segments (reset
    # by a screen wipe and on a language change). 0 = no context.
    context_sentences: int = Field(default=1, ge=0, le=10)

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data):
        """Fill the preset's covered fields from presets.yaml, on the RAW
        dict -- before defaults apply, which is the only point where "set in
        config.yaml" and "defaulted" can be told apart. Explicit values win.
        An unknown preset name is an error (a typo must not silently mean
        "the defaults")."""
        if isinstance(data, dict):
            from narrator.config.presets import apply_preset

            data = apply_preset(data)
        return data

    @model_validator(mode="before")
    @classmethod
    def _migrate_device_keys(cls, data):
        """Accept the retired `device` (auto/vulkan/cpu) + `gpu_device`
        (auto/<index>) pair, mapped onto `unit` with a log line: cpu -> cpu,
        an explicit index -> vulkan:<n>, otherwise auto (the first discrete
        GPU, as `gpu_device: auto` meant). An explicit `unit` wins."""
        if not isinstance(data, dict) or ("device" not in data and "gpu_device" not in data):
            return data
        from narrator.asr.units import unit_from_legacy

        data = dict(data)
        device, gpu_device = data.pop("device", None), data.pop("gpu_device", None)
        if "unit" in data:
            logger.info("config: asr.device/asr.gpu_device ignored, asr.unit is set")
        else:
            data["unit"] = unit_from_legacy(device, gpu_device)
            logger.info(
                "config: asr.device %r + asr.gpu_device %r are now asr.unit %r",
                getattr(device, "value", device), gpu_device, data["unit"],
            )
        return data

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

    @model_validator(mode="before")
    @classmethod
    def _migrate_decoder_keys(cls, data):
        """Accept the segmenter's settings under the names they have
        elsewhere (obs-localvocal, todo B, early drafts of this schema),
        mapped onto the schema's names with a log line; an explicit new key
        always wins (setdefault).

          decoder: localagreement | local-agreement | agreement -> local_agreement
          decoder: segment | segmented | localvocal              -> segmenter
          segmentation_mode: disabled (LocalVocal's "VAD off")   -> fixed
          duration_filter_threshold                              -> max_timestamp_ratio
          n_context_sentences                                    -> context_sentences
          max_utterance_ms / max_context_seconds, segmenter only, with no
            max_segment_ms: the smaller one becomes max_segment_ms (the
            local-agreement rollover cap is the same idea: the longest
            stretch decoded as one)
        """
        if not isinstance(data, dict):
            return data
        aliases = {
            "decoder": {
                "localagreement": "local_agreement", "local-agreement": "local_agreement",
                "agreement": "local_agreement", "segment": "segmenter", "segmented": "segmenter",
                "localvocal": "segmenter",
            },
            "segmentation_mode": {"disabled": "fixed", "none": "fixed", "off": "fixed", False: "fixed"},
        }
        for key, mapping in aliases.items():
            value = data.get(key)
            folded = value.strip().lower() if isinstance(value, str) else value
            if folded in mapping:
                if folded != mapping[folded]:
                    logger.info("config: asr.%s %r read as %r", key, value, mapping[folded])
                data[key] = mapping[folded]
        for old, new in (("duration_filter_threshold", "max_timestamp_ratio"), ("n_context_sentences", "context_sentences")):
            if old in data:
                value = data.pop(old)
                if new not in data:
                    data[new] = value
                    logger.info("config: asr.%s is now asr.%s (value %r kept)", old, new, value)
                else:
                    logger.info("config: asr.%s ignored, asr.%s is set", old, new)
        if data.get("decoder") == "segmenter" and "max_segment_ms" not in data:
            caps = []
            if data.get("max_utterance_ms") is not None:
                caps.append(int(data["max_utterance_ms"]))
            if data.get("max_context_seconds") is not None:
                caps.append(int(float(data["max_context_seconds"]) * 1000))
            if caps:
                data["max_segment_ms"] = min(caps)
                logger.info(
                    "config: asr.max_segment_ms taken from max_utterance_ms/max_context_seconds (%d ms)",
                    data["max_segment_ms"],
                )
        return data

    @model_validator(mode="after")
    def _check_segmenter(self) -> "AsrConfig":
        """The segmenter's lengths must fit together -- checked only while
        the segmenter is selected (local agreement never reads them), and
        only for what reads them: the cap in vad mode, the segment duration
        in hybrid/fixed, the partial cadence with partials on."""
        if self.decoder is not AsrDecoder.SEGMENTER:
            return self
        partials = self.show_partial_captions
        if self.segmentation_mode is SegmentationMode.VAD:
            if partials and self.partial_latency_ms >= self.max_segment_ms:
                raise ValueError(
                    f"asr.partial_latency_ms ({self.partial_latency_ms}) must be smaller than "
                    f"asr.max_segment_ms ({self.max_segment_ms})"
                )
            if self.end_silence_ms >= self.max_segment_ms:
                raise ValueError(
                    f"asr.end_silence_ms ({self.end_silence_ms}) must be smaller than "
                    f"asr.max_segment_ms ({self.max_segment_ms})"
                )
        elif partials and self.partial_latency_ms >= self.segment_duration_ms:
            raise ValueError(
                f"asr.partial_latency_ms ({self.partial_latency_ms}) must be smaller than "
                f"asr.segment_duration_ms ({self.segment_duration_ms})"
            )
        return self

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

    @field_validator("unit", mode="before")
    @classmethod
    def _check_unit(cls, value):
        from narrator.asr.units import normalize_unit

        return normalize_unit(value)

    @field_validator("unit_identity")
    @classmethod
    def _strip_identity(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _check_model_unit(self) -> "AsrConfig":
        """The model must be able to run on the unit's kind (the model
        catalog, narrator.asr.catalog: whisper.cpp runs on cpu/cuda/vulkan,
        ONNX models not on Vulkan, NeMo only on CUDA)."""
        from narrator.asr.catalog import unit_kind_problem

        problem = unit_kind_problem(self.model, self.unit_kind)
        if problem is not None:
            raise ValueError(f"asr.unit {self.unit}: {problem}")
        return self

    @property
    def unit_kind(self) -> str:
        """"auto", "cpu", "vulkan" or "cuda"."""
        return self.unit.partition(":")[0]

    @property
    def unit_index(self) -> int | None:
        index = self.unit.partition(":")[2]
        return int(index) if index else None

    @field_validator("vocabulary")
    @classmethod
    def _clean_vocabulary(cls, value: list[str]) -> list[str]:
        from narrator.asr.vocabulary import merge_vocabulary

        return merge_vocabulary(value)

    @field_validator("vocabulary_list")
    @classmethod
    def _check_vocabulary_list(cls, value: str) -> str:
        from narrator.asr.vocabulary import is_valid_list_name

        value = value.strip()
        if value and not is_valid_list_name(value):
            raise ValueError(
                f"vocabulary_list {value!r} must be a plain list name (letters, digits, space, '.', '-', '_')"
            )
        return value


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
    # Typewriter-style caption output (narrator.captions.pacer): text is
    # revealed per character, word or sentence, faster when a backlog
    # builds; "off" shows each caption the moment it arrives.
    pacing: CaptionPacing = CaptionPacing.OFF

    @field_validator("pacing", mode="before")
    @classmethod
    def _yaml_bool_pacing(cls, value):
        # YAML 1.1 reads an unquoted `off` as False (and `on` as True).
        if value is False:
            return CaptionPacing.OFF
        return value

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


class ShortcutsConfig(BaseModel):
    """Web UI keyboard shortcuts (narrator.config.shortcuts has the syntax).
    "" leaves an action unbound. Browser-reserved combos (Ctrl+N/T/W, ...)
    are refused: a page cannot intercept them."""

    clear: str = "Escape"
    mode_toggle: str = "F2"
    focus_entry: str = "F4"
    # Fires only while the manual caption entry has focus.
    submit: str = "Enter"
    pause_toggle: str = "F8"
    help: str = "?"

    @field_validator("*")
    @classmethod
    def _check_combo(cls, value: str) -> str:
        from narrator.config.shortcuts import validate_shortcut

        return validate_shortcut(value)

    @model_validator(mode="after")
    def _check_unique(self) -> "ShortcutsConfig":
        seen: dict[str, str] = {}
        for action, combo in self.model_dump().items():
            if combo and combo in seen:
                raise ValueError(f"shortcut {combo} is bound to both {seen[combo]} and {action}")
            seen[combo] = action
        return self


class ApiToken(BaseModel):
    """A per-integration token for the control API (Companion, scripts):
    it may read /api/state and call the control actions, nothing else.
    Revoke an integration by deleting its entry."""

    name: str = Field(min_length=1)
    token: str = Field(min_length=24, pattern=r"^[A-Za-z0-9_\-]+$")


class OscConfig(BaseModel):
    """Optional OSC (UDP) control input, a thin adapter onto the control
    API (narrator.web.osc). OSC has no authentication: keep it on loopback."""

    enabled: bool = False
    bind: str = "127.0.0.1"
    port: int = Field(default=9000, ge=1, le=65535)


class WebConfig(BaseModel):
    """The browser admin UI (narrator.web), replacing the old Tk GUI."""

    enabled: bool = True
    # Listens on every address (other computers can reach it, behind the
    # login password); `narrator --local-only` limits it to this computer.
    # Deliberately a startup flag, not a config/UI setting.
    port: int = Field(default=8765, ge=1, le=65535)
    # Live preview frame rate cap (frames are only sent when they change).
    preview_fps: float = Field(default=5.0, gt=0, le=30)
    shortcuts: ShortcutsConfig = Field(default_factory=ShortcutsConfig)
    api_tokens: list[ApiToken] = Field(default_factory=list)
    osc: OscConfig = Field(default_factory=OscConfig)

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_network_keys(cls, data):
        """`bind` / `allowed_hosts` were config settings; who may connect is
        now the `--local-only` startup flag. Old keys are dropped."""
        if isinstance(data, dict) and ("bind" in data or "allowed_hosts" in data):
            data = {k: v for k, v in data.items() if k not in ("bind", "allowed_hosts")}
            logger.info("config: ignoring web.bind / web.allowed_hosts -- the web UI is reachable from other "
                        "computers unless narrator is started with --local-only")
        return data

    @model_validator(mode="after")
    def _check_token_names(self) -> "WebConfig":
        names = [t.name for t in self.api_tokens]
        if len(set(names)) != len(names):
            raise ValueError("web.api_tokens names must be unique")
        tokens = [t.token for t in self.api_tokens]
        if len(set(tokens)) != len(tokens):
            raise ValueError("web.api_tokens must not reuse a token")
        return self


class BlocklistAction(str, Enum):
    MASK = "mask"
    REPLACE = "replace"
    DROP = "drop"


class CaptionsConfig(BaseModel):
    """Caption text policy. The blocklist (narrator.captions.blocklist) is
    applied to ASR hypotheses before agreement and to manual captions."""

    # Words that must never reach the screen: whole-word, case- and
    # diacritics-blind; a trailing * matches every word starting with the
    # stem (kurw* -> kurwa, kurwy, kurwą).
    blocklist: list[str] = Field(default_factory=list)
    # Also use the bundled Polish/English starter list.
    blocklist_builtin: bool = False
    # mask: k***a; replace: blocklist_replacement; drop: remove the word.
    blocklist_action: BlocklistAction = BlocklistAction.MASK
    blocklist_replacement: str = "***"

    @field_validator("blocklist")
    @classmethod
    def _check_blocklist(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for entry in value:
            entry = entry.strip()
            if not entry:
                continue
            stem = entry[:-1] if entry.endswith("*") else entry
            if "*" in stem:
                raise ValueError(f"blocklist entry {entry!r}: '*' is only allowed at the end (a stem)")
            if not re.fullmatch(r"[^\W_]+", stem):
                raise ValueError(f"blocklist entry {entry!r} must be a single word (letters/digits)")
            if entry.endswith("*") and len(stem) < 3:
                raise ValueError(f"blocklist stem {entry!r} is too short (at least 3 letters before '*')")
            if entry not in cleaned:
                cleaned.append(entry)
        return cleaned

    @model_validator(mode="after")
    def _check_replacement(self) -> "CaptionsConfig":
        if self.blocklist_action == BlocklistAction.REPLACE and not self.blocklist_replacement.strip():
            raise ValueError(
                "captions.blocklist_replacement must not be empty with blocklist_action: replace "
                "(use blocklist_action: drop to remove words)"
            )
        return self


class RootConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    canvas: CanvasConfig = Field(default_factory=CanvasConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)
    ndi: NdiConfig = Field(default_factory=NdiConfig)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_gui_to_web(cls, data):
        """Accept the retired `gui:` section (the Tk GUI, replaced by the web
        UI): `gui.enabled` becomes `web.enabled` unless `web.enabled` is set
        explicitly, so a config that ran without a UI keeps doing so."""
        if isinstance(data, dict) and "gui" in data:
            data = dict(data)
            gui = data.pop("gui")
            if isinstance(gui, dict) and "enabled" in gui:
                web = dict(data.get("web") or {})
                web.setdefault("enabled", gui["enabled"])
                data["web"] = web
        return data

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_autoconfigure(cls, data):
        """Drop a leftover `autoconfigure:` section: the search moved out of
        the app into the narrator-tune tool, which keeps its settings in its
        own config/tune.yaml (and migrates this section there on first use)."""
        if isinstance(data, dict) and "autoconfigure" in data:
            data = {k: v for k, v in data.items() if k != "autoconfigure"}
            logger.info(
                "Ignoring the retired `autoconfigure:` config section "
                "(narrator-tune keeps these settings in config/tune.yaml now)"
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_features(cls, data):
        """Drop the `transcript:` and `recording:` sections and the
        recording shortcut: both features were removed (2026-11-16)."""
        if not isinstance(data, dict):
            return data
        dropped = [k for k in ("transcript", "recording") if k in data]
        if dropped:
            data = {k: v for k, v in data.items() if k not in dropped}
        web = data.get("web")
        if isinstance(web, dict) and isinstance(web.get("shortcuts"), dict) and "recording_toggle" in web["shortcuts"]:
            shortcuts = {k: v for k, v in web["shortcuts"].items() if k != "recording_toggle"}
            data = {**data, "web": {**web, "shortcuts": shortcuts}}
            dropped.append("web.shortcuts.recording_toggle")
        if dropped:
            logger.info("config: ignoring %s (session recording and the transcript file were removed)",
                        ", ".join(dropped))
        return data

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
