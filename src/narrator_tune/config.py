"""narrator-tune's own settings: config/tune.yaml.

These used to be the `autoconfigure:` section of narrator's config.yaml.
The live pipeline never read them, so they moved out with the tool; the
narrator schema drops a leftover `autoconfigure:` key. tune.yaml is created
on first use, from such a leftover section when the narrator config still
has one (so an existing setup keeps its tuning), else from the template
tune_defaults.yaml next to this module.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from narrator_tune.paths import TUNE_CONFIG_PATH

logger = logging.getLogger(__name__)

DEFAULTS_PATH = Path(__file__).with_name("tune_defaults.yaml")

# Decode settings the search may change (asr.<name>), in search order;
# `search_space` can override their value grids. `decoder` comes first: the
# decoder shoot-out (todo C5) -- local_agreement vs segmenter -- decides
# which of the decoder-specific settings below matter at all (the search
# skips the other decoder's; autoconfigure.DECODER_SPECIFIC).
SEARCH_SETTINGS = (
    "decoder",
    "max_context_seconds",
    "vad_threshold",
    "end_silence_ms",
    "step_ms",
    "agreement_passes",
    "vad_preroll_ms",
    "prompt_carryover_chars",
    "max_timestamp_ratio",
    "min_mean_token_p",
    # segmenter only
    "max_segment_ms",
    "segmentation_mode",
    "partial_latency_ms",
    "context_sentences",
)


class TuneConfig(BaseModel):
    """Tuning for `narrator-tune configure`, which searches the model and
    decode settings (asr.*) under which live captions best match the
    reference transcripts in scripts/test_audio/, and writes the winner into
    narrator's config.yaml.
    """

    # Models considered. Ones missing from the local pywhispercpp cache are
    # skipped (no surprise multi-GB downloads unless --allow-download).
    # `narrator-tune configure --models ...` overrides this list for one run.
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
    # Disk guards for the decoded-excerpt cache (var/tune/cache).
    min_free_disk_mb: float = Field(default=1024.0, ge=0.0)
    cache_max_mb: float = Field(default=1024.0, gt=0.0)
    # Optional per-setting value grids replacing the built-in ones, e.g.
    # {"step_ms": [400, 600, 900]}, {"decoder": ["segmenter"]}.
    search_space: dict[str, list[int | float | str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_bounds(self) -> "TuneConfig":
        if self.screen_excerpts > self.dev_excerpts:
            raise ValueError(
                "screen_excerpts must be <= dev_excerpts "
                "(the screen set is the first part of the dev set)"
            )
        unknown = sorted(set(self.search_space) - set(SEARCH_SETTINGS))
        if unknown:
            raise ValueError(f"search_space has unknown settings {unknown}; known: {list(SEARCH_SETTINGS)}")
        empty = sorted(name for name, values in self.search_space.items() if not values)
        if empty:
            raise ValueError(f"search_space entries need at least one value: {empty}")
        from narrator.asr.whisper_languages import whisper_language_codes

        valid = set(whisper_language_codes())
        for code in self.reference_languages:
            if code not in valid:
                raise ValueError(f"reference_languages entry {code!r} is not a language whisper supports")
        return self


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return raw


def ensure_tune_config(path: Path = TUNE_CONFIG_PATH, narrator_config_path: Path | None = None) -> None:
    """Create `path` if missing: from a leftover `autoconfigure:` section of
    the narrator config at `narrator_config_path` when there is a valid one
    (so an existing setup keeps its tuning), else from the template."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    leftover = None
    if narrator_config_path is not None and narrator_config_path.exists():
        try:
            leftover = _read_yaml(narrator_config_path).get("autoconfigure")
            if isinstance(leftover, dict):
                TuneConfig.model_validate(leftover)
            else:
                leftover = None
        except (OSError, ValueError, yaml.YAMLError) as exc:  # ValidationError is a ValueError
            logger.warning("Ignoring the leftover `autoconfigure:` section of %s: %s", narrator_config_path, exc)
            leftover = None
    if leftover is None:
        path.write_text(DEFAULTS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Created %s from the defaults", path)
        return
    header = (
        "# Settings for `narrator-tune configure`; schema: narrator_tune.config.TuneConfig\n"
        f"# (template with comments: {DEFAULTS_PATH}).\n"
        f"# Migrated from the `autoconfigure:` section of {narrator_config_path}.\n"
    )
    path.write_text(header + yaml.safe_dump(leftover, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("Created %s from the leftover `autoconfigure:` section of %s", path, narrator_config_path)


def load_tune_config(path: Path = TUNE_CONFIG_PATH, narrator_config_path: Path | None = None) -> TuneConfig:
    """The tool's settings from `path`, created first if missing (see
    ensure_tune_config). Raises ValidationError on an invalid file."""
    ensure_tune_config(path, narrator_config_path)
    return TuneConfig.model_validate(_read_yaml(path))
