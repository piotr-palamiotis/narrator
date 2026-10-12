"""`narrator --autoconfigure`: find the ASR settings under which the LIVE
captioner agrees best with the reference transcripts in scripts/test_audio/,
on this machine, and write them into config.yaml.

Accuracy is measured by replaying excerpts of the reference library through
the real StreamingDecoder on a virtual clock (stream_sim) -- the captions a
live audience would have seen, at their live timing -- and scoring them
against the SRTs (library_eval). The search, in stages:

  0. Library: references in the wanted language(s) only (an English
     translation track next to Polish audio would score as 100% error),
     split by title hash into dev (tuning) and holdout (confirmation);
     excerpts are decoded once into a disk-guarded cache. An OFFLINE
     transcription of every excerpt (no streaming) gives the ceiling --
     the best whisper agreement with the reference -- and excerpts where
     even that is hopeless (misaligned or mislabelled SRTs) are dropped.
  1. Bench: each locally cached candidate model is timed on this hardware
     (a model that cannot keep up with real time is out), then CALIBRATED:
     a few excerpts are replayed with real decode timings, and the replay's
     decode-cost model is fitted to those calls. Bench windows alone
     underestimated large-v3's streaming decodes by ~30% on the reference
     machine -- streaming windows are speech-dense, and decode time grows
     with the words emitted -- and the delay cap and load limit below are
     only as good as that model.
  2. Model screen: every candidate with the operator's current settings.
  3. Coordinate search on the best model: one setting at a time, a new
     value adopted only when it saves more errors than the noise (paired
     standard error) and keeps the median caption delay within
     `max_latency_s` and the worker under 90% busy.
  4. Confirmation on the full dev set: tuned best vs runner-up model with
     the same settings vs the operator's current config.
  5. Holdout: the winner is written only if it is not worse than the
     current config on recordings the search never saw.

Everything decoded is kept in a resumable result store, so a rerun (or an
interrupted run) only decodes what it has not seen; the budget
(`autoconfigure.time_budget_minutes`) bounds new decoding.

The device is never changed (the configured device is final by policy) and
neither are the language whitelist, display settings or silence timers --
only the model and the decode settings the search measured.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from narrator.asr import library_eval
from narrator.asr.library_eval import (
    CACHE_DIR,
    AudioCache,
    ConfigScore,
    Evaluator,
    Excerpt,
    ModelProfile,
    ResultStore,
    decode_settings,
    paired_delta,
    run_probe_subprocess,
)
from narrator.asr.stream_sim import DecodeCostModel
from narrator.config.loader import load_config, save_config
from narrator.config.schema import AUTOCONFIGURE_SEARCH_SETTINGS, AsrConfig, RootConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_FIXTURES_DIR = REPO_ROOT / "scripts" / "test_audio"
CLIPS_DIR = REPO_ROOT / "scripts" / "test_clips"
REPORTS_DIR = REPO_ROOT / "scripts" / "autoconfigure_reports"
MANIFEST_PATH = REPORTS_DIR / "manifest.json"
COST_MODELS_PATH = CACHE_DIR / "cost_models.json"

# Kept for callers of the previous module API.
_run_probe_subprocess = run_probe_subprocess

DEFAULT_CANDIDATE_MODELS = ("large-v3", "large-v3-turbo", "medium", "small", "base", "tiny")

# The order settings are searched in: biggest expected effect first (the
# rollover length decides how often long narration is cut mid-speech).
SEARCH_ORDER = AUTOCONFIGURE_SEARCH_SETTINGS
DEFAULT_SEARCH_SPACE: dict[str, list] = {
    "max_context_seconds": [10.0, 15.0, 20.0, 28.0],
    "vad_threshold": [0.35, 0.5, 0.65],
    "end_silence_ms": [400, 650, 1000],
    "step_ms": [300, 500, 800, 1200],
    "agreement_passes": [2, 3],
    "vad_preroll_ms": [0, 250],
    "prompt_carryover_chars": [0, 200],
}

# The settings that mostly decide caption delay (searched first when the
# starting point is over the delay cap).
LATENCY_SETTINGS = ("step_ms", "agreement_passes")

# A new setting must save at least this many errors on the screen set (on
# top of beating one paired standard error) to be adopted: below that,
# "better" is indistinguishable from excerpt-sampling noise.
MIN_GAIN_WORDS = 3
NOISE_Z = 1.0
# Worker occupancy above this leaves no headroom: decode jitter turns
# straight into caption delay.
MAX_BUSY_FRACTION = 0.9
# Immediate word repeats surviving into captions mean the decoder's
# repetition collapse is not working for this model/config.
MAX_DOUBLED_RATE = 0.005
# The winner may not be worse than the current config on the holdout set
# by more than this (absolute WER), or nothing is written.
HOLDOUT_TOLERANCE = 0.005
# Model used for the offline ceiling (falls back to the first candidate).
CEILING_MODEL = "large-v3"
# Search candidates are first replayed on this many excerpts (the start of
# the search set, so nothing is wasted for survivors); one that overloads
# the worker or blows the delay cap well beyond noise there is pruned
# before it costs a full evaluation -- slow settings are exactly the most
# expensive to evaluate.
PILOT_EXCERPTS = 4
PILOT_MAX_BUSY = 1.0
PILOT_LATENCY_FACTOR = 1.5
# Excerpts replayed with real timings to calibrate each model's cost model,
# and the fewest decode calls a fit is trusted on.
CALIBRATION_EXCERPTS = 3
MIN_CALIBRATION_CALLS = 20


@dataclass
class Tuning:
    """Autoconfigure's knobs, from config.yaml's `autoconfigure:` section
    (GUI Autoconfigure tab). Defaults mirror the schema's."""

    candidate_models: list[str] = field(default_factory=lambda: list(DEFAULT_CANDIDATE_MODELS))
    slow_factor: float = 1.5
    time_budget_minutes: float = 720.0
    max_latency_s: float | None = 3.0
    excerpt_seconds: float = 90.0
    screen_excerpts: int = 32
    dev_excerpts: int = 96
    holdout_excerpts: int = 48
    reference_languages: list[str] = field(default_factory=lambda: ["pl"])
    max_ceiling_wer: float = 0.85
    min_free_disk_mb: float = 1024.0
    cache_max_mb: float = 1024.0
    search_space: dict[str, list] = field(default_factory=lambda: dict(DEFAULT_SEARCH_SPACE))
    search_passes: int = 2

    @classmethod
    def from_config(cls, config: RootConfig) -> "Tuning":
        section = config.autoconfigure
        space = dict(DEFAULT_SEARCH_SPACE)
        space.update({k: list(v) for k, v in section.search_space.items()})
        return cls(
            candidate_models=list(section.candidate_models),
            slow_factor=section.slow_factor,
            time_budget_minutes=section.time_budget_minutes,
            max_latency_s=section.max_latency_s or None,  # 0 = no cap
            excerpt_seconds=section.excerpt_seconds,
            screen_excerpts=section.screen_excerpts,
            dev_excerpts=section.dev_excerpts,
            holdout_excerpts=section.holdout_excerpts,
            reference_languages=list(section.reference_languages),
            max_ceiling_wer=section.max_ceiling_wer,
            min_free_disk_mb=section.min_free_disk_mb,
            cache_max_mb=section.cache_max_mb,
            search_space=space,
        )

    def quick(self) -> "Tuning":
        """The `--quick` variant: small samples, one search pass, ~1 h -- a
        smoke run of the whole pipeline more than a reliable tuning."""
        return Tuning(
            candidate_models=self.candidate_models,
            slow_factor=self.slow_factor,
            time_budget_minutes=min(self.time_budget_minutes, 60.0),
            max_latency_s=self.max_latency_s,
            excerpt_seconds=self.excerpt_seconds,
            screen_excerpts=min(self.screen_excerpts, 6),
            dev_excerpts=min(self.dev_excerpts, 10),
            holdout_excerpts=min(self.holdout_excerpts, 6),
            reference_languages=self.reference_languages,
            max_ceiling_wer=self.max_ceiling_wer,
            min_free_disk_mb=self.min_free_disk_mb,
            cache_max_mb=self.cache_max_mb,
            search_space=self.search_space,
            search_passes=1,
        )


class Budget:
    """Wall-clock budget for new decoding."""

    def __init__(self, minutes: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._deadline = clock() + minutes * 60.0

    def remaining(self) -> float:
        return self._deadline - self._clock()


# ---------------------------------------------------------------------------
# Pure decision logic (unit-tested without GPU)
# ---------------------------------------------------------------------------


def with_setting(config: AsrConfig, name: str, value) -> AsrConfig:
    """`config` with one decode setting changed, validated. The rollover
    length is one setting in two fields: the decoder rolls over at the
    smaller of max_utterance_ms and max_context_seconds, so both move."""
    update = {name: value}
    if name == "max_context_seconds":
        update["max_utterance_ms"] = int(round(float(value) * 1000))
    return AsrConfig.model_validate({**config.model_dump(), **update})


def feasible(score: ConfigScore, max_latency_s: float | None) -> bool:
    """A configuration the operator could actually run: results for (nearly)
    every excerpt, hardware headroom, caption delay within the cap, and no
    surviving word repeats. A few excerpts the decoder crashed on do not
    disqualify -- comparisons are paired on the excerpts both sides have."""
    if score.excerpts == 0 or score.missing > max(1, 0.1 * (score.excerpts + score.missing)):
        return False
    if score.busy_fraction > MAX_BUSY_FRACTION:
        return False
    if score.doubled_rate > MAX_DOUBLED_RATE:
        return False
    if max_latency_s is not None and score.latency_p50 is not None and score.latency_p50 > max_latency_s:
        return False
    return True


def pilot_rejects(score: ConfigScore, max_latency_s: float | None) -> bool:
    """Clearly unusable on the pilot excerpts (looser than `feasible`: a
    handful of excerpts is noisy)."""
    if score.excerpts == 0:
        return False  # no evidence either way; let the full evaluation decide
    if score.busy_fraction > PILOT_MAX_BUSY:
        return True
    return (
        max_latency_s is not None
        and score.latency_p50 is not None
        and score.latency_p50 > PILOT_LATENCY_FACTOR * max_latency_s
    )


def significant(saved: float, se: float) -> bool:
    """Errors saved beyond excerpt-sampling noise."""
    return saved >= MIN_GAIN_WORDS and saved >= NOISE_Z * se


def pick_setting(
    incumbent: ConfigScore,
    candidates: list[tuple[object, ConfigScore]],
    max_latency_s: float | None,
) -> tuple[object, ConfigScore] | None:
    """The candidate value to adopt, or None to keep the incumbent.

    A feasible incumbent is only replaced by a feasible candidate that saves
    significantly more errors (the largest such saving wins). An infeasible
    incumbent (e.g. the current settings overrun the latency cap) is
    replaced by the feasible candidate with the fewest errors even without
    a gain -- feasibility comes first.
    """
    usable = [(value, score) for value, score in candidates if feasible(score, max_latency_s)]
    if not usable:
        return None
    if not feasible(incumbent, max_latency_s):
        return min(usable, key=lambda vs: vs[1].errors)
    best: tuple[object, ConfigScore] | None = None
    best_saved = 0.0
    for value, score in usable:
        saved, se, _ = paired_delta(incumbent, score)
        if significant(saved, se) and saved > best_saved:
            best, best_saved = (value, score), saved
    return best


def rank_models(screened: list[tuple[str, ConfigScore]]) -> list[str]:
    """Models best-first by screen WER. A model the hardware cannot carry
    (worker over its load limit, results missing, words doubled) ranks
    after the rest; the delay cap is deliberately NOT applied here: delay is
    mostly step_ms and agreement_passes, which the search tunes -- measured
    on the reference machine, large-v3 was the most accurate model while
    sitting just over a 3s cap at step 800."""
    ordered = sorted(
        screened,
        key=lambda ms: (not feasible(ms[1], None), ms[1].wer if ms[1].excerpts else math.inf),
    )
    return [model for model, score in ordered if score.excerpts]


def choose_winner(
    finalists: list[tuple[str, AsrConfig, ConfigScore, float]],
    max_latency_s: float | None,
) -> tuple[str, AsrConfig, ConfigScore] | None:
    """Among (label, config, dev score, decode cost) finalists: the fewest
    dev errors among feasible ones -- except that a faster model within
    noise of the best is preferred (same accuracy, less hardware, lower
    delay)."""
    usable = [f for f in finalists if feasible(f[2], max_latency_s)]
    if not usable:
        return None
    best = min(usable, key=lambda f: f[2].errors)
    for other in sorted(usable, key=lambda f: f[3]):
        if other[3] >= best[3]:
            break
        saved, se, _ = paired_delta(other[2], best[2])
        if not significant(saved, se):
            best = other
            break
    return best[0], best[1], best[2]


def holdout_gate(winner: ConfigScore, baseline: ConfigScore | None) -> bool:
    """Write only if the winner is not worse than the current settings on
    recordings the search never saw."""
    if baseline is None or baseline.excerpts == 0:
        return winner.excerpts > 0
    return winner.excerpts > 0 and winner.wer <= baseline.wer + HOLDOUT_TOLERANCE


def settings_diff(old: AsrConfig, new: AsrConfig) -> dict[str, tuple]:
    """{field: (old, new)} for the model and every decode setting that moved."""
    changes: dict[str, tuple] = {}
    if old.model != new.model:
        changes["model"] = (old.model, new.model)
    old_settings, new_settings = decode_settings(old), decode_settings(new)
    for name, value in new_settings.items():
        if old_settings.get(name) != value:
            changes[name] = (old_settings.get(name), value)
    return changes


def apply_settings(config_path: Path, winner: AsrConfig) -> RootConfig:
    """Write the winner's model and decode settings into config.yaml,
    keeping everything else (device, language whitelist, display, silence
    timers). Validated through the schema, saved atomically, with the
    previous file kept as config.yaml.bak."""
    config = load_config(config_path)
    updates = {"model": winner.model, **decode_settings(winner)}
    updates.pop("language_whitelist", None)  # the operator's decision, never tuned
    asr = AsrConfig.model_validate({**config.asr.model_dump(), **updates})
    updated = RootConfig.model_validate({**config.model_dump(), "asr": asr.model_dump()})
    backup = config_path.with_name(config_path.name + ".bak")
    backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    save_config(updated, config_path)
    return updated


# ---------------------------------------------------------------------------
# Models: availability, bench, cost models
# ---------------------------------------------------------------------------


def model_available(model: str) -> bool:
    """True when pywhispercpp would load `model` without downloading it."""
    if Path(model).is_file():
        return True
    try:
        from pywhispercpp.constants import MODELS_DIR
    except Exception:  # pragma: no cover - broken install
        return False
    models_dir = Path(MODELS_DIR)
    return any(
        (models_dir / name).is_file() for name in (model, f"{model}.bin", f"ggml-{model}.bin")
    )


def _load_cost_models() -> dict:
    try:
        return json.loads(COST_MODELS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cost_models(entries: dict) -> None:
    COST_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = COST_MODELS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(COST_MODELS_PATH)


def slowest_decode_ratio(decode_seconds_by_window: dict[str, float]) -> float:
    """Worst decode-time / window-length ratio across benched windows."""
    ratios = [
        float(seconds) / float(label.rstrip("s"))
        for label, seconds in decode_seconds_by_window.items()
        if float(label.rstrip("s")) > 0
    ]
    return max(ratios) if ratios else math.inf


@dataclass
class Candidate:
    model: str
    profile: ModelProfile | None = None
    bench: dict = field(default_factory=dict)
    error: str | None = None


def profile_model(model: str, device: str, tuning: Tuning, bench_audio: Path | None, rebench: bool) -> Candidate:
    """Bench `model` (or reuse its stored bench) and build its profile.

    Bench results are kept per (model, device) in the cache: re-benching
    every run would give a slightly different cost model each time, and
    with it a different replay -- every cached decode would be wasted.
    """
    entries = _load_cost_models()
    key = f"{model}|{device}"
    bench = None if rebench else entries.get(key)
    if bench is None:
        payload = {"model": model, "device": device}
        if bench_audio is not None:
            payload["audio_npy"] = str(bench_audio)
        bench = run_probe_subprocess("narrator.asr.bench", payload, 900.0)
        if "error" in bench:
            return Candidate(model, error=f"bench failed: {bench['error']}")
        entries[key] = bench
        _save_cost_models(entries)
    timings = {str(k): float(v) for k, v in bench.get("decode_seconds_by_window", {}).items()}
    if not timings:
        return Candidate(model, bench=bench, error="bench produced no timings")
    ratio = slowest_decode_ratio(timings)
    if ratio > tuning.slow_factor:
        return Candidate(
            model,
            bench=bench,
            error=f"too slow for live use on this machine ({ratio:.2f}x real time > {tuning.slow_factor}x)",
        )
    cost = DecodeCostModel.fit(timings, float(bench.get("detect_seconds", 0.0)))
    profile = ModelProfile(model=model, device=device, resolved_device=str(bench.get("resolved_device", "")), cost=cost)
    return Candidate(model, profile=profile, bench=bench)


def calibrate_profile(
    profile: ModelProfile,
    config: AsrConfig,
    excerpts: list[Excerpt],
    evaluator: Evaluator,
    recalibrate: bool = False,
) -> bool:
    """Fit `profile.cost` to real streaming decodes (see the module
    docstring); the fit is stored next to the bench, so later runs reuse it
    -- and with it their cached results. Returns True when calibrated."""
    if not excerpts or excerpts[0].kind == "synthetic":
        return False
    entries = _load_cost_models()
    key = f"{profile.model}|{profile.device}"
    entry = entries.get(key, {})
    if "calibrated" in entry and not recalibrate:
        profile.cost = DecodeCostModel.from_dict(entry["calibrated"])
        return True
    config = config.model_copy(update={"model": profile.model})
    evaluator.evaluate(profile, [config], excerpts, mode="measured")
    calls: list[tuple[float, int, float]] = []
    detects: list[float] = []
    for excerpt in excerpts:
        record = evaluator.record(excerpt, profile, config, "measured")
        if record is not None:
            result = record["result"]
            calls.extend((float(w), int(n), float(t)) for w, n, t in result.get("calls", []))
            detects.extend(float(t) for t in result.get("detects", []))
    if len(calls) < MIN_CALIBRATION_CALLS:
        logger.info("  %s: only %d timed decodes; keeping the bench fit", profile.model, len(calls))
        return False
    detect_s = statistics.median(detects) if detects else profile.cost.detect_s
    bench_cost = profile.cost
    profile.cost = DecodeCostModel.fit_calls(calls, detect_s)
    entry["calibrated"] = profile.cost.to_dict()
    entry["calibration_calls"] = len(calls)
    entries[key] = entry
    _save_cost_models(entries)
    logger.info(
        "  %-16s calibrated on %d decodes: 10s of speech decodes in %.2fs (bench fit said %.2fs)",
        profile.model, len(calls), profile.cost.typical_decode_seconds(), bench_cost.typical_decode_seconds(),
    )
    return True


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """What a run decided, for the CLI table and the summary report."""

    written: bool
    reason: str
    baseline: AsrConfig
    winner: AsrConfig | None
    changes: dict[str, tuple] = field(default_factory=dict)
    table: list[dict] = field(default_factory=list)  # rows: label, set, score
    report_path: Path | None = None
    candidates: list[Candidate] = field(default_factory=list)


class _Run:
    """One autoconfigure run's state; methods are the stages."""

    def __init__(
        self,
        config_path: Path,
        tuning: Tuning,
        budget: Budget,
        evaluator: Evaluator,
        profiles: dict[str, ModelProfile],
        recalibrate: bool = False,
    ) -> None:
        self.recalibrate = recalibrate
        self.config_path = config_path
        self.config = load_config(config_path)
        self.baseline = self.config.asr
        self.tuning = tuning
        self.budget = budget
        self.evaluator = evaluator
        self.profiles = profiles  # every benched model, incl. the current one
        self.ceiling_by_set: dict[str, float] = {}
        self._dev: list[Excerpt] = []
        self.report: dict = {
            "started": library_eval.now_iso(),
            "baseline": self.baseline.model_dump(mode="json"),
            "tuning": dict(tuning.__dict__),
            "stages": {},
        }

    # -- helpers ------------------------------------------------------------

    def _note(self, stage: str, label: str, config: AsrConfig, excerpt_set: str, score: ConfigScore) -> None:
        entry = {
            "label": label,
            "set": excerpt_set,
            "model": config.model,
            "settings": decode_settings(config),
            **score.to_dict(),
        }
        if stage in ("confirm", "holdout", "trusted", "whitelist_check"):
            # Decision-stage detail for later analysis: which excerpts the
            # final candidates fail on ([errors, reference words] per id).
            entry["per_excerpt"] = {k: list(v) for k, v in score.per_excerpt.items()}
        self.report["stages"].setdefault(stage, []).append(entry)
        logger.info(
            "  %-28s %-8s WER %5.1f%% (S%d D%d I%d / %d words) delay p50 %s busy %.0f%%%s",
            label[:28],
            excerpt_set,
            100 * score.wer,
            score.substitutions,
            score.deletions,
            score.insertions,
            score.reference_words,
            "n/a" if score.latency_p50 is None else f"{score.latency_p50:.1f}s",
            100 * score.busy_fraction,
            f" [{score.missing} missing]" if score.missing else "",
        )

    def _evaluate(self, stage: str, label: str, config: AsrConfig, excerpts: list[Excerpt], excerpt_set: str) -> ConfigScore:
        (score,) = self.evaluator.evaluate(self.profiles[config.model], [config], excerpts)
        self._note(stage, label, config, excerpt_set, score)
        return score

    def _affordable(self, profile: ModelProfile, configs: list[AsrConfig], excerpts: list[Excerpt], reserve_s: float = 0.0) -> bool:
        need = self.evaluator.estimate_seconds(profile, configs, excerpts)
        return need == 0.0 or need <= self.budget.remaining() - reserve_s

    def _final_stages_seconds(self, tuned: AsrConfig, runner_up: AsrConfig | None, dev, holdout) -> float:
        """Wall time the confirmation and holdout stages still need."""
        estimate = self.evaluator.estimate_seconds
        need = estimate(self.profiles[tuned.model], [tuned], dev) + estimate(self.profiles[tuned.model], [tuned], holdout)
        if runner_up is not None:
            need += estimate(self.profiles[runner_up.model], [runner_up], dev)
        if self.baseline.model in self.profiles:
            need += estimate(self.profiles[self.baseline.model], [self.baseline], dev)
            need += estimate(self.profiles[self.baseline.model], [self.baseline], holdout)
        return need

    # -- stages -------------------------------------------------------------

    def excerpts(self) -> tuple[list[Excerpt], list[Excerpt], list[Excerpt], list[Excerpt]]:
        """(screen, dev, holdout, trusted) excerpt sets, audio cached."""
        t = self.tuning
        library = library_eval.scan_library(REAL_FIXTURES_DIR, t.reference_languages)
        if not library.recordings:
            if not (CLIPS_DIR / "manifest.json").exists():
                raise RuntimeError(
                    f"no reference recordings in {REAL_FIXTURES_DIR} and no synthetic clips in "
                    f"{CLIPS_DIR} (run scripts/gen_test_clips.py)"
                )
            clips = library_eval.synthetic_excerpts(CLIPS_DIR)
            logger.info("No real recordings found; using %d synthetic Piper clips", len(clips))
            self.report["library"] = {"kind": "synthetic", "clips": len(clips)}
            return clips, clips, [], []
        dev = library_eval.sample_excerpts(library.recordings, t.dev_excerpts, t.excerpt_seconds, split="dev")
        holdout = library_eval.sample_excerpts(library.recordings, t.holdout_excerpts, t.excerpt_seconds, split="holdout")
        trusted = library_eval.sample_excerpts(library.trusted, min(8, t.holdout_excerpts), t.excerpt_seconds)
        screen = dev[: t.screen_excerpts]
        logger.info(
            "Library: %d recordings scored, %d with human subtitles (a separate check), %d excluded",
            len(library.recordings), len(library.trusted), len(library.excluded),
        )
        for name, reason in library.excluded:
            logger.info("  excluded: %s (%s)", name, reason)
        logger.info(
            "Excerpts: %d screen / %d dev / %d holdout / %d trusted, %.0fs each",
            len(screen), len(dev), len(holdout), len(trusted), t.excerpt_seconds,
        )
        self.report["library"] = {
            "kind": "real",
            "recordings": len(library.recordings),
            "trusted": len(library.trusted),
            "excluded": library.excluded,
            "excerpts": {"screen": len(screen), "dev": len(dev), "holdout": len(holdout), "trusted": len(trusted)},
        }
        logger.info("Preparing excerpt audio (decoded once, cached in %s)...", self.evaluator.cache.directory)
        self.evaluator.cache.prepare(dev + holdout + trusted, progress=lambda message: logger.info("  %s", message))
        return screen, dev, holdout, trusted

    def drop_unmatchable(self, ceiling_profile: ModelProfile, sets: list[list[Excerpt]]) -> list[list[Excerpt]]:
        """Offline ceiling; excerpts even it cannot match are dropped."""
        everything = list({x.id: x for group in sets for x in group}.values())
        if not everything or everything[0].kind == "synthetic":
            return sets
        logger.info("Offline ceiling (%s, whole-excerpt transcription, %d excerpts)...", ceiling_profile.model, len(everything))
        language = self.tuning.reference_languages[0] if self.tuning.reference_languages else None
        (ceiling,) = self.evaluator.evaluate(ceiling_profile, [self.baseline], everything, mode="offline", language=language)
        unmatchable = {
            excerpt_id
            for excerpt_id, (errors, words) in ceiling.per_excerpt.items()
            if words and errors / words > self.tuning.max_ceiling_wer
        }
        self.report["ceiling"] = {"model": ceiling_profile.model, "unmatchable": sorted(unmatchable), **ceiling.to_dict()}
        if unmatchable:
            logger.info(
                "  %d excerpt(s) dropped: even offline %s disagrees with their reference by more than %.0f%%",
                len(unmatchable), ceiling_profile.model, 100 * self.tuning.max_ceiling_wer,
            )
        kept = [[x for x in group if x.id not in unmatchable] for group in sets]
        for name, group in zip(("screen", "dev", "holdout", "trusted"), kept):
            ids = {x.id for x in group}
            pairs = [v for k, v in ceiling.per_excerpt.items() if k in ids]
            words = sum(w for _, w in pairs)
            if words:
                self.ceiling_by_set[name] = sum(e for e, _ in pairs) / words
        logger.info(
            "  ceiling WER: %s",
            ", ".join(f"{name} {100 * wer:.1f}%" for name, wer in self.ceiling_by_set.items()) or "n/a",
        )
        return kept

    def run(self, search_models: list[str]) -> Outcome:
        outcome = self._decide(search_models)
        final = outcome.winner if outcome.written and outcome.winner is not None else self.baseline
        if self._dev:
            self.whitelist_check(final, self._dev)
        return outcome

    def whitelist_check(self, config: AsrConfig, dev: list[Excerpt]) -> None:
        """Report-only: the final settings again, restricted to the scored
        reference language. A multi-language whitelist costs detection time
        and -- on audio that is all one language -- wrong-language captions
        (measured on the reference library: music and cabaret excerpts
        decoded as English). Whether that is worth it is the operator's
        call; the whitelist is never changed by autoconfigure."""
        reference = self.tuning.reference_languages[:1]
        if len(config.language_whitelist) <= 1 or not reference or reference[0] not in config.language_whitelist:
            return
        if config.model not in self.profiles:
            return
        single = config.model_copy(update={"language_whitelist": reference})
        if not self._affordable(self.profiles[config.model], [single], dev):
            logger.info("Whitelist check skipped: not enough budget left")
            return
        logger.info("Whitelist check (report only): the final settings with language_whitelist %s:", reference)
        self._evaluate("whitelist_check", f"same, whitelist {reference}", single, dev, "dev")

    def _decide(self, search_models: list[str]) -> Outcome:
        t = self.tuning
        if not search_models:
            raise RuntimeError("no candidate model could be benchmarked; see the per-model messages above")
        screen, dev, holdout, trusted = self.excerpts()
        self._dev = dev
        if screen and screen[0].kind != "synthetic":
            logger.info("Calibrating decode-cost models (%d excerpts replayed with real timings)...", CALIBRATION_EXCERPTS)
            for profile in self.profiles.values():
                calibrate_profile(profile, self.baseline, screen[:CALIBRATION_EXCERPTS], self.evaluator, self.recalibrate)
            self.report["cost_models"] = {m: p.cost.to_dict() for m, p in self.profiles.items()}
        ceiling_profile = self.profiles.get(CEILING_MODEL) or self.profiles[search_models[0]]
        screen, dev, holdout, trusted = self.drop_unmatchable(ceiling_profile, [screen, dev, holdout, trusted])
        if not screen:
            raise RuntimeError("no usable excerpts left to evaluate")
        self._dev = dev
        has_baseline = self.baseline.model in self.profiles

        # Stage 2: every candidate model with the current settings.
        logger.info("Model screen (current settings, %d excerpts):", len(screen))
        screened: list[tuple[str, ConfigScore]] = []
        for model in search_models:
            config = self.baseline.model_copy(update={"model": model})
            if not self._affordable(self.profiles[model], [config], screen):
                logger.info("  %s: skipped, not enough budget left", model)
                continue
            screened.append((model, self._evaluate("model_screen", model, config, screen, "screen")))
        ranking = rank_models(screened)
        if not ranking:
            raise RuntimeError("no model produced results on the screen set")
        best_model = ranking[0]
        runner_up_model = ranking[1] if len(ranking) > 1 else None
        logger.info(
            "Best model with the current settings: %s%s",
            best_model, f" (runner-up {runner_up_model})" if runner_up_model else "",
        )

        # Stage 3: coordinate search on the best model, then (budget
        # permitting) one pass on the runner-up from the tuned settings.
        tuned = self.search(self.baseline.model_copy(update={"model": best_model}), screen, dev, holdout)
        tuned_runner = tuned.model_copy(update={"model": runner_up_model}) if runner_up_model else None
        if tuned_runner is not None:
            spare = self.budget.remaining() - self._final_stages_seconds(tuned, tuned_runner, dev, holdout)
            if spare > 0:
                logger.info("Budget left: one search pass on the runner-up model %s", runner_up_model)
                tuned_runner = self.search(tuned_runner, screen, dev, holdout, passes=1)

        # Stage 4: confirmation on the full dev set.
        logger.info("Confirmation on the dev set (%d excerpts):", len(dev))
        finalists: list[tuple[str, AsrConfig, ConfigScore, float]] = []
        entries: list[tuple[str, AsrConfig]] = [("tuned " + tuned.model, tuned)]
        if tuned_runner is not None:
            entries.append(("tuned " + tuned_runner.model, tuned_runner))
        if has_baseline:
            entries.append(("current settings", self.baseline))
        dev_scores: dict[str, ConfigScore] = {}
        for label, config in entries:
            optional = label.startswith("tuned ") and config is tuned_runner
            if optional and not self._affordable(self.profiles[config.model], [config], dev):
                logger.info("  %s: skipped, not enough budget left", label)
                continue
            score = self._evaluate("confirm", label, config, dev, "dev")
            dev_scores[label] = score
            finalists.append((label, config, score, self.profiles[config.model].cost.typical_decode_seconds()))
        choice = choose_winner(finalists, t.max_latency_s)
        if choice is None:
            return self._finish(False, "no configuration met the delay cap and headroom limits on the dev set", None)
        winner_label, winner, winner_dev = choice
        if winner_label == "current settings" or not settings_diff(self.baseline, winner):
            return self._finish(False, "the current settings are already the best found", winner)
        baseline_dev = dev_scores.get("current settings")
        if baseline_dev is not None and feasible(baseline_dev, t.max_latency_s):
            saved, se, _ = paired_delta(baseline_dev, winner_dev)
            if not significant(saved, se):
                return self._finish(
                    False,
                    f"no significant gain over the current settings on dev ({saved:+d} errors, noise ±{se:.0f})",
                    winner,
                )

        # Stage 5: holdout -- recordings the search never saw.
        if holdout:
            logger.info("Holdout check (%d excerpts from recordings the search never saw):", len(holdout))
            winner_holdout = self._evaluate("holdout", "winner", winner, holdout, "holdout")
            baseline_holdout = (
                self._evaluate("holdout", "current settings", self.baseline, holdout, "holdout") if has_baseline else None
            )
            if not holdout_gate(winner_holdout, baseline_holdout):
                detail = (
                    f"{100 * winner_holdout.wer:.1f}% vs {100 * baseline_holdout.wer:.1f}% for the current settings"
                    if baseline_holdout is not None
                    else "no results"
                )
                return self._finish(False, f"the winner did not hold up on the holdout set ({detail})", winner)
        if trusted:
            checks = [("winner", winner)] + ([("current settings", self.baseline)] if has_baseline else [])
            need = sum(
                self.evaluator.estimate_seconds(self.profiles[c.model], [c], trusted) for _, c in checks
            )
            if need <= self.budget.remaining():
                logger.info("Human-subtitle check (%d excerpts, report only):", len(trusted))
                for label, config in checks:
                    self._evaluate("trusted", label, config, trusted, "trusted")
            else:
                logger.info("Human-subtitle check skipped: not enough budget left")
        return self._finish(True, "better than the current settings on dev and holdout", winner)

    def search(self, start: AsrConfig, screen, dev, holdout, passes: int | None = None) -> AsrConfig:
        """Coordinate search from `start`; returns the best config found."""
        t = self.tuning
        profile = self.profiles[start.model]
        incumbent = start
        logger.info("Settings search on %s (%d excerpts):", start.model, len(screen))
        incumbent_score = self._evaluate("search", "start", incumbent, screen, "screen")
        for _ in range(passes if passes is not None else t.search_passes):
            moved = False
            order = list(SEARCH_ORDER)
            if not feasible(incumbent_score, t.max_latency_s):
                # Over the delay cap: try the delay levers first, so the
                # remaining settings are judged from a usable starting point.
                order.sort(key=lambda name: name not in LATENCY_SETTINGS)
            for name in order:
                if name not in t.search_space or not hasattr(incumbent, name):
                    continue
                current = getattr(incumbent, name)
                options: list[tuple[object, AsrConfig]] = []
                for value in t.search_space[name]:
                    if value == current:
                        continue
                    try:
                        options.append((value, with_setting(incumbent, name, value)))
                    except ValidationError:
                        continue
                if not options:
                    continue
                reserve = self._final_stages_seconds(incumbent, None, dev, holdout)
                if not self._affordable(profile, [c for _, c in options], screen, reserve_s=reserve):
                    logger.info("  budget reached; stopping the search here")
                    return incumbent
                if len(screen) > PILOT_EXCERPTS:
                    pilot = self.evaluator.evaluate(profile, [c for _, c in options], screen[:PILOT_EXCERPTS])
                    kept = []
                    for (value, config), score in zip(options, pilot):
                        if pilot_rejects(score, t.max_latency_s):
                            self._note("search", f"{name}={value} (pruned by pilot)", config, "pilot", score)
                        else:
                            kept.append((value, config))
                    options = kept
                    if not options:
                        continue
                scores = self.evaluator.evaluate(profile, [c for _, c in options], screen)
                for (value, config), score in zip(options, scores):
                    self._note("search", f"{name}={value}", config, "screen", score)
                picked = pick_setting(incumbent_score, [(v, s) for (v, _), s in zip(options, scores)], t.max_latency_s)
                if picked is not None:
                    value, incumbent_score = picked
                    logger.info("  -> adopting %s=%s", name, value)
                    incumbent = with_setting(incumbent, name, value)
                    moved = True
            if not moved:
                break
        return incumbent

    def _finish(self, written: bool, reason: str, winner: AsrConfig | None) -> Outcome:
        changes = settings_diff(self.baseline, winner) if winner is not None else {}
        self.report["decision"] = {
            "written": written,
            "reason": reason,
            "winner": None if winner is None else winner.model_dump(mode="json"),
            "changes": {k: list(v) for k, v in changes.items()},
            "ceiling_wer": self.ceiling_by_set,
        }
        self.report["finished"] = library_eval.now_iso()
        return Outcome(written=written, reason=reason, baseline=self.baseline, winner=winner, changes=changes)


def autoconfigure(
    config_path: Path,
    models: list[str] | None = None,
    *,
    quick: bool = False,
    budget_minutes: float | None = None,
    dry_run: bool = False,
    allow_download: bool = False,
    rebench: bool = False,
    evaluator: Evaluator | None = None,
) -> Outcome:
    """Run the staged search and (unless `dry_run` or a gate fails) write
    the winning model + decode settings into `config_path`."""
    config = load_config(config_path)
    tuning = Tuning.from_config(config)
    if quick:
        tuning = tuning.quick()
    if budget_minutes is not None:
        tuning.time_budget_minutes = budget_minutes
    budget = Budget(tuning.time_budget_minutes)
    if evaluator is None:
        cache = AudioCache(CACHE_DIR / "excerpts", max_mb=tuning.cache_max_mb, min_free_mb=tuning.min_free_disk_mb)
        evaluator = Evaluator(cache, ResultStore())
    device = config.asr.device.value

    chosen = list(dict.fromkeys(list(models or tuning.candidate_models) + [config.asr.model]))
    candidates: list[Candidate] = []
    bench_audio = _bench_audio_path(evaluator, tuning)
    logger.info("Benchmarking %d model(s) on this machine: %s", len(chosen), ", ".join(chosen))
    for model in chosen:
        if not allow_download and not model_available(model):
            candidates.append(Candidate(model, error="not downloaded (pass --allow-download to fetch it)"))
            logger.info("  %-16s skipped: not in the local model cache", model)
            continue
        candidate = profile_model(model, device, tuning, bench_audio, rebench)
        candidates.append(candidate)
        if candidate.error:
            logger.info("  %-16s %s", model, candidate.error)
        else:
            cost = candidate.profile.cost
            logger.info(
                "  %-16s 10s of speech decodes in %.2fs, language detection %.2fs",
                model, cost.typical_decode_seconds(), cost.detect_s,
            )
    # Models asked for explicitly (--models) are the candidates; the
    # current model is benched too so the current settings can be scored.
    if models:
        search_candidates = [c for c in candidates if c.model in models]
    else:
        search_candidates = [c for c in candidates if c.model in tuning.candidate_models]

    profiles = {c.model: c.profile for c in candidates if c.profile is not None}
    run = _Run(config_path, tuning, budget, evaluator, profiles, recalibrate=rebench)
    outcome = run.run([c.model for c in search_candidates if c.profile is not None])
    outcome.candidates = candidates
    run.report["candidates"] = [
        {"model": c.model, "error": c.error, "bench": c.bench, "cost": c.profile.cost.to_dict() if c.profile else None}
        for c in candidates
    ]
    run.report["budget_minutes"] = tuning.time_budget_minutes
    run.report["decode_minutes_spent"] = round(evaluator.decode_seconds_spent / 60.0, 1)
    if outcome.written and not dry_run and outcome.winner is not None:
        apply_settings(config_path, outcome.winner)
    elif outcome.written and dry_run:
        outcome.written = False
        outcome.reason += " (dry run: config.yaml not changed)"
        run.report["decision"]["reason"] = outcome.reason
        run.report["decision"]["written"] = False
    outcome.table = [
        row
        for stage in ("confirm", "holdout", "trusted", "whitelist_check")
        for row in run.report["stages"].get(stage, [])
    ]
    from narrator.asr.report_library import save_summary

    outcome.report_path = save_summary(run.report, REAL_FIXTURES_DIR)
    return outcome


def _bench_audio_path(evaluator: Evaluator, tuning: Tuning) -> Path | None:
    """A cached real excerpt to bench on (decode time depends on how much
    speech a window holds; a synthetic tone is the last resort)."""
    try:
        library = library_eval.scan_library(REAL_FIXTURES_DIR, tuning.reference_languages)
        sample = library_eval.sample_excerpts(library.recordings, 1, tuning.excerpt_seconds, split="dev")
    except OSError:
        return None
    if not sample:
        return None
    return evaluator.cache.prepare(sample).get(sample[0].id)
