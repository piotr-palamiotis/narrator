"""`narrator autoconfigure`: benchmark candidate ASR models on this machine,
pick the best model+device for *live* captioning, and write the choice into
the user's config.yaml so it persists.

Ranking criterion (accuracy first, then live performance):
  1. Every candidate first has to *qualify*: no crash, no decode step slower
     than 1.5x real time at any probed window, and WER under the floor.
  2. Among qualifiers, the highest combined score wins: 70% speed (10s-window
     decode time, lower better) + 30% accuracy (WER on the Polish fixtures).

Why both gates: the large-v3 lesson from the stress test -- it's accurate
offline, but before the sentence-loop collapse fix (streaming_decoder.py
`_collapse_repetitions`) its streaming decode hit repetition loops that
slowed steps and collapsed live accuracy (~74% WER vs medium's ~8%). An
autoconfigurer that benchmarked offline quality alone would have picked it.
With that fix in place large-v3 now streams cleanly (measured: 0.06x
real-time decode at a 10s window, ~2% streaming WER on the fixtures) and can
legitimately win on capable hardware; the gates remain as regression guards
-- a decode step slower than 1.5x real time at any probed window still means
the queue-saturation regime, and a streaming WER over the floor still means
garbled captions, whichever model causes them.

Pure decision logic (`grade_candidate`, `decide`, `derive_step_ms`) is
testable without touching GPU or models: see tests/test_autoconfigure.py.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from narrator.config.loader import load_config, save_config
from narrator.config.schema import AsrDevice

if TYPE_CHECKING:
    from narrator.config.schema import RootConfig

logger = logging.getLogger(__name__)

# Models considered, in quality-tier order (fast/cheap ones last). Candidates
# missing from the local pywhispercpp cache are skipped by the bench
# subprocess failing fast, so listing them all is harmless.
CANDIDATE_MODELS = ("large-v3", "medium", "small", "base", "tiny")

# ---------------------------------------------------------------------------
# Decision constants. Documented where used; tuned from the stress-harness
# measurements on the reference machine (RX 6700 XT, see CLAUDE.md).
# ---------------------------------------------------------------------------
# Decode step slower than 1.5x real time at any probed window -> disqualified:
# that is the onset of the queue-saturation regime that destroyed large-v3.
SLOW_FACTOR = 1.5
# WER above this disqualifies a candidate outright. Originally sized when
# large-v3's streaming decode hit 0.74 pre-collapse-fix; it stays as a
# regression net (nothing legitimately useful lives above 0.30).
WER_FLOOR = 0.30
# Worst-window decode ratio used for the speed term.
SPEED_WINDOW_LABEL = "10.0s"
# 10s decode considered "perfect" for the speed term (reference-machine
# measurements: medium 0.33s, large-v3 0.58s -- both cap at 1.0, so the
# score then differentiates on accuracy; slower machines spread out).
SPEED_REFERENCE_S = 0.6
# Score weights: speed dominates because all qualifiers are already accurate
# enough; latency is the user-visible differentiator for live captions.
SPEED_WEIGHT = 0.7
ACCURACY_WEIGHT = 0.3
# step_ms bounds -- the range the stress harness validated on the reference
# hardware (350ms measured best; 250..800 covers slower machines and CPUs).
STEP_MIN_MS = 250
STEP_MAX_MS = 800
# Headroom over the measured 10s decode time: decode steps must fit inside
# the cadence or the decoder self-overruns and warns (streaming_decoder.py).
STEP_SAFETY_FACTOR = 1.35


@dataclass
class BenchResult:
    """One candidate's measurements, graded by `grade_candidate`."""

    model: str
    device: str
    resolved_device: str = ""
    error: str | None = None
    # median decode seconds per window size (label like "10.0s")
    decode_seconds_by_window: dict[str, float] = field(default_factory=dict)
    # quality probe results (None = probe not run/failed)
    wer: float | None = None
    doubled: bool = False
    qualified: bool = False
    score: float = float("-inf")

    @property
    def slowest_decode_ratio(self) -> float:
        """Worst decode-time / window-length ratio across probed windows."""
        if not self.decode_seconds_by_window:
            return float("inf")
        ratios = []
        for label, seconds in self.decode_seconds_by_window.items():
            window = float(label.rstrip("s"))
            if window > 0:
                ratios.append(seconds / window)
        return max(ratios) if ratios else float("inf")


@dataclass
class Decision:
    model: str
    device: str
    step_ms: int
    reason: str
    candidates: list["BenchResult"] = field(default_factory=list)


def _run_probe_subprocess(module: str, payload: dict, timeout_s: float) -> dict:
    """Run a bench/quality-probe worker module in a fresh subprocess and parse
    its `BENCH_JSON:` line. Any failure becomes {"error": ...} rather than an
    exception -- a broken candidate must disqualify itself, not crash the run.

    Fresh process per candidate: a native crash or VRAM exhaustion from one
    model cannot leak into the next.
    """
    command = [sys.executable, "-m", module, json.dumps(payload)]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"{module} timed out after {timeout_s:.0f}s"}

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("BENCH_JSON:"):
            try:
                return json.loads(line[len("BENCH_JSON:") :])
            except json.JSONDecodeError:
                return {"error": f"{module} produced unparseable JSON"}
    return {"error": f"{module} exited {completed.returncode} with no BENCH_JSON line"}


def run_bench_subprocess(
    model: str, device: str, windows_s: list[float], timeout_s: float = 240.0
) -> dict:
    """Run bench.py in a fresh subprocess (see _run_probe_subprocess)."""
    return _run_probe_subprocess(
        "narrator.asr.bench",
        {"model": model, "device": device, "windows_s": windows_s},
        timeout_s,
    )


def grade_candidate(result: BenchResult) -> None:
    """Set `qualified` and `score` on a BenchResult (in place).

    Score = SPEED_WEIGHT * speed_term + ACCURACY_WEIGHT * accuracy_term, both
    in [0, 1]:
        speed term    = min(1, SPEED_REFERENCE_S / s10)
                        (s10 = 10s-window decode seconds; faster -> higher)
        accuracy term = 1 - min(1, wer / WER_FLOOR)
                        (WER 0 -> 1.0, WER >= WER_FLOOR -> 0)

    Candidates with a bench error, no measurements, a decode ratio over
    SLOW_FACTOR, WER over WER_FLOOR, or doubled words in the probe's FINAL
    text do not qualify.
    """
    if result.error or not result.decode_seconds_by_window:
        result.qualified = False
        result.score = float("-inf")
        return

    ratio = result.slowest_decode_ratio
    if ratio > SLOW_FACTOR:
        result.qualified = False
        result.score = float("-inf")
        return

    if result.wer is not None and result.wer > WER_FLOOR:
        result.qualified = False
        result.score = float("-inf")
        return

    # The probe's `doubled` flag means immediate repeated words survived into
    # FINAL captions. The streaming decoder collapses those; if they appear
    # anyway, the text-layer fix is absent or broken and this measurement
    # does not describe the shipping decoder. Disqualify rather than rank.
    if result.doubled:
        result.qualified = False
        result.score = float("-inf")
        return

    s10 = result.decode_seconds_by_window.get(SPEED_WINDOW_LABEL, float("inf"))
    accuracy_term = 1.0 - min(1.0, (result.wer if result.wer is not None else WER_FLOOR) / WER_FLOOR)
    speed_term = min(1.0, SPEED_REFERENCE_S / s10) if s10 > 0 else 0.0
    result.score = SPEED_WEIGHT * speed_term + ACCURACY_WEIGHT * accuracy_term
    result.qualified = True


def decide(candidates: list[BenchResult]) -> Decision:
    """Pure decision over graded candidates. Raises RuntimeError when nothing
    qualifies -- the caller surfaces that instead of writing a bad config."""
    qualified = [c for c in candidates if c.qualified]
    if not qualified:
        raise RuntimeError(
            "no ASR model qualified on this machine; ensure at least one whisper "
            "model is available and the Vulkan runtime (or CPU fallback) works"
        )
    winner = max(qualified, key=lambda c: c.score)

    s10 = winner.decode_seconds_by_window.get(SPEED_WINDOW_LABEL, float("nan"))
    wer_text = f"{winner.wer:.1%}" if winner.wer is not None else "not measured"
    reason = (
        f"{winner.model} on {winner.resolved_device or winner.device}: "
        f"10s decode {s10:.2f}s ({winner.slowest_decode_ratio:.2f}x real time worst window), "
        f"WER {wer_text}, score {winner.score:.3f}"
    )
    return Decision(
        model=winner.model,
        device=winner.device,
        step_ms=derive_step_ms(winner.decode_seconds_by_window),
        reason=reason,
        candidates=candidates,
    )


def derive_step_ms(decode_seconds_by_window: dict[str, float]) -> int:
    """step_ms sized so a decode step fits the cadence with headroom.

    A decode step slower than step_ms makes the decoder lag behind real time
    (streaming_decoder.py warns); the step must also stay above the decode
    time itself so consecutive ticks don't pile up. We derive from the 10s
    window (the decoder's practical steady-state window under continuous
    speech), multiply by a safety factor, and clamp to the validated range.
    """
    s10 = decode_seconds_by_window.get(SPEED_WINDOW_LABEL)
    if s10 is None:
        return _round_step(STEP_MIN_MS)
    raw_ms = s10 * 1000.0 * STEP_SAFETY_FACTOR
    return _round_step(max(STEP_MIN_MS, min(STEP_MAX_MS, int(raw_ms))))


def _round_step(step_ms: int) -> int:
    """Round step_ms to 50ms granularity (cosmetic; stable configs across runs)."""
    return int(round(step_ms / 50.0) * 50)


def apply_decision(config_path: Path, decision: Decision) -> "RootConfig":
    """Write model/device/step_ms into config.yaml (atomic write via
    loader.save_config), preserving every other field. Note there is
    deliberately no per-model quality tier here: the WER probe decides.
    """
    config = load_config(config_path)
    updated_asr = config.asr.model_copy(
        update={
            "model": decision.model,
            "device": decision.device,
            "step_ms": decision.step_ms,
        }
    )
    updated = config.model_copy(update={"asr": updated_asr})
    save_config(updated, config_path)
    return updated


def autoconfigure(config_path: Path, models: list[str] | None = None) -> Decision:
    """Benchmark candidates (bench + quality probe subprocesses), decide, and
    write the winning config. Returns the Decision for CLI display.
    """
    chosen_models = models or list(CANDIDATE_MODELS)

    candidates: list[BenchResult] = []
    for model in chosen_models:
        result = _measure_candidate(model)
        grade_candidate(result)
        candidates.append(result)
        if result.error:
            logger.info("candidate %-12s failed: %s", model, result.error)
        else:
            logger.info(
                "candidate %-12s 10s=%.2fs ratio=%.2fx WER=%s",
                model,
                result.decode_seconds_by_window.get(SPEED_WINDOW_LABEL, float("nan")),
                result.slowest_decode_ratio,
                f"{result.wer:.1%}" if result.wer is not None else "n/a",
            )

    decision = decide(candidates)
    apply_decision(config_path, decision)
    return decision


def _measure_candidate(model: str) -> BenchResult:
    """Bench + quality-probe one model, each in its own subprocess."""
    result = BenchResult(model=model, device="auto")

    bench_payload = run_bench_subprocess(model, "auto", [2.0, 5.0, 10.0])
    if "error" in bench_payload:
        result.error = bench_payload["error"]
        return result
    result.resolved_device = str(bench_payload.get("resolved_device", ""))
    result.decode_seconds_by_window = {
        str(k): float(v) for k, v in bench_payload.get("decode_seconds_by_window", {}).items()
    }

    probe_payload = _run_quality_probe(model)
    if "error" not in probe_payload:
        result.wer = float(probe_payload["wer"])  # type: ignore[arg-type]
        result.doubled = bool(probe_payload.get("doubled", False))

    return result


def _run_quality_probe(model: str, timeout_s: float = 300.0) -> dict:
    """Run quality_probe.py in a fresh subprocess (see _run_probe_subprocess)."""
    return _run_probe_subprocess(
        "narrator.asr.quality_probe",
        {"model": model, "device": "auto"},
        timeout_s,
    )
