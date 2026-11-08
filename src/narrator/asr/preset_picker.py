"""`narrator --pick-preset`: a short real benchmark that picks asr.preset.

Not a GPU-name lookup: every candidate model is run on this machine, on the
configured device, against a bundled deterministic Polish clip
(data/preset_clip_pl.wav, the Piper fixtures of scripts/test_clips):

  1. bench (narrator.asr.bench, its own subprocess): fixed windows, to
     screen out a model that cannot even decode in real time and to give
     the language-detection cost;
  2. probe (narrator.asr.preset_probe, its own subprocess): about a minute
     of real streaming decodes of the clip (stream_sim, measured timings),
     a DecodeCostModel fitted to them, then one deterministic stream_sim
     replay per preset that uses the model -- its worker load and caption
     delay p50 (FINAL emit time minus when the words were spoken);
  3. the most accurate tier (presets.yaml order) with load <= MAX_LOAD and
     delay p50 <= MAX_DELAY_S wins.

Tiers are tried most accurate first and the run stops at the first one that
fits (--all measures every tier). The configured asr.unit is used as is
-- no GPU->CPU downgrade: a model that fails on that unit is just not a
candidate. Only `asr.preset` is written (config.loader.write_asr_preset,
previous file -> config.yaml.bak). Models not available locally are
skipped unless `download=True`.

The clip is TTS speech with pauses between sentences, lighter than dense
live narration; MAX_LOAD (0.7, vs narrator-tune's 0.9) is the headroom for
that and for decode jitter.
"""
from __future__ import annotations

import difflib
import json
import math
import statistics
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

DATA_DIR = Path(__file__).with_name("data")
CLIP_WAV = DATA_DIR / "preset_clip_pl.wav"
CLIP_META = DATA_DIR / "preset_clip_pl.json"

MAX_LOAD = 0.7
MAX_DELAY_S = 3.0
# A model whose benched decode is slower than this multiple of real time
# at any window cannot carry MAX_LOAD; its probe is skipped.
BENCH_MAX_RATIO = 1.0
BENCH_WINDOWS_S = [2.0, 5.0, 10.0]
PROBE_TARGET_SECONDS = 60.0  # ~1 min of real decodes per model
BENCH_TIMEOUT_S = 900.0
PROBE_TIMEOUT_S = 1800.0


# ---------------------------------------------------------------------------
# The clip
# ---------------------------------------------------------------------------


def load_clip() -> tuple[np.ndarray, list[dict]]:
    """(float32 mono 16 kHz audio, segments [{start_s, end_s, reference}])."""
    with wave.open(str(CLIP_WAV), "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    meta = json.loads(CLIP_META.read_text(encoding="utf-8"))
    return audio, meta["segments"]


def reference_text(segments: list[dict]) -> str:
    return " ".join(seg["reference"] for seg in segments)


def reference_word_times(segments: list[dict]) -> list[tuple[str, float]]:
    """(normalized word, estimated time spoken), interpolated evenly over
    each segment (the clip has no word timings)."""
    from narrator.asr.metrics import normalize

    out: list[tuple[str, float]] = []
    for seg in segments:
        words = normalize(seg["reference"]).split()
        span = float(seg["end_s"]) - float(seg["start_s"])
        for i, word in enumerate(words):
            out.append((word, float(seg["start_s"]) + span * (i + 0.5) / len(words)))
    return out


def caption_delays(finals: list[tuple[float, str]], segments: list[dict]) -> list[float]:
    """Delay of every captioned word that matches a reference word: its
    FINAL's emit time minus when it was spoken (difflib word alignment, so
    a dropped or misheard word does not shift the rest)."""
    from narrator.asr.metrics import normalize

    reference = reference_word_times(segments)
    hypothesis = [(word, float(t)) for t, text in finals for word in normalize(text).split()]
    matcher = difflib.SequenceMatcher(
        None, [w for w, _ in reference], [w for w, _ in hypothesis], autojunk=False
    )
    delays: list[float] = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            delays.append(hypothesis[block.b + k][1] - reference[block.a + k][1])
    return delays


def delay_p50(finals: list[tuple[float, str]], segments: list[dict]) -> float | None:
    delays = caption_delays(finals, segments)
    return statistics.median(delays) if delays else None


# ---------------------------------------------------------------------------
# Decision (pure)
# ---------------------------------------------------------------------------


@dataclass
class TierResult:
    name: str
    model: str
    step_ms: int
    agreement_passes: int
    load: float | None = None
    delay_p50: float | None = None
    wer: float | None = None
    error: str | None = None  # why the tier could not be measured

    def fits(self, max_load: float = MAX_LOAD, max_delay_s: float = MAX_DELAY_S) -> bool:
        return (
            self.error is None
            and self.load is not None
            and self.delay_p50 is not None
            and self.load <= max_load
            and self.delay_p50 <= max_delay_s
        )


def choose_preset(
    rows: list[TierResult], max_load: float = MAX_LOAD, max_delay_s: float = MAX_DELAY_S
) -> TierResult | None:
    """The first (= most accurate) tier that fits, or None."""
    return next((row for row in rows if row.fits(max_load, max_delay_s)), None)


def slowest_decode_ratio(decode_seconds_by_window: dict[str, float]) -> float:
    ratios = [
        float(seconds) / float(label.rstrip("s"))
        for label, seconds in decode_seconds_by_window.items()
        if float(label.rstrip("s")) > 0
    ]
    return max(ratios) if ratios else math.inf


# ---------------------------------------------------------------------------
# Measurement (subprocesses)
# ---------------------------------------------------------------------------


def run_json_subprocess(module: str, payload: dict, timeout_s: float) -> dict:
    """Run `python -m module <payload>` and parse its `BENCH_JSON:` line.
    A fresh process per model: a native crash or VRAM held by one model
    cannot affect the next. Failures come back as {"error": ...}."""
    command = [sys.executable, "-m", module, json.dumps(payload)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"error": f"{module} timed out after {timeout_s:.0f}s"}
    except OSError as exc:
        return {"error": f"{module} failed to start: {exc}"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("BENCH_JSON:"):
            try:
                return json.loads(line[len("BENCH_JSON:") :])
            except json.JSONDecodeError:
                break
    tail = " | ".join(ln.strip() for ln in proc.stderr.splitlines() if ln.strip())[-400:]
    return {"error": f"{module} exited with {proc.returncode} without a result ({tail})"}


Runner = Callable[[str, dict, float], dict]


def measure_model(
    model: str,
    tiers: list,
    asr_settings: dict,
    *,
    runner: Runner = run_json_subprocess,
    bench_audio: Path | None = None,
    log: Callable[[str], None] = print,
) -> list[TierResult]:
    """Bench `model`, then probe every tier in `tiers` (presets using it).
    `asr_settings` is the operator's asr section (unit, unit identity,
    language whitelist and every non-preset setting are taken from it as
    they are -- the unit is never changed)."""
    rows = [TierResult(t.name, t.model, t.step_ms, t.agreement_passes) for t in tiers]
    unit = asr_settings["unit"]
    bench_payload = {
        "model": model,
        "unit": unit,
        "unit_identity": asr_settings.get("unit_identity", ""),
        "windows_s": BENCH_WINDOWS_S,
    }
    if bench_audio is not None:
        bench_payload["audio_npy"] = str(bench_audio)
    log(f"  {model}: bench on unit '{unit}'...")
    bench = runner("narrator.asr.bench", bench_payload, BENCH_TIMEOUT_S)
    if "error" in bench:
        for row in rows:
            row.error = f"bench failed: {bench['error']}"
        return rows
    timings = {str(k): float(v) for k, v in bench.get("decode_seconds_by_window", {}).items()}
    ratio = slowest_decode_ratio(timings)
    if ratio > BENCH_MAX_RATIO:
        for row in rows:
            row.error = f"too slow on this machine ({ratio:.2f}x real time)"
        return rows
    log(f"  {model}: {bench.get('resolved_device', '?')}, 10s window {timings.get('10.0s', float('nan')):.2f}s; "
        f"~{PROBE_TARGET_SECONDS:.0f}s of streaming decodes...")
    probe = runner(
        "narrator.asr.preset_probe",
        {
            "model": model,
            "asr": asr_settings,
            "tiers": [{"name": t.name, "step_ms": t.step_ms, "agreement_passes": t.agreement_passes} for t in tiers],
            "bench": bench,
            "target_seconds": PROBE_TARGET_SECONDS,
        },
        PROBE_TIMEOUT_S,
    )
    if "error" in probe:
        for row in rows:
            row.error = f"probe failed: {probe['error']}"
        return rows
    by_name = {t["name"]: t for t in probe.get("tiers", [])}
    for row in rows:
        result = by_name.get(row.name)
        if result is None:
            row.error = "probe returned no result"
            continue
        row.load = float(result["busy_fraction"])
        row.delay_p50 = None if result.get("delay_p50") is None else float(result["delay_p50"])
        row.wer = None if result.get("wer") is None else float(result["wer"])
    return rows


@dataclass
class PickOutcome:
    rows: list[TierResult] = field(default_factory=list)
    chosen: TierResult | None = None
    written: bool = False
    backup: Path | None = None
    overrides: dict = field(default_factory=dict)


def pick_preset(
    config_path: Path,
    *,
    dry_run: bool = False,
    download: bool = False,
    measure_all: bool = False,
    runner: Runner = run_json_subprocess,
    model_available: Callable[[str], bool] | None = None,
    log: Callable[[str], None] = print,
) -> PickOutcome:
    import yaml

    from narrator.config.loader import load_config, write_asr_preset
    from narrator.config.presets import explicit_overrides, load_presets

    if model_available is None:
        from narrator.paths import find_model_file

        def model_available(model: str) -> bool:
            return find_model_file(model) is not None

    config = load_config(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    asr_settings = config.asr.model_dump(mode="json")
    presets = load_presets()
    outcome = PickOutcome()

    log(f"Benchmarking presets on unit '{config.asr.unit}' (the configured unit; never changed)")
    measured: dict[str, list[TierResult]] = {}
    with tempfile.TemporaryDirectory(prefix="narrator-pick-") as tmp:
        bench_audio = Path(tmp) / "clip.npy"
        audio, _ = load_clip()
        np.save(bench_audio, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
        for preset in presets.values():
            if preset.model not in measured:
                if not download and not model_available(preset.model):
                    measured[preset.model] = [
                        TierResult(p.name, p.model, p.step_ms, p.agreement_passes,
                                   error="model not downloaded (pass --download to fetch it)")
                        for p in presets.values() if p.model == preset.model
                    ]
                else:
                    tiers = [p for p in presets.values() if p.model == preset.model]
                    measured[preset.model] = measure_model(
                        preset.model, tiers, asr_settings, runner=runner, bench_audio=bench_audio, log=log
                    )
            row = next(r for r in measured[preset.model] if r.name == preset.name)
            outcome.rows.append(row)
            if row.fits() and not measure_all:
                break

    outcome.chosen = choose_preset(outcome.rows)
    if outcome.chosen is not None:
        raw_asr = dict(raw.get("asr") or {})
        raw_asr["preset"] = outcome.chosen.name
        outcome.overrides = explicit_overrides(raw_asr)
        if not dry_run:
            outcome.backup = write_asr_preset(config_path, outcome.chosen.name)
            outcome.written = True
    return outcome


def format_rows(rows: list[TierResult]) -> list[str]:
    lines = [f"  {'preset':<10} {'model':<16} {'step':>5} {'passes':>6} {'load':>6} {'delay p50':>9} {'WER':>6}  result"]
    for row in rows:
        load = "-" if row.load is None else f"{100 * row.load:.0f}%"
        delay = "-" if row.delay_p50 is None else f"{row.delay_p50:.2f}s"
        wer = "-" if row.wer is None else f"{100 * row.wer:.0f}%"
        verdict = row.error or ("fits" if row.fits() else
                                f"over the limits (load <= {100 * MAX_LOAD:.0f}%, delay <= {MAX_DELAY_S:.0f}s)")
        lines.append(
            f"  {row.name:<10} {row.model:<16} {row.step_ms:>5} {row.agreement_passes:>6} "
            f"{load:>6} {delay:>9} {wer:>6}  {verdict}"
        )
    return lines


def main(config_path: Path, *, dry_run: bool, download: bool, measure_all: bool) -> int:
    """CLI body of `narrator --pick-preset`."""
    outcome = pick_preset(config_path, dry_run=dry_run, download=download, measure_all=measure_all)
    print()
    for line in format_rows(outcome.rows):
        print(line)
    print("  (load = ASR worker busy fraction; delay = word spoken -> caption, on the bundled clip; "
          "WER on that clip is informational only)")
    if outcome.chosen is None:
        print(f"\nNo preset fits this machine; {config_path} was not changed.")
        return 1
    print(f"\nChosen preset: {outcome.chosen.name}")
    if outcome.written:
        print(f"Written asr.preset: {outcome.chosen.name} to {config_path} (previous file: {outcome.backup})")
        print("A running narrator picks the change up by itself (config hot reload).")
    else:
        print(f"Dry run: {config_path} not changed.")
    if outcome.overrides:
        fields = ", ".join(f"{k}={v}" for k, v in outcome.overrides.items())
        print(f"Note: config.yaml sets {fields} explicitly; those override the preset "
              "(remove them to follow it).")
    return 0
