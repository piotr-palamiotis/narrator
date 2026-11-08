"""Subprocess half of `narrator --pick-preset` (see preset_picker).

    python -m narrator.asr.preset_probe '<json payload>'

Loads one model on the configured device, replays the bundled clip through
the real StreamingDecoder with measured timings (stream_sim, measured=True)
for about `target_seconds` of decoding, fits a DecodeCostModel to those
calls, then replays the clip once per tier on that cost model and reports
each tier's worker load and caption delay p50 as a `BENCH_JSON:` line.
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_CALIBRATION_PASSES = 3
MIN_FIT_CALLS = 3


def _tier_config(asr: dict, model: str, tier: dict):
    from narrator.config.schema import AsrConfig

    return AsrConfig.model_validate(
        {
            **asr,
            "preset": "custom",
            "model": model,
            "step_ms": int(tier["step_ms"]),
            "agreement_passes": int(tier["agreement_passes"]),
        }
    )


def run_probe(payload: dict) -> dict:
    from narrator.asr.asr_backend import create_backend
    from narrator.asr.metrics import wer
    from narrator.asr.preset_picker import delay_p50, load_clip, reference_text
    from narrator.asr.stream_sim import DecodeCostModel, simulate_stream

    model = payload["model"]
    asr = payload["asr"]
    tiers = payload["tiers"]
    target_seconds = float(payload.get("target_seconds", 60.0))
    audio, segments = load_clip()
    reference = reference_text(segments)

    first = _tier_config(asr, model, tiers[0])
    print(f"[probe] loading {model} on {first.unit}...", file=sys.stderr, flush=True)
    backend = create_backend(first)  # DevicePolicyError if the unit fails: no fallback
    try:
        backend.transcribe(audio[: 2 * 16000])  # warm-up: one-time shader compilation
        calls: list[tuple[float, int, float]] = []
        detects: list[float] = []
        started = time.monotonic()
        for _ in range(MAX_CALIBRATION_PASSES):
            result = simulate_stream(backend, first, audio, measured=True)
            calls.extend(result.calls)
            detects.extend(result.detects)
            if time.monotonic() - started >= target_seconds:
                break
        bench = payload.get("bench") or {}
        detect_s = statistics.median(detects) if detects else float(bench.get("detect_seconds", 0.0))
        if len(calls) >= MIN_FIT_CALLS:
            cost = DecodeCostModel.fit_calls(calls, detect_s)
        else:
            cost = DecodeCostModel.fit(bench["decode_seconds_by_window"], detect_s)
        print(f"[probe] {len(calls)} decodes in {time.monotonic() - started:.0f}s; cost {cost.to_dict()}",
              file=sys.stderr, flush=True)

        out = []
        for tier in tiers:
            config = _tier_config(asr, model, tier)
            sim = simulate_stream(backend, config, audio, cost)
            out.append(
                {
                    "name": tier["name"],
                    "busy_fraction": round(sim.busy_fraction, 4),
                    "delay_p50": None if (d := delay_p50(sim.finals, segments)) is None else round(d, 3),
                    "wer": round(wer(reference, sim.text), 4) if sim.finals else 1.0,
                    "decode_calls": sim.decode_calls,
                }
            )
        return {
            "model": model,
            "resolved_device": backend.resolved_device_description,
            "cost": cost.to_dict(),
            "calibration_calls": len(calls),
            "tiers": out,
        }
    finally:
        backend.close()


def main() -> int:
    try:
        result = run_probe(json.loads(sys.argv[1]))
        print("BENCH_JSON:" + json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        print(f"[probe] failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("BENCH_JSON:" + json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
