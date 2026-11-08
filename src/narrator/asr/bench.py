"""Benchmark one ASR model on the local hardware and print JSON to stdout.

Run inside a *subprocess* by narrator_tune.autoconfigure so a native crash or
a multi-minute model load cannot take down the orchestrator (same isolation
reason the ASR worker is a process, not a thread). All stderr/log output is
discarded by the caller; only the final `BENCH_JSON:`-prefixed line is
consumed.

Design notes:
  * Inference params mirror backend.py's decode-time choices exactly
    (audio_ctx sized to the window, temperature_inc=0) so results transfer
    to the live pipeline.
  * Windows are cut from a real library excerpt when the caller passes one
    (`audio_npy`, an int16 .npy from the narrator-tune cache), else from the
    stress fixtures (scripts/test_clips/*.wav) when present; otherwise a
    fixed deterministic pseudo-speech carrier is synthesized, so the bench
    is runnable on any machine. Whisper's output text is irrelevant: the
    bench measures decode *time*, dominated by window length.
  * Language detection is timed too (`detect_seconds`): with a multi-entry
    whitelist the decoder runs it once per utterance, and the streaming
    replay (stream_sim) charges it to the worker like a decode.
  * For each window size we report the median of several reps (first rep
    discarded as warm-up), because a single-shot number on Vulkan includes
    one-time shader-compilation jitter.
  * Each candidate model is measured in a fresh bench.py subprocess, so VRAM
    from a previous candidate cannot poison the next.
"""
from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Match the stress harness / pipeline sample geometry.
SAMPLE_RATE = 16000
_WARMUP_REPS = 1
_TIMED_REPS = 3
REPO_ROOT = Path(__file__).resolve().parents[3]  # src/narrator/asr/bench.py -> repo root
CLIPS_DIR = REPO_ROOT / "scripts" / "test_clips"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bench")


def _load_real_audio() -> np.ndarray | None:
    """Concatenate the stress fixtures into one float32 array at 16kHz, or None."""
    try:
        from narrator.asr.metrics import load_wav_16k_mono

        chunks = [
            load_wav_16k_mono(wav) for wav in sorted(CLIPS_DIR.glob("*.wav"))
        ]
        chunks = [c for c in chunks if len(c) > 0]
        if not chunks:
            return None
        return np.concatenate(chunks)
    except Exception:
        logger.debug("no usable fixtures at %s; using synthetic carrier", CLIPS_DIR)
        return None


def _synthetic_audio(seconds: float) -> np.ndarray:
    """Deterministic pseudo-speech carrier (harmonics + amplitude envelope)."""
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    audio = envelope * (
        0.4 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 330.0 * t)
        + 0.2 * np.sin(2 * np.pi * 550.0 * t)
    )
    return audio.astype(np.float32)


def _load_npy_audio(path: str) -> np.ndarray | None:
    try:
        return np.load(path).astype(np.float32) / 32768.0
    except Exception:
        logger.debug("could not load bench audio from %s", path)
        return None


def _bench_model(
    model: str, unit: str, windows_s: list[float], audio_npy: str | None = None, unit_identity: str = ""
) -> dict:
    from narrator.asr.asr_backend import create_backend
    from narrator.config.schema import AsrConfig

    config = AsrConfig(model=model, unit=unit, unit_identity=unit_identity, language_whitelist=["pl"])
    print(f"[bench] loading model '{model}' on unit '{unit}'...", file=sys.stderr, flush=True)
    backend = create_backend(config)
    print(f"[bench] model loaded: {backend.resolved_device_description}", file=sys.stderr, flush=True)

    audio = _load_npy_audio(audio_npy) if audio_npy else None
    if audio is None:
        audio = _load_real_audio()
    needed = max(windows_s) + 1.0
    if audio is None or len(audio) / SAMPLE_RATE < needed:
        audio = _synthetic_audio(needed)
        print(f"[bench] using synthetic audio ({needed:.1f}s)", file=sys.stderr, flush=True)
    else:
        duration = len(audio) / SAMPLE_RATE
        print(f"[bench] loaded {duration:.1f}s of real fixtures", file=sys.stderr, flush=True)

    results = {}
    for window_s in windows_s:
        samples = audio[: int(window_s * SAMPLE_RATE)]
        print(f"[bench] warmup + timing {window_s:.1f}s window...", file=sys.stderr, flush=True)
        # warm-up rep: first Vulkan call pays one-time shader compilation
        backend.transcribe(samples)
        timings = []
        for rep in range(_TIMED_REPS):
            start = time.monotonic()
            backend.transcribe(samples)
            elapsed = time.monotonic() - start
            timings.append(elapsed)
            print(f"[bench]   {window_s:.1f}s rep {rep + 1}/{_TIMED_REPS}: {elapsed:.3f}s", file=sys.stderr, flush=True)
        results[f"{window_s:.1f}s"] = round(statistics.median(timings), 3)
        print(f"[bench]   {window_s:.1f}s median: {results[f'{window_s:.1f}s']:.3f}s", file=sys.stderr, flush=True)

    # Language detection: a fixed-size slice regardless of window length
    # (backend._LANGUAGE_DETECT_MAX_SECONDS), so one size suffices.
    detect_audio = audio[: int(5.0 * SAMPLE_RATE)]
    backend.detect_language(detect_audio)  # warm-up
    detect_timings = []
    for _ in range(_TIMED_REPS):
        start = time.monotonic()
        backend.detect_language(detect_audio)
        detect_timings.append(time.monotonic() - start)
    detect_seconds = round(statistics.median(detect_timings), 3)
    print(f"[bench]   language detection median: {detect_seconds:.3f}s", file=sys.stderr, flush=True)
    backend.close()

    return {
        "model": model,
        "unit": unit,
        # The unit's kind, under the name narrator-tune's cached benches use.
        "device": config.unit_kind,
        "resolved_device": backend.resolved_device_description,
        "decode_seconds_by_window": results,
        "detect_seconds": detect_seconds,
    }


def main() -> int:
    try:
        payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
        model = payload.get("model", "medium")
        unit = payload.get("unit")
        if unit is None:  # an older caller: asr.device + asr.gpu_device
            from narrator.asr.units import unit_from_legacy

            unit = unit_from_legacy(payload.get("device", "auto"), payload.get("gpu_device", "auto"))
        windows = payload.get("windows_s", [2.0, 5.0, 10.0, 20.0])
        result = _bench_model(
            model,
            str(unit),
            [float(w) for w in windows],
            payload.get("audio_npy"),
            str(payload.get("unit_identity", "")),
        )
        print("BENCH_JSON:" + json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        # Single JSON error line on stdout; human-readable detail on stderr.
        print(f"[bench] failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("BENCH_JSON:" + json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        logger.exception("benchmark failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
