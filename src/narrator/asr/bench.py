"""Benchmark one ASR model on the local hardware and print JSON to stdout.

Run inside a *subprocess* by narrator.asr.autoconfigure so a native crash or
a multi-minute model load cannot take down the orchestrator (same isolation
reason the ASR worker is a process, not a thread). All stderr/log output is
discarded by the caller; only the final `BENCH_JSON:`-prefixed line is
consumed.

Design notes:
  * Inference params mirror backend.py's decode-time choices exactly
    (audio_ctx sized to the window, temperature_inc=0) so results transfer
    to the live pipeline.
  * Windows are cut from the stress fixtures (scripts/test_clips/*.wav) when
    present; otherwise a fixed deterministic pseudo-speech carrier is
    synthesized, so the bench is runnable on any machine. Whisper's output
    text on the synthetic carrier is irrelevant: the bench measures decode
    *time*, which is dominated by window length, not by the exact words.
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


def _bench_model(model: str, device: str, windows_s: list[float]) -> dict:
    from narrator.asr.backend import WhisperBackend, audio_ctx_for_seconds
    from narrator.config.schema import AsrConfig

    config = AsrConfig(model=model, device=device, language_whitelist=["pl"])
    backend = WhisperBackend(config)

    audio = _load_real_audio()
    needed = max(windows_s) + 1.0
    if audio is None or len(audio) / SAMPLE_RATE < needed:
        audio = _synthetic_audio(needed)

    results = {}
    for window_s in windows_s:
        samples = audio[: int(window_s * SAMPLE_RATE)]
        # warm-up rep: first Vulkan call pays one-time shader compilation
        backend.transcribe(samples)
        timings = []
        for _ in range(_TIMED_REPS):
            start = time.monotonic()
            backend.transcribe(samples)
            timings.append(time.monotonic() - start)
        results[f"{window_s:.1f}s"] = round(statistics.median(timings), 3)

    return {
        "model": model,
        "device": device,
        "resolved_device": backend.resolved_device_description,
        "decode_seconds_by_window": results,
    }


def main() -> int:
    try:
        payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
        model = payload.get("model", "medium")
        device = payload.get("device", "auto")
        windows = payload.get("windows_s", [2.0, 5.0, 10.0])
        result = _bench_model(model, str(device), [float(w) for w in windows])
        print("BENCH_JSON:" + json.dumps(result))
        return 0
    except Exception as exc:
        # Single JSON error line on stdout; human-readable detail on stderr.
        print("BENCH_JSON:" + json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        logger.exception("benchmark failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
