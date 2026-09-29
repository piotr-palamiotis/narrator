"""Benchmark WhisperBackend decode latency (GPU vs CPU) on the window sizes the
streaming decoder actually produces. Uses /tmp/test16k.wav (16kHz mono s16le),
decoded from test.mp3 via gst-launch (see CLAUDE.md).

Usage: .venv/bin/python scripts/bench_decode.py [model] [--cpu]
"""
from __future__ import annotations

import sys
import time
import wave

import numpy as np

sys.path.insert(0, "src")
from narrator.asr.backend import WhisperBackend, audio_ctx_for_seconds  # noqa: E402
from narrator.config.schema import AsrConfig  # noqa: E402

WINDOW_SECONDS = (2.0, 5.0, 10.0, 20.0)


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1, "expected 16kHz mono"
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = args[0] if args else "large-v3"
    use_cpu = "--cpu" in sys.argv

    audio = load_wav("/tmp/test16k.wav")
    print(f"clip: {len(audio)/16000:.1f}s @16kHz mono")

    device = "cpu" if use_cpu else "vulkan"
    backend = WhisperBackend(
        AsrConfig(model=model, device=device, language_whitelist=["pl"])  # type: ignore[arg-type]
    )
    print(f"backend: {backend.resolved_device_description} (model={model})")

    # warm-up: first call pays one-time kernel/shader compilation on Vulkan
    backend.transcribe(audio[: int(2.0 * 16000)])
    print(f"{'window':>8} {'audio_ctx':>10} {'decode_s':>10}")
    for seconds in WINDOW_SECONDS:
        samples = audio[: int(seconds * 16000)]
        start = time.monotonic()
        segments = backend.transcribe(samples)
        elapsed = time.monotonic() - start
        text = " ".join(t for _, _, t in segments if t)
        preview = (text[:60] + "…") if len(text) > 60 else text
        print(f"{seconds:>7.1f}s {audio_ctx_for_seconds(seconds):>10} {elapsed:>9.2f}s  | {preview}")


if __name__ == "__main__":
    main()
