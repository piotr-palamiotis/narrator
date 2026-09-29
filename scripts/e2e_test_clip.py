"""End-to-end test of the ASR pipeline against test.mp3, without a microphone.

Decodes test.mp3 to 16kHz mono WAV (gst-launch, see CLAUDE.md), then drives
the *real* StreamingDecoder (Silero VAD + WhisperBackend on GPU) with 30ms
frames through the same drain-then-tick shape the ASR worker process uses
(worker.py). Prints the FINAL caption events and wall-clock timing.

Usage: .venv/bin/python scripts/e2e_test_clip.py [wav_path]
"""
from __future__ import annotations

import logging
import queue
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from narrator.asr.backend import WhisperBackend  # noqa: E402
from narrator.asr.events import CaptionEvent, CaptionEventType  # noqa: E402
from narrator.asr.streaming_decoder import StreamingDecoder  # noqa: E402
from narrator.asr.vad import SileroVad  # noqa: E402
from narrator.config.schema import AsrConfig  # noqa: E402

FRAME_MS = 30
SAMPLE_RATE = 16000


def decode_mp3(mp3: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    subprocess.run(
        [
            "gst-launch-1.0", "-q",
            "filesrc", f"location={mp3}",
            "!", "decodebin", "!", "audioconvert", "!", "audioresample",
            "!", "audio/x-raw,format=S16LE,channels=1,rate=16000",
            "!", "wavenc", "!", "filesink", f"location={wav_path}",
        ],
        check=True,
        capture_output=True,
    )
    with wave.open(wav_path, "rb") as wav:
        raw = wav.readframes(wav.getnframes())
    Path(wav_path).unlink(missing_ok=True)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    mp3 = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "test.mp3")
    step_ms = 500
    if "--step-ms" in sys.argv:
        step_ms = int(sys.argv[sys.argv.index("--step-ms") + 1])
    audio = decode_mp3(mp3)
    duration = len(audio) / SAMPLE_RATE
    print(f"clip: {mp3} -> {duration:.1f}s @16kHz mono")

    events: queue.Queue[CaptionEvent] = queue.Queue()
    config = AsrConfig(model="large-v3", device="vulkan", language_whitelist=["pl"], step_ms=step_ms)
    backend = WhisperBackend(config)
    print(f"backend: {backend.resolved_device_description}")

    decoder = StreamingDecoder(backend, SileroVad(config.vad_threshold), config, events.put)

    # Same shape as worker.py: drain every pending frame with
    # run_decode_step=False, then tick() exactly once.
    frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)
    frames = [audio[i : i + frame_samples] for i in range(0, len(audio), frame_samples)]

    started = time.monotonic()
    pending: queue.Queue[np.ndarray] = queue.Queue()
    for frame in frames:
        pending.put(frame)
        decoder.feed(pending.get_nowait(), run_decode_step=False)
        while not pending.empty():
            decoder.feed(pending.get_nowait(), run_decode_step=False)
        decoder.tick()
    elapsed = time.monotonic() - started

    finals: list[str] = []
    status = ""
    while not events.empty():
        event = events.get_nowait()
        if event.type is CaptionEventType.FINAL:
            finals.append(event.text)
        elif event.type is CaptionEventType.STATUS:
            status = event.text

    print(f"\nstatus: {status}")
    print(f"wall clock: {elapsed:.1f}s for {duration:.1f}s of audio (real-time factor {elapsed/duration:.2f}x)")
    print(f"final caption events: {len(finals)}")
    print("--- transcript (as committed by local agreement) ---")
    print(" ".join(finals))


if __name__ == "__main__":
    main()
