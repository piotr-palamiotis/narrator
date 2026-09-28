"""Quality probe: run the real streaming decoder over the stress fixtures,
score FINAL captions vs the known references, and print JSON to stdout.

Run inside a *subprocess* by narrator.asr.autoconfigure (same crash-isolation
rationale as bench.py; only the `BENCH_JSON:`-prefixed line is consumed).

This is the *accuracy* half of autoconfigure, and it deliberately replays the
STREAMING decode -- GrowingWindow + local agreement + VAD force-finalize at
the configured step_ms/agreement/end_silence -- not one-shot transcription.
The stress test showed why the distinction is decisive: large-v3 transcribes
short clips cleanly offline, but its streaming decode hits repetition loops
(growing windows re-decoded repeatedly), its steps slow to seconds, and live
accuracy collapses (~74% WER vs medium's ~8%). An offline probe would crown
exactly the wrong model; this one measures what live captioning experiences.

Frames are fed in real time (30ms per 30ms of wall clock) using the worker's
drain-then-tick shape, so decode steps contend with accumulating audio the
same way they do in the live pipeline. Cost: about the duration of the
fixtures (~30s) per candidate model.
"""
from __future__ import annotations

import json
import logging
import queue
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
REPO_ROOT = Path(__file__).resolve().parents[3]  # src/narrator/asr/quality_probe.py -> repo root
CLIPS_DIR = REPO_ROOT / "scripts" / "test_clips"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("quality_probe")


def _load_audio(wav: Path) -> np.ndarray:
    from narrator.asr.metrics import load_wav_16k_mono

    return load_wav_16k_mono(wav)


def _stream_transcribe(backend, config, audio: np.ndarray) -> str:
    """Decode one clip through the real StreamingDecoder at real-time pace.

    Mirrors worker.py's loop shape: drain all pending frames with
    run_decode_step=False, then tick() once, feeding one 30ms frame per 30ms
    of wall time so windows grow during slow decodes exactly as they do live.
    """
    from narrator.asr.events import CaptionEventType
    from narrator.asr.streaming_decoder import StreamingDecoder
    from narrator.asr.vad import SileroVad

    events: queue.Queue = queue.Queue()
    decoder = StreamingDecoder(backend, SileroVad(config.vad_threshold), config, events.put)

    frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)
    frames = [audio[i : i + frame_samples] for i in range(0, len(audio), frame_samples)]

    pending: queue.Queue = queue.Queue()
    next_frame_at = time.monotonic()
    for frame in frames:
        pending.put(frame)
        decoder.feed(pending.get_nowait(), run_decode_step=False)
        while not pending.empty():
            decoder.feed(pending.get_nowait(), run_decode_step=False)
        decoder.tick()
        next_frame_at += FRAME_MS / 1000.0
        delay = next_frame_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    # Tail silence to let end_silence force-finalize the last utterance.
    silence = np.zeros(frame_samples, dtype=np.float32)
    tail_deadline = time.monotonic() + (config.end_silence_ms / 1000.0) + 2.0
    while time.monotonic() < tail_deadline:
        pending.put(silence)
        decoder.feed(pending.get_nowait(), run_decode_step=False)
        while not pending.empty():
            decoder.feed(pending.get_nowait(), run_decode_step=False)
        decoder.tick()
        time.sleep(FRAME_MS / 1000.0)

    finals: list[str] = []
    while not events.empty():
        event = events.get_nowait()
        if event.type is CaptionEventType.FINAL:
            finals.append(event.text)
    return " ".join(finals).strip()


def _probe_model(model: str, device: str, language: str, clips_dirs: list[Path]) -> dict:
    from narrator.asr.backend import WhisperBackend
    from narrator.asr.metrics import has_word_doubling, wer
    from narrator.config.schema import AsrConfig

    # Probe with default decode params so every candidate is measured under
    # identical, config-shipped conditions (autoconfigure only overrides
    # model/device/step_ms afterward).
    config = AsrConfig(model=model, device=device, language=language)
    backend = WhisperBackend(config)

    wers: list[float] = []
    doubled_any = False
    texts: dict[str, str] = {}
    wer_by_clip: dict[str, float] = {}
    for clips_dir in clips_dirs:
        manifest_path = clips_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no fixture manifest at {manifest_path}; run scripts/gen_test_clips.py first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for clip in manifest["clips"]:
            wav = clips_dir / f"{clip['id']}.wav"
            reference = clip.get("reference", clip["transcript"])
            text = _stream_transcribe(backend, config, _load_audio(wav))
            key = clip["id"] if len(clips_dirs) == 1 else f"{clips_dir.name}/{clip['id']}"
            texts[key] = text
            score = wer(reference, text)
            wers.append(score)
            wer_by_clip[key] = round(score, 4)
            if has_word_doubling(text):
                doubled_any = True

    return {
        "model": model,
        "device": device,
        "language": language,
        "resolved_device": backend.resolved_device_description,
        "probe": "streaming",
        "wer": round(statistics.mean(wers), 4),
        "wer_by_clip": wer_by_clip,
        "doubled": doubled_any,
        "texts": texts,
    }


def main() -> int:
    try:
        payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
        model = str(payload.get("model", "medium"))
        device = str(payload.get("device", "auto"))
        language = str(payload.get("language", "pl"))
        clips_dirs = [Path(d) for d in payload.get("clips_dirs", [str(CLIPS_DIR)])]
        result = _probe_model(model, device, language, clips_dirs)
        print("BENCH_JSON:" + json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print("BENCH_JSON:" + json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        logger.exception("quality probe failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
