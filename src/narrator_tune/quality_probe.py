"""Quality probe: decode library excerpts the way the live captioner would
and append the raw captions to the result store.

Run inside a *subprocess* by narrator_tune.library_eval.Evaluator (same
crash/VRAM-isolation rationale as bench.py; only the `BENCH_JSON:`-prefixed
summary line is consumed). The job -- which excerpts, under which decode
configurations -- arrives as a JSON file. Every finished (excerpt, config)
result is appended to the store immediately, so a crash or timeout loses at
most the excerpt in flight and the parent resumes from there.

Modes:
  stream    the default: the real decoder (asr.decoder: StreamingDecoder or
            the segmenter) replayed on a virtual
            clock charged by the model's fitted decode-cost model
            (stream_sim) -- live behavior, deterministic, ~0.1-0.25x real
            time on the reference GPU.
  measured  the same replay, charging real decode wall time instead
            (fidelity check).
  realtime  30 ms frames fed at real-time pace into the worker's own
            drain-then-tick loop (fidelity check only: one hour per hour of
            audio -- the method that timed out every candidate before).
  offline   whole-excerpt transcription, no streaming: the best agreement
            whisper can reach with the reference (the "ceiling").

Why streaming replay decides, not offline transcription: large-v3
transcribes short clips cleanly offline, but before the sentence-loop
collapse fix its streaming decode hit repetition loops that collapsed live
accuracy (~74% WER vs medium's ~8%); an offline probe would have crowned
exactly the wrong model.
"""
from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("quality_probe")


class _AudioSource:
    """Excerpt audio for the job: cached .npy files, else decoded from the
    recording in one pipe pass per recording (the disk-full fallback)."""

    def __init__(self, tasks: list[dict]) -> None:
        self._audio: dict[str, np.ndarray] = {}
        self._uncached: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        seen: set[str] = set()
        for task in tasks:
            excerpt_id, audio = task["excerpt_id"], task["audio"]
            if excerpt_id in seen:
                continue
            seen.add(excerpt_id)
            if audio["cache"] is None and not audio["synthetic"]:
                self._uncached[audio["path"]].append((excerpt_id, audio["start_s"], audio["end_s"]))

    def get(self, task: dict) -> np.ndarray:
        excerpt_id, audio = task["excerpt_id"], task["audio"]
        if excerpt_id not in self._audio:
            if audio["cache"] is not None:
                self._audio[excerpt_id] = np.load(audio["cache"])
            elif audio["synthetic"]:
                from narrator.asr.metrics import load_wav_16k_mono

                return load_wav_16k_mono(Path(audio["path"]))
            else:
                from narrator_tune.real_fixtures import read_pcm_windows

                group = self._uncached.pop(audio["path"])
                pcm = read_pcm_windows(Path(audio["path"]), [(s, e) for _, s, e in group])
                for (other_id, _, _), samples in zip(group, pcm):
                    self._audio[other_id] = samples
        return self._audio[excerpt_id].astype(np.float32) / 32768.0


def _offline(backend, audio: np.ndarray, language: str):
    from narrator.asr.stream_sim import SimResult
    from narrator.asr.decoder_common import _collapse_repetitions

    started = time.monotonic()
    segments = backend.transcribe(audio, language=language)
    text = " ".join(_collapse_repetitions(" ".join(t for _, _, t in segments if t).split()))
    duration = len(audio) / SAMPLE_RATE
    return SimResult(
        finals=[(duration, text)] if text else [],
        audio_seconds=duration,
        decode_calls=1,
        detect_calls=0,
        busy_seconds=time.monotonic() - started,
    )


def _realtime(backend, config, audio: np.ndarray):
    """The live loop on the wall clock: a producer thread delivers 30 ms
    frames at real-time pace, the consumer is worker.py's drain-then-tick."""
    from narrator.asr.decoder_common import create_decoder
    from narrator.asr.events import CaptionEventType
    from narrator.asr.stream_sim import SimResult
    from narrator.asr.vad import SileroVad

    frames: queue.Queue = queue.Queue()
    finals: list[tuple[float, str]] = []
    started = time.monotonic()

    def on_event(event) -> None:
        if event.type is CaptionEventType.FINAL and event.text.strip():
            finals.append((time.monotonic() - started, event.text.strip()))

    decoder = create_decoder(backend, SileroVad(threshold=config.vad_threshold), config, on_event)
    frame_samples = SAMPLE_RATE * FRAME_MS // 1000
    tail = [np.zeros(frame_samples, dtype=np.float32)] * (config.end_silence_ms // FRAME_MS + 4)

    def produce() -> None:
        chunks = [audio[i : i + frame_samples] for i in range(0, len(audio), frame_samples)] + tail
        for index, chunk in enumerate(chunks, start=1):
            delay = started + index * FRAME_MS / 1000.0 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            frames.put(chunk)
        frames.put(None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    done = False
    while not done:
        frame = frames.get()
        if frame is None:
            break
        decoder.feed(frame, run_decode_step=False)
        while True:
            try:
                frame = frames.get_nowait()
            except queue.Empty:
                break
            if frame is None:
                done = True
                break
            decoder.feed(frame, run_decode_step=False)
        decoder.tick()
    producer.join()
    flush = getattr(decoder, "flush", None)
    if flush is not None:
        flush()
    return SimResult(
        finals=finals,
        audio_seconds=len(audio) / SAMPLE_RATE,
        decode_calls=0,
        detect_calls=0,
        busy_seconds=0.0,
        rollovers=int(getattr(decoder, "rollover_count", 0)),
    )


def run_job(job: dict) -> dict:
    from narrator.asr.backend import WhisperBackend
    from narrator.asr.units import unit_or_legacy
    from narrator_tune.library_eval import ResultStore, now_iso
    from narrator.asr.stream_sim import DecodeCostModel, simulate_stream
    from narrator.config.schema import AsrConfig

    mode = job["mode"]
    tasks = job["tasks"]
    store = ResultStore(Path(job["store_path"]))
    cost = DecodeCostModel.from_dict(job["cost_model"]) if job.get("cost_model") else None
    audio_source = _AudioSource(tasks)
    # Virtual time is what matters in replay; real-time overrun warnings
    # from the decoder would only be noise here.
    logging.getLogger("narrator.asr.streaming_decoder").setLevel(logging.ERROR)
    logging.getLogger("narrator.asr.segmenter").setLevel(logging.ERROR)

    # One backend per (language whitelist, flash_attn, unit identity): detection
    # reads the whitelist from the backend's own config, flash_attn is a
    # whisper.cpp context parameter fixed at model load, and the unit's
    # identity is checked at load (a
    # task's flash_attn used to be silently ignored here). In practice every
    # task of a job shares one.
    by_whitelist: dict[tuple, list[dict]] = defaultdict(list)
    for task in tasks:
        asr = task["asr"]
        group_key = (
            tuple(asr.get("language_whitelist", [])),
            bool(asr.get("flash_attn", False)),
            asr.get("unit_identity", ""),
        )
        by_whitelist[group_key].append(task)

    total_audio = total_wall = 0.0
    resolved = ""
    done = 0
    # job["device"] is the asr.unit (an older job may carry an asr.device value).
    unit = unit_or_legacy(job["device"])
    for (whitelist, flash_attn, unit_identity), group in by_whitelist.items():
        print(f"[probe] loading model '{job['model']}' on unit '{unit}'...", file=sys.stderr, flush=True)
        backend = WhisperBackend(
            AsrConfig(
                model=job["model"],
                unit=unit,
                unit_identity=unit_identity,
                language_whitelist=list(whitelist),
                flash_attn=flash_attn,
            )
        )
        resolved = backend.resolved_device_description
        print(f"[probe] model loaded: {resolved}; {len(group)} decode(s) in {mode} mode", file=sys.stderr, flush=True)
        offline_language = job.get("language") or (whitelist[0] if whitelist else "pl")
        try:
            if mode in ("measured", "realtime"):
                # Timed modes: the first decodes of a fresh process pay
                # one-time GPU setup, which must not count as decode time.
                warm = audio_source.get(group[0])
                for seconds in (2.0, 10.0, 20.0):
                    backend.transcribe(warm[: int(seconds * SAMPLE_RATE)], language=offline_language)
            for task in group:
                config = AsrConfig.model_validate(task["asr"])
                audio = audio_source.get(task)
                started = time.monotonic()
                if mode == "offline":
                    result = _offline(backend, audio, offline_language)
                elif mode == "realtime":
                    result = _realtime(backend, config, audio)
                else:
                    result = simulate_stream(
                        backend, config, audio, cost=cost, measured=(mode == "measured")
                    )
                wall = time.monotonic() - started
                store.append(
                    {
                        "key": task["key"],
                        "excerpt_id": task["excerpt_id"],
                        "mode": mode,
                        "config_signature": task["config_signature"],
                        "resolved_device": resolved,
                        "wall_seconds": round(wall, 3),
                        "created": now_iso(),
                        **job.get("meta", {}),
                        "result": result.to_dict(),
                    }
                )
                done += 1
                total_audio += result.audio_seconds
                total_wall += wall
                print(
                    f"[probe] {done}/{len(tasks)} {task['excerpt_id'][:60]}: "
                    f"{len(result.finals)} captions, {result.audio_seconds:.0f}s audio in {wall:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            backend.close()
    return {
        "done": done,
        "audio_seconds": round(total_audio, 3),
        "wall_seconds": round(total_wall, 3),
        "resolved_device": resolved,
    }


def main() -> int:
    try:
        payload = json.loads(sys.argv[1] if len(sys.argv) > 1 else "{}")
        job = json.loads(Path(payload["job_file"]).read_text(encoding="utf-8"))
        result = run_job(job)
        print("BENCH_JSON:" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"[probe] failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("BENCH_JSON:" + json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        logger.exception("quality probe failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
