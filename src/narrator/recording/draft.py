"""draft.srt: an offline best-effort transcription of a finished recording.

Runs in its own process (`narrator --draft-recording <session dir>`,
started by RecordingManager at low priority) so a native whisper crash or a
multi-GB model never touches the live app. The style is
scripts/eval_transcript.py's: bounded <= 60 s spans with a fresh context
each (one poisoned stretch cannot ruin the rest), here cut at the silences
Silero VAD finds, with silence-only stretches skipped (whisper invents text
over silence). Decoding uses whisper's own defaults -- full 30 s encoder
context and the temperature-fallback ladder the live path disables for
latency -- since accuracy is all that matters offline.

The draft is deliberately NOT blocklist- or text_filter-filtered: it is the
raw starting point for a human correction, and the reference library needs
what was actually said. Only whisper's repetition loops are collapsed.

Parameters come from <session>/draft_request.json, written by the manager
(model, device, languages, vocabulary), so the subprocess needs no config.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from narrator.asr.offline import SAMPLE_RATE, planned_speech_spans, segments_to_cues, vad_probabilities
from narrator.recording.session import AUDIO_FILE, DRAFT_REQUEST, DRAFT_SRT, write_json_atomic
from narrator.recording.srt import format_srt

logger = logging.getLogger(__name__)

EXIT_MODEL_MISSING = 3


class DraftError(RuntimeError):
    pass


def build_request(config, language_whitelist: list[str], vocabulary: list[str]) -> dict:
    """The draft parameters for one recording (config: RootConfig)."""
    rec = config.recording
    return {
        "model": rec.draft_model,
        "unit": draft_unit(rec.draft_device, config.asr),
        "unit_identity": config.asr.unit_identity,
        "flash_attn": config.asr.flash_attn,
        "language_whitelist": list(language_whitelist),
        "vocabulary": list(vocabulary),
        "vad_threshold": config.asr.vad_threshold,
    }


def draft_unit(draft_device, asr) -> str:
    """The processing unit for the draft: recording.draft_device cpu -> cpu;
    otherwise the live ASR's GPU unit when it has one (the same card, with
    the same identity check), else auto."""
    if getattr(draft_device, "value", draft_device) == "cpu":
        return "cpu"
    return asr.unit if asr.unit_kind in ("vulkan", "cuda") else "auto"


def write_request(session_dir: Path, request: dict) -> None:
    write_json_atomic(session_dir / DRAFT_REQUEST, request)


def load_audio(path: Path) -> np.ndarray:
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _default_backend(request: dict):
    from narrator.asr.backend import WhisperBackend
    from narrator.config.schema import AsrConfig
    from narrator.paths import find_model_file

    if find_model_file(request["model"]) is None:
        # A background job must not start a multi-GB download on its own.
        raise DraftError(
            f"draft model {request['model']!r} is not available locally; download it first "
            "(e.g. use it once as asr.model) or set recording.draft_model to a model you have"
        )
    # A request written before asr.unit existed has device/gpu_device:
    # AsrConfig migrates those.
    legacy = {k: request[k] for k in ("device", "gpu_device") if k in request and "unit" not in request}
    config = AsrConfig(
        **({"unit": request["unit"]} if "unit" in request else legacy),
        unit_identity=request.get("unit_identity", "") if "unit" in request and request["unit"] != "auto" else "",
        model=request["model"],
        flash_attn=bool(request.get("flash_attn", False)),
        language_whitelist=request.get("language_whitelist") or [],
        strip_bracketed_annotations=False,
    )
    return WhisperBackend(config)


def _default_vad(request: dict):
    from narrator.asr.vad import SileroVad

    return SileroVad(threshold=float(request.get("vad_threshold", 0.5)))


def _transcribe(backend, chunk: np.ndarray, language: str, prompt: str):
    # Straight to the pywhispercpp model with whisper's offline defaults;
    # WhisperBackend.transcribe is tuned for live latency (shrunken encoder
    # context, no temperature fallback, a token cap). backend.py is left
    # untouched on purpose: narrator-tune keys its cached decodes on its source.
    return backend._model.transcribe(
        chunk,
        language=language,
        initial_prompt=prompt,
        single_segment=False,
        audio_ctx=0,
        temperature_inc=0.2,
        max_tokens=0,
    )


def generate_draft(
    session_dir: Path,
    request: dict,
    *,
    backend_factory: Callable[[dict], object] = _default_backend,
    vad_factory: Callable[[dict], object] = _default_vad,
    transcribe: Callable = _transcribe,
    log: Callable[[str], None] = print,
) -> int:
    """Decode <session_dir>/audio.flac and write draft.srt; returns the cue
    count."""
    from narrator.asr.vocabulary import build_initial_prompt

    audio = load_audio(session_dir / AUDIO_FILE)
    duration = len(audio) / SAMPLE_RATE
    probs, window_s = vad_probabilities(audio, vad_factory(request))
    spans = planned_speech_spans(probs, window_s, duration, threshold=float(request.get("vad_threshold", 0.5)))
    log(f"draft: {duration / 60:.1f} min of audio, {len(spans)} speech span(s), model {request['model']}")
    whitelist = list(request.get("language_whitelist") or [])
    prompt = build_initial_prompt(list(request.get("vocabulary") or []), None)
    segments: list[tuple[float, float, str]] = []
    backend = backend_factory(request) if spans else None
    started = time.monotonic()
    for number, (span_start, span_end) in enumerate(spans, start=1):
        chunk = audio[int(span_start * SAMPLE_RATE) : int(span_end * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE // 10:
            continue
        if len(whitelist) == 1:
            language = whitelist[0]
        else:
            detected = backend.detect_language(chunk)
            language = detected[0] if detected else (whitelist[0] if whitelist else "auto")
        for seg in transcribe(backend, chunk, language, prompt):
            segments.append((seg.t0 / 100.0 + span_start, seg.t1 / 100.0 + span_start, seg.text))
        log(f"draft: span {number}/{len(spans)} ({span_start:.0f}-{span_end:.0f} s, {language}) done "
            f"after {time.monotonic() - started:.0f} s")
    cues = segments_to_cues(segments)
    tmp = session_dir / (DRAFT_SRT + ".tmp")
    tmp.write_text(format_srt(cues), encoding="utf-8")
    tmp.replace(session_dir / DRAFT_SRT)
    log(f"draft: wrote {len(cues)} cue(s) to {session_dir / DRAFT_SRT}")
    return len(cues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="narrator --draft-recording")
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    try:
        os.nice(10)  # background work: the live app keeps the CPU
    except (AttributeError, OSError):
        pass
    try:
        request = json.loads((args.session_dir / DRAFT_REQUEST).read_text(encoding="utf-8"))
        generate_draft(args.session_dir, request, log=lambda line: print(line, flush=True))
    except DraftError as exc:
        print(f"draft failed: {exc}", file=sys.stderr, flush=True)
        return EXIT_MODEL_MISSING
    except Exception as exc:
        logger.exception("draft failed")
        print(f"draft failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
