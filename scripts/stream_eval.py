"""Streaming-pipeline ASR accuracy on a real recording + SRT reference.

Runs the app's REAL captioning algorithm -- `StreamingDecoder` (local
agreement, VAD gating, repetition collapse, per-utterance language lock)
over `WhisperBackend` -- on a slice of a recording, replayed on the virtual
clock of `narrator.asr.stream_sim`: decode sequencing matches the live
worker (drain-then-tick, a decode step occupies the worker for as long as it
really takes) without waiting in real time. Scores the FINAL caption text
against the SRT reference with `eval_transcript`'s normalization (rollup
collapse, non-speech + foreign-window handling).

This measures the pipeline that would actually caption a live event, as
opposed to `eval_transcript.py`, which measures best-effort offline
transcription of the same audio. `narrator --autoconfigure` runs the same
replay over many excerpts of the whole library.

Usage:
  python scripts/stream_eval.py <audio> <reference.srt> [--start S] [--end S]
      [--model large-v3] [--language pl] [--step-ms 500] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "scripts")

from eval_transcript import (  # noqa: E402
    _words,
    collapse_reference_cues,
    load_audio_16k_mono,
    parse_srt,
)

from narrator.asr.backend import WhisperBackend  # noqa: E402
from narrator.asr.stream_sim import simulate_stream  # noqa: E402
from narrator.config.schema import AsrConfig  # noqa: E402


def run_stream(audio, config: AsrConfig) -> list[tuple[float, str]]:
    """FINAL captions as (emit time in audio seconds, text), charging each
    decode its real wall time (`measured`: no cost model needed)."""
    backend = WhisperBackend(config)
    try:
        return simulate_stream(backend, config, audio, measured=True).finals
    finally:
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("reference_srt")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=1e9)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="pl")
    parser.add_argument("--step-ms", type=int, default=500)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    import jiwer

    from narrator.asr.streaming_decoder import _collapse_repetitions

    cues = parse_srt(args.reference_srt)
    window = [c for c in cues if c.start < args.end and c.end > args.start]
    ref, non_speech, excluded, _ = collapse_reference_cues(window)
    ref = _collapse_repetitions(ref)
    if not ref:
        print("no in-scope reference text in window", file=sys.stderr)
        return 2
    print(f"reference: {len(ref)} words after rollup collapse "
          f"({len(non_speech)} non-speech, {len(excluded)} foreign windows)", flush=True)

    audio, rate = load_audio_16k_mono(args.audio)
    lo, hi = int(args.start * rate), int(min(args.end, len(audio) / rate) * rate)
    audio = audio[lo:hi]
    print(f"streaming {len(audio)/rate:.1f}s of audio through StreamingDecoder "
          f"({args.model}, step {args.step_ms}ms) ...", flush=True)

    config = AsrConfig(
        model=args.model,
        language_whitelist=[args.language],
        step_ms=args.step_ms,
    )
    finals = run_stream(audio, config)
    hyp_words = [w for _, text in finals for w in _words(text)]
    print(f"{len(finals)} FINAL captions, {len(hyp_words)} words")

    wer = jiwer.wer(" ".join(ref), " ".join(hyp_words)) if hyp_words else 1.0
    print(f"\n=== streaming WER ({args.start:.0f}-{args.end:.0f}s) ===")
    print(f"WER: {wer*100:.1f}%   (ref {len(ref)}, hyp {len(hyp_words)})")
    print("\nfirst captions:")
    for t, text in finals[:8]:
        print(f"  [{t:6.1f}] {text}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"streaming_wer": wer, "finals": finals}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
