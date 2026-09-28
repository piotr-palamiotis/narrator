"""Generate deterministic Polish TTS test clips for stress-testing narrator.

Uses Piper (pl_PL-darkman-medium) to synthesize sentences with known
transcripts. The resulting scripts/test_clips/*.wav files plus manifest.json
are the fixtures consumed by scripts/stress_full_pipeline.py.

Determinism: piper 1.8's VITS generator has no seed, so nonzero noise scales
produce different audio every run. With --noise-scale 0 --noise-w-scale 0 the
output is byte-identical across runs (verified via md5sum), so regenerating
the fixtures never invalidates a recorded baseline.

Usage:
  scripts/gen_test_clips.py                 # write clips + manifest into scripts/test_clips/
  scripts/gen_test_clips.py --out DIR       # different output dir
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPER_PYTHON = sys.executable  # run piper from the same venv as everything else

VOICE = "pl_PL-darkman-medium"
VOICE_PATH = REPO_ROOT / f"{VOICE}.onnx"
VOICE_CONFIG = REPO_ROOT / f"{VOICE}.onnx.json"

# piper 1.8's VITS generator has no seed control, so *any* nonzero noise
# scale makes outputs non-deterministic across runs. noise_scale=0 /
# noise_w_scale=0 collapse the stochastic ops to their mean values, which
# yields byte-identical WAVs across runs (verified via md5) at the cost of a
# slightly more monotone voice -- the right trade for a test fixture.
NOISE_SCALE = "0"
NOISE_W_SCALE = "0"

# Fixed, hand-written Polish sentences with unambiguous punctuation and
# numerals spelled the way whisper is expected to transcribe them (whisper
# writes "2 000" for "dwa tysiące" in pl; verify against your model's habit
# and edit here if a systematic mismatch shows up in the WER report).
SENTENCES: list[dict] = [
    {
        "id": "intro",
        "text": "Witajcie na dzisiejszym spotkaniu.",
    },
    {
        # whisper (pl) writes numerals as digits without spaces, so the
        # reference must match that style or WER counts formatting as errors.
        # Synthesis text keeps "15000" (piper reads it "piętnaście tysięcy");
        # the scoring reference spells it "15 tysięcy" because whisper
        # orthographically renders that phrase as digits+words.
        "id": "numbers",
        "text": "W roku 2000 miasto liczyło 15000 mieszkańców.",
        "reference": "W roku 2000 miasto liczyło 15 tysięcy mieszkańców.",
    },
    {
        "id": "tech",
        "text": "Napisy na żywo powstają w oparciu o lokalny model rozpoznawania mowy.",
    },
    {
        "id": "punct",
        "text": "Czy to działa? Tak, działa bardzo dobrze!",
    },
    {
        "id": "long",
        "text": (
            "To zdanie jest znacznie dłuższe niż pozostałe, "
            "żeby sprawdzić, jak dekoder radzi sobie z dłuższą wypowiedzią "
            "bez przerwy pomiędzy słowami, gdy okno dekodera rośnie."
        ),
    },
    {
        "id": "mixed",
        "text": "System działa na procesorze, na karcie graficznej, albo przełącza się na procesor awaryjnie.",
    },
]

# Harness schedule: fixed playback order and inter-clip silences (seconds).
# Keep the gap comfortably above asr.end_silence_ms (650ms default) plus VAD
# decay and one finalize decode (~0.3-0.6s), so each clip is force-finalized
# BEFORE the next clip starts -- otherwise FINALs straddle window boundaries.
SCHEDULE_GAPS_S = 1.5
REPEATS_PER_SENTENCE = 3


def synth(sentence_text: str, out_path: Path) -> None:
    """One Piper invocation per sentence, deterministic settings."""
    subprocess.run(
        [
            PIPER_PYTHON, "-m", "piper",
            "-m", str(VOICE_PATH),
            "-c", str(VOICE_CONFIG),
            "--noise-scale", NOISE_SCALE,
            "--noise-w-scale", NOISE_W_SCALE,
            "--length-scale", "1.0",
            "--sentence-silence", "0.4",
            "-f", str(out_path),
        ],
        input=sentence_text,
        text=True,
        check=True,
        capture_output=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "scripts" / "test_clips")
    ap.add_argument("--schedule-only", action="store_true", help="print harness schedule JSON and exit")
    args = ap.parse_args()

    if args.schedule_only:
        print(json.dumps({
            "clips": [
                {"id": s["id"], "transcript": s["text"], "repeats": REPEATS_PER_SENTENCE,
                 "gap_after_s": SCHEDULE_GAPS_S}
                for s in SENTENCES
            ],
            "inter_clip_silence_s": SCHEDULE_GAPS_S,
        }, ensure_ascii=False, indent=2))
        return

    if not VOICE_PATH.exists():
        sys.exit(f"Piper voice not found at {VOICE_PATH}. Run: .venv/bin/python -m piper.download_voices {VOICE}")

    args.out.mkdir(parents=True, exist_ok=True)
    for s in SENTENCES:
        wav = args.out / f"{s['id']}.wav"
        synth(s["text"], wav)
        print(f"{s['id']}.wav  <-  {s['text'][:60]}")

    manifest = {
        "voice": VOICE,
        "piper_params": {"noise_scale": 0.0, "noise_w_scale": 0.0, "length_scale": 1.0, "sentence_silence": 0.4},
        "clips": [
            {"id": s["id"], "transcript": s["text"], "reference": s.get("reference", s["text"])}
            for s in SENTENCES
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmanifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
