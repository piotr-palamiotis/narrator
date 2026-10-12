"""Full-pipeline stress test: Piper TTS clips -> PipeWire virtual mic -> real
narrator app (AppController: capture, ASR worker process, renderer) -> recorded
FINAL caption events -> WER + latency + determinism report.

This exercises the *live microphone* path end-to-end, at real-time pace, with
deterministic input audio (see scripts/gen_test_clips.py) and known reference
transcripts.

What it measures per clip window:
  - text:    concatenation of FINAL events attributed to that playback window
  - wer:     word error rate vs the reference transcript (punctuation-stripped)
  - finalize_latency_s: time from the clip's audio ending to its last FINAL
  - first_final_s:      time from playback start to the first FINAL

What it reports across repeats:
  - per-clip WER mean/max, mean latencies
  - determinism: whether every repeat produced byte-identical transcript text

Usage:
  .venv/bin/python scripts/stress_full_pipeline.py [options]

The narrator config used for the run is written to scripts/output/stress_config.yaml
(edit gen/defaults there); CLI flags override the ASR fields for A/B tuning.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")  # for scripts/ (piper voice paths)

import numpy as np

from narrator.app_controller import AppController
from narrator.asr.metrics import normalize as _normalize
from narrator.asr.metrics import wer as _wer

logger = logging.getLogger("stress")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "scripts" / "output"
CLIPS_DIR = REPO_ROOT / "scripts" / "test_clips"

SINK_NAME = "narrator_mic_sink"
SOURCE_NAME = "narrator_mic_source"
# audio.device substring: must match the PortAudio device name whatever form
FRAME_MS = 30
SAMPLE_RATE = 16000

# A FINAL triggered by the *previous* clip's end-of-speech silence can land
# up to ~1s after the next clip starts (VAD decay + finalize decode over a
# long window). Attribution extends each window this far into the next clip's
# playback; the next clip's own first FINAL can't arrive that early (it needs
# VAD pickup + several decode steps to reach agreement, ~2s in).
TAIL_ATTRIBUTION_S = 1.0


# --------------------------------------------------------------------------
# transcript scoring lives in narrator.asr.metrics (shared with the
# autoconfigure quality probe so both score identically).
# --------------------------------------------------------------------------

normalize = _normalize
wer = _wer


# --------------------------------------------------------------------------
# PipeWire virtual mic
# --------------------------------------------------------------------------

def ensure_virtual_mic() -> subprocess.Popen | None:
    """Start the PipeWire loopback (virtual mic) unless one already exists.
    Returns the process when this call started it, so the caller can stop
    it again: while it exists it is the desktop's DEFAULT audio source, so
    leaving it behind silently changes the operator's microphone."""
    result = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True, text=True)
    if SINK_NAME in result.stdout:
        return None
    logger.info("starting virtual mic (%s -> %s)", SINK_NAME, SOURCE_NAME)
    loopback = subprocess.Popen(
        [
            "pw-loopback", "-n", "Narrator Mic Sink",
            "--capture-props", f'media.class=Audio/Sink node.name={SINK_NAME} node.description="Narrator Mic Sink"',
            "--playback-props", f'media.class=Audio/Source node.name={SOURCE_NAME} node.description="Narrator Mic Source"',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    return loopback


def resolve_input_device() -> str:
    """Return the PortAudio device name narrator should capture from.

    The PipeWire loopback registers 'Narrator Mic Source' and (as verified via
    wpctl) becomes the PipeWire *default source*, so PortAudio's generic
    'default'/'pipewire' ALSA devices route to it. PortAudio never lists the
    loopback as its own device, so match on the generic name instead.
    """
    import sounddevice as sd

    names = [info["name"] for info in sd.query_devices() if info.get("max_input_channels", 0) > 0]
    for candidate in ("default", "pipewire"):
        if candidate in names:
            logger.info("using input device %r (routed to PipeWire default source)", candidate)
            return candidate
    raise RuntimeError(f"no generic PipeWire capture device found; available inputs: {names}")


def preflight_capture_plays_audio(device_name: str, wav_path: Path) -> None:
    """Play 2s of the first clip and assert the captured signal is non-silent."""
    import sounddevice as sd

    device_index = next(
        i for i, info in enumerate(sd.query_devices())
        if info.get("max_input_channels", 0) > 0 and info["name"] == device_name
    )
    frames: list[np.ndarray] = []
    stream = sd.InputStream(device=device_index, channels=1, samplerate=SAMPLE_RATE,
                            blocksize=int(SAMPLE_RATE * FRAME_MS / 1000), dtype="float32")
    stream.start()
    player = subprocess.Popen(
        ["pw-play", f"--target={SINK_NAME}", str(wav_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and player.poll() is None:
        data, overflowed = stream.read(int(SAMPLE_RATE * 0.1))
        frames.append(data[:, 0].copy())
    stream.stop()
    player.wait()
    audio = np.concatenate(frames) if frames else np.zeros(1, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio**2)))
    logger.info("preflight capture RMS: %.4f (%.1fs captured)", rms, len(audio) / SAMPLE_RATE)
    if rms < 0.005:
        raise RuntimeError(
            "virtual mic carried no signal -- PipeWire loopback not working; "
            "run scripts/play_test_audio.sh manually to debug"
        )


# --------------------------------------------------------------------------
# clip schedule
# --------------------------------------------------------------------------

@dataclass
class ClipWindow:
    clip_id: str
    reference: str
    wav: Path
    duration_s: float
    gap_after_s: float = 0.8
    t_play_start: float = 0.0
    t_play_end: float = 0.0
    finals: list[tuple[float, str]] = field(default_factory=list)  # (t, text)

    @property
    def text(self) -> str:
        return " ".join(text for _, text in self.finals)


def load_schedule(repeats: int, gap_s: float) -> list[ClipWindow]:
    manifest = json.loads((CLIPS_DIR / "manifest.json").read_text(encoding="utf-8"))
    windows: list[ClipWindow] = []
    for clip in manifest["clips"]:
        wav = CLIPS_DIR / f"{clip['id']}.wav"
        if not wav.exists():
            raise SystemExit(f"missing clip {wav}; run scripts/gen_test_clips.py first")
        reference = clip.get("reference", clip["transcript"])
        with wave.open(str(wav), "rb") as w:
            duration = w.getnframes() / w.getframerate()
        for _ in range(repeats):
            windows.append(ClipWindow(clip["id"], reference, wav, duration))
    # silence gap between consecutive playbacks
    for window in windows:
        window.gap_after_s = gap_s
    windows[-1].gap_after_s = 2.0  # tail silence to flush the last utterance
    return windows


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------

def write_config(path: Path, asr_overrides: dict, language: str) -> None:
    from narrator.config.schema import AsrDevice, RootConfig

    config = RootConfig()
    config.audio.device = "default"
    config.gui.enabled = False
    config.app.log_level = "WARNING"
    # --language pl pins the whitelist to Polish (permanent-forced); "auto"
    # runs unrestricted per-utterance detection. Either way the whitelist is
    # the only language control.
    asr: dict = {
        "model": "large-v3",
        "device": AsrDevice.AUTO,
        "language_whitelist": [] if language == "auto" else [language],
    }
    asr.update(asr_overrides)
    config.asr = config.asr.model_copy(update=asr)
    from narrator.config.loader import save_config
    save_config(config, path)


def run_once(windows: list[ClipWindow], config_path: Path) -> dict:
    """One full playback pass against a freshly-started narrator app."""
    for window in windows:
        window.finals.clear()
        window.t_play_start = 0.0
        window.t_play_end = 0.0
    controller = AppController(config_path)
    controller.start()

    # Wait for the ASR worker to resolve its backend (model load can take ~30s).
    deadline = time.monotonic() + 180
    while controller.resolved_asr_backend == "unresolved":
        if time.monotonic() > deadline:
            raise RuntimeError("ASR worker never reported a backend")
        time.sleep(0.5)
    logger.info("ASR backend: %s; settling 2s before playback", controller.resolved_asr_backend)
    time.sleep(2.0)

    # Tap the renderer's final-text sink to record FINAL events with timestamps.
    renderer = controller._renderer
    recorded: list[tuple[float, str]] = []
    original_append = renderer.append_final_text

    def append_final_text(text: str) -> None:
        recorded.append((time.monotonic(), text))
        original_append(text)

    renderer.append_final_text = append_final_text  # type: ignore[method-assign]

    try:
        t0 = time.monotonic()
        for window in windows:
            window.t_play_start = time.monotonic()
            subprocess.run(
                ["pw-play", f"--target={SINK_NAME}", str(window.wav)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            window.t_play_end = time.monotonic()
            time.sleep(window.gap_after_s)

        # Grace period: let the tail utterance finalize.
        time.sleep(3.0)

        # Attribute each FINAL to the clip window whose playback interval
        # [start - epsilon, next start + TAIL_ATTRIBUTION_S) contains it.
        for t, text in recorded:
            for i, window in enumerate(windows):
                if i + 1 < len(windows):
                    window_end = windows[i + 1].t_play_start + TAIL_ATTRIBUTION_S
                else:
                    window_end = float("inf")
                if window.t_play_start - 0.2 <= t < window_end:
                    window.finals.append((t, text))
                    break

        per_clip = []
        for window in windows:
            last_final_t = max((t for t, _ in window.finals), default=None)
            per_clip.append({
                "clip": window.clip_id,
                "text": window.text,
                "reference": window.reference,
                "wer": round(wer(window.reference, window.text), 4),
                "finalize_latency_s": round(last_final_t - window.t_play_end, 3) if last_final_t else None,
                "first_final_s": round(window.finals[0][0] - window.t_play_start, 3) if window.finals else None,
            })

        transcript = [w.text for w in windows]
        # Normalized word content (punctuation/case stripped): the meaningful
        # text for stability comparison, since whisper's punctuation/casing
        # jitters between otherwise identical decodes.
        transcript_normalized = [normalize(w.text) for w in windows]
        return {
            "backend": controller.resolved_asr_backend,
            "per_clip": per_clip,
            "transcript": transcript,
            "transcript_normalized": transcript_normalized,
            "wer_mean": round(float(np.mean([c["wer"] for c in per_clip])), 4),
            "latency_mean_s": round(float(np.mean([
                c["finalize_latency_s"] for c in per_clip if c["finalize_latency_s"] is not None
            ])), 3) if any(c["finalize_latency_s"] is not None for c in per_clip) else None,
            "clips_empty": sum(1 for c in per_clip if not c["text"]),
        }
    finally:
        controller.stop()


def _run_passes(args, overrides: dict) -> list[dict]:
    device_name = resolve_input_device()
    first_wav = CLIPS_DIR / "intro.wav"
    if not first_wav.exists():
        raise SystemExit("clips missing; run scripts/gen_test_clips.py first")
    preflight_capture_plays_audio(device_name, first_wav)

    windows = load_schedule(args.repeats, args.gap)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUT_DIR / "stress_config.yaml"
    write_config(config_path, overrides, args.language)

    passes = []
    for pass_index in range(args.passes):
        logger.info("=== pass %d/%d ===", pass_index + 1, args.passes)
        result = run_once(windows, config_path)
        passes.append(result)
        logger.info(
            "pass %d: WER mean=%.3f, finalize latency mean=%ss, empty clips=%d",
            pass_index + 1, result["wer_mean"], result["latency_mean_s"], result["clips_empty"],
        )

    return passes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3, help="playbacks of each sentence per pass")
    ap.add_argument("--passes", type=int, default=2, help="number of full passes (determinism check)")
    ap.add_argument("--gap", type=float, default=2.5,
                    help="silence between playbacks (s); keep > end_silence_ms/1000 + VAD lag "
                         "+ finalize decode so each clip finalizes before the next starts")
    ap.add_argument("--model", default=None, help="asr.model override (e.g. large-v3-turbo)")
    ap.add_argument("--step-ms", type=int, default=None, help="asr.step_ms override")
    ap.add_argument("--agreement-passes", type=int, default=None, help="asr.agreement_passes override")
    ap.add_argument("--end-silence-ms", type=int, default=None, help="asr.end_silence_ms override")
    ap.add_argument("--language", default="pl", choices=["pl", "auto"],
                    help="whitelist shortcut: 'pl' pins Polish; 'auto' detects per utterance "
                         "(any language whisper reports)")
    args = ap.parse_args()

    overrides: dict = {}
    if args.model:
        overrides["model"] = args.model
    if args.step_ms:
        overrides["step_ms"] = args.step_ms
    if args.agreement_passes:
        overrides["agreement_passes"] = args.agreement_passes
    if args.end_silence_ms:
        overrides["end_silence_ms"] = args.end_silence_ms

    loopback = ensure_virtual_mic()
    try:
        passes = _run_passes(args, overrides)
    finally:
        if loopback is not None:
            logger.info("stopping the virtual mic this run started")
            loopback.terminate()
            loopback.wait(timeout=10)

    # Two determinism levels: exact transcript text (affected by whisper's
    # punctuation/casing jitter and by ms-level window-padding differences
    # between live repeats), and per-clip WER after punctuation-stripped
    # normalization (the meaningful regression metric).
    exact_deterministic = all(p["transcript"] == passes[0]["transcript"] for p in passes[1:])
    wer_vectors = [tuple(round(c["wer"], 4) for c in p["per_clip"]) for p in passes]
    wer_stable = all(v == wer_vectors[0] for v in wer_vectors[1:])
    # Per-playback stability: how many individual playbacks produced the same
    # normalized words as the same playback in pass 1 (punctuation ignored).
    total = len(passes[0]["transcript_normalized"])
    stable = sum(
        all(p["transcript_normalized"][i] == passes[0]["transcript_normalized"][i] for p in passes[1:])
        for i in range(total)
    )
    report = {
        "overrides": overrides,
        "deterministic_across_passes": exact_deterministic,
        "wer_stable_across_passes": wer_stable,
        "playbacks_stable_normalized": f"{stable}/{total}",
        "passes": passes,
    }
    report_path = OUT_DIR / "stress_results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"exact-transcript determinism across {len(passes)} passes: {exact_deterministic}")
    print(f"per-clip WER stability across passes:          {wer_stable}")
    print(f"playbacks with identical words (punct-stripped): {report['playbacks_stable_normalized']}")
    for i, p in enumerate(passes, 1):
        print(f"pass {i}: WER mean {p['wer_mean']:.3f} | latency {p['latency_mean_s']}s | empty {p['clips_empty']}")
    print(f"\nfirst-pass per-clip detail:")
    for c in passes[0]["per_clip"]:
        print(f"  {c['clip']:<8} WER {c['wer']:.2f}  lat {c['finalize_latency_s']}  | {c['text'][:70]}")
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
