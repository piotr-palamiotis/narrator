"""Live demo of whitelist-based language selection (pl+en whitelist).

Plays a deterministic schedule of Polish and English Piper fixtures through
the PipeWire virtual mic into a real AppController and prints the observed
caption timeline next to the expected behavior: with a pl+en whitelist each
utterance is decoded in the highest-confidence allowed language, so captions
follow the speaker's language per utterance.

Usage: .venv/bin/python scripts/live_language_demo.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from stress_full_pipeline import (  # noqa: E402
    OUT_DIR,
    SINK_NAME,
    ensure_virtual_mic,
    resolve_input_device,
)

from narrator.app_controller import AppController  # noqa: E402
from narrator.config.loader import save_config  # noqa: E402
from narrator.config.schema import AsrDevice, RootConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PL = REPO / "scripts" / "test_clips"
EN = REPO / "scripts" / "test_clips_en"
GAP_S = 2.5
TAIL_ATTRIBUTION_S = 1.0

# (label, wav, spoken language, expected caption language)
SCHEDULE = [
    ("pl-1  greet ", PL / "intro.wav", "pl", "pl"),
    ("pl-2  punct ", PL / "punct.wav", "pl", "pl"),
    ("en-1  intro ", EN / "english.wav", "en", "pl"),  # streak 1/2: default language
    ("en-2  again ", EN / "english.wav", "en", "pl"),  # streak 2/2: flips AFTER this
    ("en-3  after ", EN / "english.wav", "en", "en"),  # now captioned in English
    ("pl-3  back1 ", PL / "punct.wav", "pl", "en"),    # streak 1/2: still English
    ("pl-4  back2 ", PL / "intro.wav", "pl", "en"),    # streak 2/2: flips AFTER this
    ("pl-5  final ", PL / "intro.wav", "pl", "pl"),    # Polish again
]


def main() -> None:
    import logging

    logging.basicConfig(level=logging.WARNING)

    ensure_virtual_mic()
    device_name = resolve_input_device()
    print(f"input device: {device_name}")

    config_path = OUT_DIR / "live_demo_config.yaml"
    config = RootConfig()
    config.audio.device = "default"
    config.gui.enabled = False
    config.app.log_level = "WARNING"
    config.asr = config.asr.model_copy(
        update={
            "model": "large-v3",
            "device": AsrDevice.AUTO,
            "language_whitelist": ["pl", "en"],
            "step_ms": 900,
        }
    )
    save_config(config, config_path)

    controller = AppController(config_path)
    controller.start()
    deadline = time.monotonic() + 180
    while controller.resolved_asr_backend == "unresolved":
        if time.monotonic() > deadline:
            raise SystemExit("ASR worker never reported a backend")
        time.sleep(0.5)
    print(f"backend: {controller.resolved_asr_backend}; settling 2s")
    time.sleep(2.0)

    recorded: list[tuple[float, str]] = []
    renderer = controller._renderer
    original = renderer.append_final_text

    def append_final_text(text: str) -> None:
        recorded.append((time.monotonic(), text))
        original(text)

    renderer.append_final_text = append_final_text  # type: ignore[method-assign]

    windows: list[dict] = []
    try:
        for label, wav, spoken, expected in SCHEDULE:
            t_start = time.monotonic()
            subprocess.run(
                ["pw-play", f"--target={SINK_NAME}", str(wav)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            windows.append({
                "label": label, "spoken": spoken, "expected": expected,
                "t_start": t_start, "t_end": time.monotonic(), "finals": [],
            })
            time.sleep(GAP_S)
        time.sleep(3.0)  # tail flush
    finally:
        controller.stop()

    # Attribute FINALs to playback windows (tail spills into the next window).
    for t, text in recorded:
        for i, w in enumerate(windows):
            end = windows[i + 1]["t_start"] + TAIL_ATTRIBUTION_S if i + 1 < len(windows) else float("inf")
            if w["t_start"] - 0.2 <= t < end:
                w["finals"].append(text)
                break

    print("\n=== live language-switching demo ===")
    hits = 0
    for w in windows:
        text = " ".join(w["finals"]).strip() or "(nothing)"
        ok = w["expected"] in ("pl", "en") and text != "(nothing)"
        # expected-language check: does the caption's language match the expectation?
        def looks_like(lang: str, s: str) -> bool:
            cyr = any("\u0400" <= ch <= "\u04FF" for ch in s)
            if lang == "en":
                return not cyr and all(ord(c) < 0x0180 for c in s)
            return not cyr if lang == "pl" else cyr
        match = looks_like(w["expected"], text)
        hits += match
        print(
            f"{w['label']} spoken={w['spoken']} -> expected [{w['expected']}]: "
            f"{'OK ' if match else 'MISS'} | {text[:90]}"
        )
    print(f"\n{hits}/{len(windows)} windows matched the expected caption language")


if __name__ == "__main__":
    main()
