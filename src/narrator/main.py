"""CLI entry point: `narrator` (see pyproject.toml [project.scripts])."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from platformdirs import user_config_dir

from narrator.app_controller import AppController
from narrator.config.loader import ensure_config_exists

logger = logging.getLogger(__name__)


def default_config_path() -> Path:
    return Path(user_config_dir("narrator")) / "config.yaml"


def _list_devices() -> None:
    from narrator.audio.devices import list_input_devices

    for device in list_input_devices():
        print(f"[{device.index}] {device.name} (default rate: {device.default_sample_rate:.0f} Hz)")


def _run_autoconfigure(config_path: Path, models: list[str] | None) -> int:
    """Benchmark candidate models, pick the best for live captioning, and
    persist the choice into config.yaml. See narrator.asr.autoconfigure."""
    from narrator.asr.autoconfigure import autoconfigure

    print("Benchmarking candidate ASR models (speed + accuracy on the Polish "
          "fixtures); this loads each model once and can take several minutes...")
    decision = autoconfigure(config_path, models=models)
    print(f"\nWinner: {decision.reason}")
    print(f"Configured: asr.model={decision.model}, asr.device={decision.device}, "
          f"asr.step_ms={decision.step_ms}")
    print(f"Written to: {config_path}")
    print("\nPer-candidate results:")
    for c in decision.candidates:
        if c.error:
            print(f"  {c.model:<12} FAILED: {c.error}")
        elif c.qualified:
            wer_text = f"{c.wer:.1%}" if c.wer is not None else "n/a"
            print(f"  {c.model:<12} 10s={c.decode_seconds_by_window.get('10.0s', float('nan')):.2f}s "
                  f"ratio={c.slowest_decode_ratio:.2f}x WER={wer_text} score={c.score:.3f}")
        else:
            reason = "doubled words" if c.doubled else ""
            if reason:
                print(f"  {c.model:<12} disqualified: {reason}")
            else:
                print(f"  {c.model:<12} disqualified (ratio={c.slowest_decode_ratio:.2f}x, "
                      f"WER={c.wer if c.wer is not None else 'n/a'})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="narrator", description="Live captions streamed as an NDI RGBA source")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (default: platform user config dir)")
    parser.add_argument("--list-devices", action="store_true", help="List microphone input devices and exit")
    parser.add_argument("--no-gui", action="store_true", help="Force headless mode even if gui.enabled is true in config")
    parser.add_argument("--autoconfigure", action="store_true",
                        help="Benchmark ASR models, write the best model/device/step_ms to config, and exit")
    parser.add_argument("--models", nargs="+", default=None,
                        help="With --autoconfigure: candidate models to benchmark (default: all known)")
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        return 0

    config_path = args.config or default_config_path()
    ensure_config_exists(config_path)

    if args.autoconfigure:
        return _run_autoconfigure(config_path, args.models)

    controller = AppController(config_path)
    controller.start()

    def _handle_signal(signum, frame) -> None:
        logger.info("Received signal %s, shutting down", signum)
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    gui_enabled = controller._config.gui.enabled and not args.no_gui
    if gui_enabled:
        from narrator.gui.app import run_gui

        try:
            run_gui(controller, config_path)
        finally:
            controller.stop()
    else:
        logger.info("Running headless. Ctrl+C to stop. Config: %s", config_path)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            controller.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
