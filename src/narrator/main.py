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


def main() -> int:
    parser = argparse.ArgumentParser(prog="narrator", description="Live captions streamed as an NDI RGBA source")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (default: platform user config dir)")
    parser.add_argument("--list-devices", action="store_true", help="List microphone input devices and exit")
    parser.add_argument("--no-gui", action="store_true", help="Force headless mode even if gui.enabled is true in config")
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        return 0

    config_path = args.config or default_config_path()
    ensure_config_exists(config_path)

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
