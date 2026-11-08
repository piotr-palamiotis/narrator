"""CLI entry point: `narrator` (see pyproject.toml [project.scripts])."""
from __future__ import annotations

import argparse
import logging
import signal
import multiprocessing
import sys
import threading
from pathlib import Path

from narrator.app_controller import AppController
from narrator.config.loader import ensure_config_exists
from narrator.paths import platform_config_path

logger = logging.getLogger(__name__)


def default_config_path() -> Path:
    """Config for a bare `narrator` run. The service (and anyone wanting the
    self-contained layout) passes --config <repo>/config/config.yaml."""
    return platform_config_path()


def run_headless(controller, shutdown: threading.Event, poll_seconds: float = 0.5) -> int:
    """Block until `shutdown` is set (SIGTERM/SIGINT), then run the one
    shutdown path (controller.stop(): web UI, NDI, ASR worker, audio,
    transcript flush). Returns the exit status: 0 for a requested shutdown.

    Since the 2026-10-17 failure policy the app never stops itself: an ASR
    crash loop, a missing unit/model/mic or NDI runtime leave the app up with
    a problem shown in the web UI (narrator.health). The `stop_requested`
    branch stays as a safety net -- should anything ever stop the controller
    without a signal, exit 1 so systemd's Restart=on-failure brings it back.
    """
    try:
        while not shutdown.wait(poll_seconds):
            if controller.stop_requested:
                break
    finally:
        controller.stop()
    return 0 if shutdown.is_set() else 1


def install_signal_handlers(on_signal) -> None:
    """Route SIGINT and SIGTERM to `on_signal(signum)`."""

    def _handle(signum, frame) -> None:
        logger.info("Received signal %s, shutting down", signum)
        on_signal(signum)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _list_devices() -> None:
    from narrator.audio.devices import list_input_devices

    for device in list_input_devices():
        print(
            f"[{device.index}] {device.name} "
            f"({device.max_input_channels} ch, default rate: {device.default_sample_rate:.0f} Hz)"
        )

    # NDI audio sources on the network; selected with `audio.device: "ndi:<name>"`.
    # Discovery never fails the listing (no NDI runtime -> just no entries).
    from narrator.audio.ndi_capture import list_ndi_sources

    print("NDI sources (audio.device: \"ndi:<name>\"):")
    try:  # narrator's own output (ndi.source_name) is not a valid audio source
        from narrator.config.loader import load_config

        own_name = load_config(default_config_path()).ndi.source_name
    except Exception:
        own_name = None
    sources = list_ndi_sources(timeout_s=2.0, own_stream_name=own_name)
    for name in sources:
        print(f"  ndi:{name}")
    if not sources:
        print("  (none found)")


def main() -> int:
    # Frozen build: the spawn-started ASR worker re-executes this binary;
    # this makes that child run the worker instead of the whole app. No-op
    # elsewhere.
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(prog="narrator", description="Live captions streamed as an NDI RGBA source")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml, created from the defaults if missing (default: platform user config dir; the service passes <repo>/config/config.yaml)")
    parser.add_argument("--list-devices", action="store_true", help="List microphone input devices and exit")
    parser.add_argument(
        "--pick-preset", action="store_true",
        help="Benchmark this machine (on the configured asr.unit) and set asr.preset to the most accurate "
             "preset it can carry; previous config kept as config.yaml.bak",
    )
    parser.add_argument("--dry-run", action="store_true", help="With --pick-preset: report the choice, write nothing")
    parser.add_argument("--download", action="store_true",
                        help="With --pick-preset: download preset models missing locally (default: skip them)")
    parser.add_argument("--all-presets", action="store_true",
                        help="With --pick-preset: measure every preset instead of stopping at the first that fits")
    parser.add_argument(
        "--headless", action="store_true",
        help="Service mode (what the systemd unit passes): never open a browser. The web UI still "
             "starts when web.enabled is true -- the service is controlled from the browser",
    )
    parser.add_argument(
        "--no-web", "--no-gui", dest="no_web", action="store_true",
        help="Run without the web UI (and OSC input) even if web.enabled is true; implies --headless. "
             "--no-gui is the old spelling from the Tk GUI",
    )
    parser.add_argument(
        "--open-ui", action="store_true",
        help="Open the web UI of the narrator already running (logged in with its startup token) and exit",
    )
    parser.add_argument(
        "--log-file", action="store_true",
        help="Also write a rotating log to <repo>/var/log/narrator.log (off by default; stderr/journald always)",
    )
    # The recording draft pass runs as a subprocess of the app (this flag is
    # how the frozen build reaches it; a checkout uses -m narrator.recording.draft).
    parser.add_argument("--draft-recording", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.draft_recording is not None:
        from narrator.recording.draft import main as draft_main

        return draft_main([str(args.draft_recording)])

    if args.list_devices:
        _list_devices()
        return 0

    if args.open_ui:
        return open_ui()

    if args.pick_preset:
        from narrator.asr.preset_picker import main as pick_preset_main

        config_path = args.config or default_config_path()
        ensure_config_exists(config_path)
        return pick_preset_main(
            config_path, dry_run=args.dry_run, download=args.download, measure_all=args.all_presets
        )

    if args.log_file:
        from narrator.logsetup import enable_file_logging

        log_path = enable_file_logging()

    web_wanted = not args.no_web
    if web_wanted:
        # Before the controller starts, so the web UI's log view has the
        # startup lines too.
        from narrator.web.logbuffer import install_log_ring

        install_log_ring()

    config_path = args.config or default_config_path()
    ensure_config_exists(config_path)

    controller = AppController(config_path)
    controller.start()
    logging.getLogger().setLevel(controller._config.app.log_level)
    if args.log_file:
        logger.info("Also logging to %s", log_path)

    if web_wanted:
        from narrator.web.server import WebServer

        web = WebServer(controller)
        controller.attach_web(web)  # config changes reach it; stop() stops it first
        web.start()
        if not (args.headless or args.no_web) and web.login_url() and _has_display():
            _open_browser(web.login_url())

    shutdown = threading.Event()
    install_signal_handlers(lambda signum: shutdown.set())
    logger.info("Running. Ctrl+C or SIGTERM to stop. Config: %s", config_path)
    return run_headless(controller, shutdown)


def _has_display() -> bool:
    if sys.platform.startswith("linux"):
        import os

        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def _open_browser(url: str) -> None:
    """Open the UI in the default browser without blocking startup."""
    import webbrowser

    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()


def open_ui() -> int:
    """`narrator --open-ui`: open the running instance's web UI, logged in.
    The token file (<var>/web-token.json, 0600) is written by the running
    app's web server and removed when it stops."""
    from narrator.web.security import default_token_path, read_token_file

    path = default_token_path()
    try:
        info = read_token_file(path)
    except (OSError, ValueError):
        print(f"No running narrator web UI found ({path} is missing or unreadable).", file=sys.stderr)
        return 1
    url = f"{info['url']}/auth?token={info['token']}"
    print(f"Opening {info['url']}")
    import webbrowser

    if not webbrowser.open(url):
        print(f"Could not start a browser; open this URL yourself:\n  {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
