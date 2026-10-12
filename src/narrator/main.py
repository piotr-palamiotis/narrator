"""CLI entry point: `narrator` (see pyproject.toml [project.scripts])."""
from __future__ import annotations

import argparse
import json
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


def _run_autoconfigure(
    config_path: Path,
    models: list[str] | None,
    *,
    quick: bool = False,
    budget_minutes: float | None = None,
    dry_run: bool = False,
    allow_download: bool = False,
    rebench: bool = False,
) -> int:
    """Search the model + decode settings whose live captions best match the
    reference recordings, and write the winner into config.yaml. See
    narrator.asr.autoconfigure."""
    from narrator.asr.autoconfigure import autoconfigure

    # Configure logging so stage progress from autoconfigure.py and the
    # forwarded subprocess output are visible (the normal app path defers
    # this to AppController, but --autoconfigure never creates one).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        outcome = autoconfigure(
            config_path,
            models=models,
            quick=quick,
            budget_minutes=budget_minutes,
            dry_run=dry_run,
            allow_download=allow_download,
            rebench=rebench,
        )
    except RuntimeError as exc:
        print(f"\nAutoconfigure stopped: {exc}", file=sys.stderr)
        print(f"{config_path} was not changed.", file=sys.stderr)
        return 1

    print("\nModels:")
    for candidate in outcome.candidates:
        if candidate.error:
            print(f"  {candidate.model:<16} {candidate.error}")
        else:
            cost = candidate.profile.cost
            print(f"  {candidate.model:<16} 10s of speech decodes in {cost.typical_decode_seconds():.2f}s")
    if outcome.table:
        print(f"\n  {'configuration':<28} {'set':<8} {'WER':>6}  {'sub/del/ins':>15}  {'delay p50':>9}  {'busy':>5}")
        for row in outcome.table:
            delay = "n/a" if row["latency_p50"] is None else f"{row['latency_p50']:.1f}s"
            edits = f"{row['substitutions']}/{row['deletions']}/{row['insertions']}"
            print(
                f"  {row['label'][:28]:<28} {row['set']:<8} {100 * row['wer']:5.1f}%  {edits:>15}  "
                f"{delay:>9}  {100 * row['busy_fraction']:4.0f}%"
            )
    print(f"\nResult: {outcome.reason}")
    if outcome.changes:
        verb = "Written to" if outcome.written else "Not written (see above); the winner would change"
        print(f"{verb} {config_path}:" if outcome.written else f"{verb}:")
        for name, (old, new) in outcome.changes.items():
            print(f"  asr.{name}: {old} -> {new}")
        if outcome.written:
            print(f"Previous config kept as {config_path.name}.bak")
    if outcome.report_path is not None:
        print(f"Report: {outcome.report_path}")
    return 0


def _run_analyze_reports() -> int:
    """Analyze accumulated autoconfigure reports via local Ollama (qwen3.5:9b)."""
    from narrator.asr.ollama_analyzer import analyze_reports, save_analysis_result

    print("Analyzing accumulated autoconfigure reports via Ollama (qwen3.5:9b)...")
    result = analyze_reports()
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print("\n=== Analysis Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    path = save_analysis_result(result)
    print(f"\nAnalysis saved to: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="narrator", description="Live captions streamed as an NDI RGBA source")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (default: platform user config dir)")
    parser.add_argument("--list-devices", action="store_true", help="List microphone input devices and exit")
    parser.add_argument("--no-gui", action="store_true", help="Force headless mode even if gui.enabled is true in config")
    parser.add_argument("--autoconfigure", action="store_true",
                        help="Find the model + decode settings whose live captions best match the "
                             "reference recordings in scripts/test_audio/, write them to config, and exit")
    parser.add_argument("--quick", action="store_true",
                        help="With --autoconfigure: small samples and one search pass (~1 h smoke run)")
    parser.add_argument("--budget-minutes", type=float, default=None,
                        help="With --autoconfigure: wall-clock budget for new decoding "
                             "(default: autoconfigure.time_budget_minutes)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --autoconfigure: run the search but do not change config.yaml")
    parser.add_argument("--allow-download", action="store_true",
                        help="With --autoconfigure: download candidate models missing from the local cache")
    parser.add_argument("--rebench", action="store_true",
                        help="With --autoconfigure: re-measure model speed instead of reusing stored timings "
                             "(after a GPU/driver change)")
    parser.add_argument("--collect-reports", action="store_true",
                        help="Accepted for compatibility: autoconfigure now always writes its report "
                             "to scripts/autoconfigure_reports/")
    parser.add_argument("--analyze-reports", action="store_true",
                        help="With --autoconfigure: analyze the reports afterwards via a local "
                             "Ollama model (qwen3.5:9b). Standalone: analyze existing reports.")
    parser.add_argument("--model", type=str, default=None,
                        help="With --autoconfigure: benchmark a single specific model "
                             "(e.g. --model large-v3). Shortcut for --models <name>.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="With --autoconfigure: candidate models to benchmark (default: all known)")
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        return 0

    config_path = args.config or default_config_path()
    ensure_config_exists(config_path)

    # Resolve explicit single-model shorthand.
    models = args.models or ([args.model] if args.model else None)

    if args.quick or args.dry_run or args.budget_minutes is not None or args.rebench or args.allow_download:
        args.autoconfigure = True  # these only make sense for an autoconfigure run
    if args.autoconfigure:
        rc = _run_autoconfigure(
            config_path,
            models,
            quick=args.quick,
            budget_minutes=args.budget_minutes,
            dry_run=args.dry_run,
            allow_download=args.allow_download,
            rebench=args.rebench,
        )
        if rc != 0:
            return rc
        if args.analyze_reports:
            print("\n--- Analyzing collected autoconfigure reports ---\n")
            return _run_analyze_reports()
        return 0

    if args.analyze_reports:
        return _run_analyze_reports()

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
