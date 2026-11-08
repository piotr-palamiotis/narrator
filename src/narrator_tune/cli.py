"""`narrator-tune`: offline tuning for narrator (see pyproject [project.scripts]).

Subcommands:
  configure  search this machine for the model + decode settings whose live
             captions best match scripts/test_audio/, and write the winner
             into narrator's config.yaml (`make configure`)
  presets    maintainer mode: tune every preset tier and write narrator's
             presets.yaml (`make tune-presets`)
  analyze    hand the accumulated run reports to a local Ollama model
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from narrator_tune.paths import TUNE_CONFIG_PATH, legacy_dirs_in_use


def default_narrator_config_path() -> Path:
    """The config `configure` writes to by default: the repo's
    config/config.yaml (what the service runs) when it exists, else
    narrator's bare-run default in the platform user config dir."""
    from narrator.paths import platform_config_path, repo_config_path

    repo_config = repo_config_path()
    if repo_config is not None and repo_config.exists():
        return repo_config
    return platform_config_path()


def _setup_logging() -> None:
    # Stage progress from autoconfigure.py and the forwarded subprocess output.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


def _legacy_notice() -> None:
    for used, wanted in legacy_dirs_in_use():
        print(
            f"Note: using the old location {used} (move it to {wanted} to adopt the new layout).",
            file=sys.stderr,
        )


def _cmd_configure(args: argparse.Namespace) -> int:
    from narrator.config.loader import ensure_config_exists
    from narrator_tune.autoconfigure import autoconfigure
    from narrator_tune.config import load_tune_config

    _setup_logging()
    _legacy_notice()
    config_path: Path = args.config or default_narrator_config_path()
    ensure_config_exists(config_path)
    tune_config = load_tune_config(args.tune_config, narrator_config_path=config_path)
    models = args.models or ([args.model] if args.model else None)
    try:
        outcome = autoconfigure(
            config_path,
            models=models,
            quick=args.quick,
            budget_minutes=args.budget_minutes,
            dry_run=args.dry_run,
            allow_download=args.allow_download,
            rebench=args.rebench,
            tune_config=tune_config,
        )
    except RuntimeError as exc:
        print(f"\nnarrator-tune configure stopped: {exc}", file=sys.stderr)
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
            print("A running narrator picks the change up by itself (config hot reload restarts its ASR worker).")
    if outcome.report_path is not None:
        print(f"Report: {outcome.report_path}")
    if args.analyze:
        print("\n--- Analyzing collected reports ---\n")
        return _cmd_analyze(args)
    return 0


def _cmd_presets(args: argparse.Namespace) -> int:
    from narrator.config.loader import ensure_config_exists
    from narrator.config.presets import PRESETS_PATH
    from narrator_tune.config import load_tune_config
    from narrator_tune.presets import format_outcomes, tune_presets

    _setup_logging()
    _legacy_notice()
    config_path: Path = args.config or default_narrator_config_path()
    ensure_config_exists(config_path)
    output: Path = args.output or PRESETS_PATH
    tune_config = load_tune_config(args.tune_config, narrator_config_path=config_path)
    try:
        outcomes = tune_presets(
            config_path,
            tiers=args.tiers,
            output=output,
            quick=args.quick,
            budget_minutes=args.budget_minutes,
            dry_run=args.dry_run,
            allow_download=args.allow_download,
            tune_config=tune_config,
        )
    except ValueError as exc:
        print(f"narrator-tune presets: {exc}", file=sys.stderr)
        return 2
    print("\nPreset tiers:")
    for line in format_outcomes(outcomes):
        print(line)
    if any(o.after is not None for o in outcomes):
        print(f"\n{'Dry run: not written to' if args.dry_run else 'Written to'} {output}")
    else:
        print(f"\nNo tier was measured; {output} was not changed.")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from narrator_tune.ollama_analyzer import DEFAULT_MODEL, analyze_reports, save_analysis_result

    _legacy_notice()
    print(f"Analyzing accumulated narrator-tune reports via Ollama ({DEFAULT_MODEL})...")
    result = analyze_reports()
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print("\n=== Analysis Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    path = save_analysis_result(result)
    print(f"\nAnalysis saved to: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="narrator-tune", description="Offline tuning for narrator's live captioner")
    sub = parser.add_subparsers(dest="command", required=True)

    configure = sub.add_parser(
        "configure",
        help="Find the model + decode settings whose live captions best match scripts/test_audio/, "
             "and write them into narrator's config.yaml",
        description="Search this machine for the model + decode settings whose live captions best match the "
                    "reference recordings in scripts/test_audio/, and write the winner into narrator's "
                    "config.yaml. Results are cached (var/tune/cache), so reruns resume.",
    )
    configure.add_argument("--config", type=Path, default=None,
                           help="narrator's config.yaml to tune (default: narrator's platform user config)")
    configure.add_argument("--tune-config", type=Path, default=TUNE_CONFIG_PATH,
                           help=f"The search's own settings (default: {TUNE_CONFIG_PATH}; created on first use)")
    configure.add_argument("--quick", action="store_true",
                           help="Small samples and one search pass (~1 h smoke run)")
    configure.add_argument("--budget-minutes", type=float, default=None,
                           help="Wall-clock budget for new decoding (default: time_budget_minutes in tune.yaml)")
    configure.add_argument("--dry-run", action="store_true",
                           help="Run the search but do not change config.yaml")
    configure.add_argument("--allow-download", action="store_true",
                           help="Download candidate models missing from the local cache")
    configure.add_argument("--rebench", action="store_true",
                           help="Re-measure model speed instead of reusing stored timings (after a GPU/driver change)")
    configure.add_argument("--model", type=str, default=None,
                           help="Benchmark a single specific model (e.g. --model large-v3); shortcut for --models")
    configure.add_argument("--models", nargs="+", default=None,
                           help="Candidate models to benchmark (default: candidate_models in tune.yaml)")
    configure.add_argument("--analyze", action="store_true",
                           help="Afterwards, analyze the accumulated reports via a local Ollama model")
    configure.set_defaults(func=_cmd_configure)

    presets = sub.add_parser(
        "presets",
        help="Maintainer mode: tune every preset tier and write narrator's presets.yaml (measured: true rows)",
        description="Run the configure search once per preset tier (the tier's model; step_ms and "
                    "agreement_passes searched; the tier's load limit and unit from its `tune:` entry) and "
                    "write the results into narrator's presets.yaml. Each tier is a full search: hours.",
    )
    presets.add_argument("--config", type=Path, default=None,
                         help="narrator config whose non-preset settings the search runs with "
                              "(default: as for configure); never written")
    presets.add_argument("--tune-config", type=Path, default=TUNE_CONFIG_PATH,
                         help=f"The search's own settings (default: {TUNE_CONFIG_PATH})")
    presets.add_argument("--tiers", nargs="+", default=None, help="Only these tiers (default: all)")
    presets.add_argument("--output", type=Path, default=None,
                         help="presets.yaml to update (default: narrator's packaged src/narrator/config/presets.yaml)")
    presets.add_argument("--quick", action="store_true", help="Small samples, one search pass per tier")
    presets.add_argument("--budget-minutes", type=float, default=None,
                         help="Budget for new decoding PER TIER (default: time_budget_minutes in tune.yaml)")
    presets.add_argument("--dry-run", action="store_true", help="Run the searches but do not write presets.yaml")
    presets.add_argument("--allow-download", action="store_true",
                         help="Download tier models missing from the local cache")
    presets.set_defaults(func=_cmd_presets)

    analyze = sub.add_parser("analyze", help="Analyze the accumulated run reports via a local Ollama model")
    analyze.set_defaults(func=_cmd_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
