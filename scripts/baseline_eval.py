"""Phase 0 baseline (todo B): the CURRENT live decoder on the reference library.

Thin driver over narrator_tune's own stages (bench, cost-model calibration,
offline ceiling, model screen, dev/trusted confirmation); no settings search
and it never writes config.yaml. Results are cached in narrator-tune's result
store, so a rerun is cheap and a different decoder/config is compared PAIRED
on exactly the same excerpts.

  python scripts/baseline_eval.py <config.yaml> <out.json> large-v3,medium,small,base,tiny [--dev] [--flash]
  python scripts/baseline_eval.py <config.yaml> <out.json> large-v3,medium \
      --variant 'seg:decoder="segmenter"' --variant 'seg-partials:decoder="segmenter",show_partial_captions=true'
  (large-v3 must always be listed: it is the offline-ceiling model.)

`--variant NAME:FIELD=JSON[,FIELD=JSON...]` (repeatable) scores further
configs -- the base plus those overrides -- for every listed model on the
screen set (and the dev set with --dev), and reports each PAIRED against
the base config of the same model on the same excerpts (errors saved and
its standard error, library_eval.paired_delta): positive = the variant is
better. The table is printed as markdown and stored under "variants".

Recordings come only through narrator_tune.real_fixtures.discover_recordings
(audio AND srt both present). Stages: screen = first 32 dev excerpts, dev = up
to 96, trusted = human-subtitle excerpts (reported separately).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from narrator_tune import autoconfigure as ac
from narrator_tune import library_eval as le
from narrator_tune.config import load_tune_config
from narrator_tune.paths import CACHE_DIR
from narrator_tune.report_library import save_summary  # noqa: F401  (kept for parity with autoconfigure)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("baseline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("out")
    parser.add_argument("models", help="comma separated; large-v3 is also the offline-ceiling model")
    parser.add_argument("--dev", action="store_true", help="also score the full dev set (and ceiling on it)")
    parser.add_argument("--flash", action="store_true", help="large-v3 with asr.flash_attn on/off on the screen set")
    parser.add_argument("--best", default="large-v3", help="model used for the dev/trusted/flash stages")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=JSON",
                        help="override an AsrConfig field for every stage, e.g. --set decoder='\"segmenter\"'")
    parser.add_argument("--variant", action="append", default=[], metavar="NAME:FIELD=JSON[,FIELD=JSON]",
                        help="also score base+overrides per model, paired against the base")
    parser.add_argument("--unit", default=None,
                        help="processing unit for every decode (default: the config's asr.unit, e.g. cuda:0)")
    parser.add_argument("--variants-only", action="store_true",
                        help="skip the flash/dev/trusted stages of the base (screen + variants only)")
    args = parser.parse_args()

    cfg_path, out = Path(args.config), Path(args.out)
    models = args.models.split(",")
    tuning = ac.Tuning.from_config(load_tune_config(narrator_config_path=cfg_path))
    tuning.candidate_models = models
    tuning.time_budget_minutes = 10**5
    cache = le.AudioCache(CACHE_DIR / "excerpts", max_mb=tuning.cache_max_mb, min_free_mb=tuning.min_free_disk_mb)
    ev = le.Evaluator(cache, le.ResultStore())
    bench_audio = ac._bench_audio_path(ev, tuning)
    unit = args.unit or _config_unit(cfg_path)
    log.info("unit: %s", unit)
    profiles = {}
    for m in models:
        c = ac.profile_model(m, unit, tuning, bench_audio, False)
        log.info("%s: %s %s", m, c.error, c.bench.get("decode_seconds_by_window"))
        if c.profile:
            profiles[m] = c.profile
    run = ac._Run(cfg_path, tuning, ac.Budget(10**5), ev, profiles)
    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in args.set)}
    base = type(run.baseline).model_validate({**run.baseline.model_dump(mode="json"), **overrides})
    run.baseline = base
    run.report["machine"] = le.machine_identity()
    run.report["overrides"] = overrides
    log.info("machine: %s ; baseline: %s", le.machine_identity(), base.model_dump(mode="json"))
    screen, dev, holdout, trusted = run.excerpts()
    run.report["excerpt_ids"] = {
        "screen": [x.id for x in screen], "dev": [x.id for x in dev],
        "holdout": [x.id for x in holdout], "trusted": [x.id for x in trusted],
    }
    for p in profiles.values():
        ac.calibrate_profile(p, base, screen[: ac.CALIBRATION_EXCERPTS], ev, False)
    run.report["cost_models"] = {m: p.cost.to_dict() for m, p in profiles.items()}
    sets = [screen, dev, trusted] if args.dev else [screen, trusted]
    res = run.drop_unmatchable(profiles["large-v3"], sets)
    screen, trusted = res[0], res[-1]
    dev = res[1] if args.dev else []
    log.info("after ceiling: screen %d dev %d trusted %d", len(screen), len(dev), len(trusted))
    for m in models:
        run._evaluate("model_screen", m, base.model_copy(update={"model": m}), screen, "screen")
        out.write_text(json.dumps(run.report, indent=1, default=str))
    base_scores = {}
    for m in models:
        (base_scores[(m, "screen")],) = ev.evaluate(profiles[m], [base.model_copy(update={"model": m})], screen)
        if dev:
            (base_scores[(m, "dev")],) = ev.evaluate(profiles[m], [base.model_copy(update={"model": m})], dev)
    rows = []
    for spec in args.variant:
        name, _, assignments = spec.partition(":")
        update = {}
        for part in _split_assignments(assignments):
            field, _, value = part.partition("=")
            update[field.strip()] = json.loads(value)
        for m in models:
            config = type(base).model_validate({**base.model_dump(mode="json"), **update, "model": m})
            for set_name, excerpts in (("screen", screen), ("dev", dev)):
                if not excerpts:
                    continue
                score = run._evaluate("variants", f"{m} {name}", config, excerpts, set_name)
                saved, se, shared = le.paired_delta(base_scores[(m, set_name)], score)
                rows.append({"variant": name, "model": m, "set": set_name, "overrides": update,
                             "saved_errors": saved, "paired_se": se, "shared_excerpts": shared,
                             **score.to_dict()})
                out.write_text(json.dumps(run.report | {"variant_table": rows}, indent=1, default=str))
    if rows:
        run.report["variant_table"] = rows
        base_rows = [{"variant": "base", "model": m, "set": st, "overrides": overrides, "saved_errors": 0,
                      "paired_se": 0.0, "shared_excerpts": sc.excerpts, **sc.to_dict()}
                     for (m, st), sc in base_scores.items()]
        print(_markdown(base_rows + rows))
    best = base.model_copy(update={"model": args.best})
    if args.variants_only:
        run.report["decode_minutes_spent"] = round(ev.decode_seconds_spent / 60.0, 1)
        out.write_text(json.dumps(run.report, indent=1, default=str))
        log.info("done")
        return 0
    if args.flash:
        for flash in (False, True):
            run._evaluate("flash_attn", f"{args.best} flash_attn={flash}", best.model_copy(update={"flash_attn": flash}), screen, "screen")
    if dev:
        run._evaluate("dev", f"{args.best} current", best, dev, "dev")
    if trusted:
        run._evaluate("trusted", f"{args.best} current", best, trusted, "trusted")
    run.report["decode_minutes_spent"] = round(ev.decode_seconds_spent / 60.0, 1)
    out.write_text(json.dumps(run.report, indent=1, default=str))
    log.info("done")
    return 0


def _config_unit(cfg_path: Path) -> str:
    """asr.unit of the config being evaluated ("auto" resolved by the backend)."""
    from narrator.config.loader import load_config

    return str(load_config(cfg_path).asr.unit)


def _split_assignments(text: str) -> list[str]:
    """Split "a=1,b=[1,2],c=\"x,y\"" on the commas between assignments
    (not inside JSON brackets or strings)."""
    parts, depth, quoted, current = [], 0, False, ""
    for ch in text:
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch in "[{":
            depth += 1
        elif not quoted and ch in "]}":
            depth -= 1
        if ch == "," and depth == 0 and not quoted:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def _fmt(value, scale=1.0, unit="", digits=2):
    return "n/a" if value is None else f"{value * scale:.{digits}f}{unit}"


def _markdown(rows: list[dict]) -> str:
    head = ("| variant | model | set | WER | S | D | I | saved vs base (paired SE) | delay p50 | delay p90 "
            "| first visible p50 / p90 | busy | rollovers |")
    lines = [head, "|" + "---|" * 13]
    for r in rows:
        paired = "-" if r["variant"] == "base" else f"{r['saved_errors']:+d} ({r['paired_se']:.1f})"
        lines.append(
            f"| {r['variant']} | {r['model']} | {r['set']} ({r['shared_excerpts']}) | {100 * r['wer']:.1f}% "
            f"| {r['substitutions']} | {r['deletions']} | {r['insertions']} | {paired} "
            f"| {_fmt(r['latency_p50'], unit=' s')} | {_fmt(r['latency_p90'], unit=' s')} "
            f"| {_fmt(r['first_visible_p50'])} / {_fmt(r['first_visible_p90'], unit=' s')} "
            f"| {100 * r['busy_fraction']:.0f}% | {r['rollovers']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
