"""`narrator-tune presets` (`make tune-presets`): the maintainer mode that
produces narrator's presets.yaml rows (narrator.config.presets).

For each tier, the regular `configure` search runs on a scratch copy of
narrator's config whose asr section starts at the tier's current row
(preset: custom): the tier's model only, searching only the settings a
preset covers (step_ms, agreement_passes), with the tier's load limit
(`tune.max_load`, default 0.7 -- what `narrator --pick-preset` demands of
an operator's machine) and its processing unit (`tune.unit`, default: the
narrator config's asr.unit; `tune.device` is still read). The other accuracy settings come from the narrator config and
are not changed here (they are hardware-independent; `configure` tunes
them). A tier whose search finishes with a configuration inside the delay
cap and its load limit gets that row with `measured: true`; otherwise its
row is left as it was.

Limits: every tier is a full `configure` run (bench, calibration, offline
ceiling, screen/dev/holdout), each with the time budget of config/tune.yaml
-- hours per tier. The model of a tier is fixed by its row (choosing
between e.g. medium and large-v3-turbo for `balanced` means editing the
row and rerunning). Tier rows are measured on THIS machine; a tier meant
for weaker hardware is only approximated by its load limit/unit.
"""
from __future__ import annotations

import datetime as _dt
import logging
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from narrator.config.loader import load_config
from narrator.config.presets import PRESET_FIELDS, PRESETS_PATH, Preset, load_presets, render_presets_yaml

logger = logging.getLogger(__name__)

DEFAULT_TIER_MAX_LOAD = 0.7
# Of the covered fields, the ones the search varies (the model is the row's).
SEARCHED_FIELDS = ("step_ms", "agreement_passes")


@dataclass
class TierOutcome:
    name: str
    before: Preset
    after: Preset | None  # None: no measured row (reason says why)
    reason: str


def tier_tuning_overrides(preset: Preset, default_space: dict[str, list]) -> dict:
    """Tuning fields for one tier's search: covered settings only, the
    tier's own value grids where it has them, the tier's load limit."""
    custom = preset.tune.get("search_space") or {}
    space = {name: list(custom.get(name, default_space[name])) for name in SEARCHED_FIELDS}
    return {
        "search_space": space,
        "max_busy_fraction": float(preset.tune.get("max_load", DEFAULT_TIER_MAX_LOAD)),
        "candidate_models": [preset.model],
    }


def _scratch_config(base_config: Path, preset: Preset, directory: Path) -> Path:
    raw = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    asr = dict(raw.get("asr") or {})
    asr.update(preset.values())
    asr["preset"] = "custom"
    # The tier's unit (`tune.unit`; `tune.device` is the older spelling).
    unit = preset.tune.get("unit", preset.tune.get("device"))
    if unit is not None:
        from narrator.asr.units import unit_or_legacy

        for old in ("device", "gpu_device", "unit_identity"):
            asr.pop(old, None)
        asr["unit"] = unit_or_legacy(unit)
    raw["asr"] = asr
    path = directory / f"{preset.name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    load_config(path)  # validate
    return path


def _row_feasible(row: dict, max_latency_s: float | None, max_load: float) -> bool:
    if not row.get("excerpts"):
        return False
    if row.get("busy_fraction", 1.0) > max_load:
        return False
    latency = row.get("latency_p50")
    return max_latency_s is None or latency is None or latency <= max_latency_s


def tier_result(preset: Preset, outcome, max_latency_s: float | None, today: str) -> TierOutcome:
    """The measured row for a tier from its search Outcome, if any."""
    max_load = float(preset.tune.get("max_load", DEFAULT_TIER_MAX_LOAD))
    chosen = None
    if outcome.written and outcome.winner is not None:
        chosen, label = outcome.winner, "tuned"
    else:
        current = next(
            (r for r in outcome.table if r.get("label") == "current settings" and r.get("set") == "dev"), None
        )
        if current is not None and _row_feasible(current, max_latency_s, max_load):
            chosen, label = outcome.baseline, "current settings"
    if chosen is None:
        return TierOutcome(preset.name, preset, None, outcome.reason)
    dev = next((r for r in outcome.table if r.get("set") == "dev" and
                (r.get("label") == "current settings" if label == "current settings" else r.get("label", "").startswith("tuned"))),
               None)
    detail = ""
    if dev is not None:
        delay = "n/a" if dev.get("latency_p50") is None else f"{dev['latency_p50']:.1f}s"
        detail = f" dev WER {100 * dev['wer']:.1f}%, delay p50 {delay}, load {100 * dev['busy_fraction']:.0f}%"
    after = replace(
        preset,
        model=chosen.model,
        step_ms=int(chosen.step_ms),
        agreement_passes=int(chosen.agreement_passes),
        measured=True,
        note=f"Measured by narrator-tune presets on {today} ({label}, load limit {max_load:.2f}):{detail}.",
    )
    return TierOutcome(preset.name, preset, after, outcome.reason)


def tune_presets(
    narrator_config: Path,
    *,
    tiers: list[str] | None = None,
    output: Path = PRESETS_PATH,
    quick: bool = False,
    budget_minutes: float | None = None,
    dry_run: bool = False,
    allow_download: bool = False,
    tune_config=None,
    run_search=None,
) -> list[TierOutcome]:
    """Search every (selected) tier and rewrite `output` with the measured
    rows (unless dry_run). `run_search` defaults to autoconfigure()."""
    from narrator_tune.autoconfigure import DEFAULT_SEARCH_SPACE, Tuning, autoconfigure
    from narrator_tune.config import load_tune_config

    if run_search is None:
        run_search = autoconfigure
    if tune_config is None:
        tune_config = load_tune_config(narrator_config_path=narrator_config)
    max_latency_s = Tuning.from_config(tune_config).max_latency_s
    presets = load_presets(output)
    unknown = sorted(set(tiers or []) - set(presets))
    if unknown:
        raise ValueError(f"unknown tiers {unknown}; presets.yaml has {list(presets)}")
    today = _dt.date.today().isoformat()
    outcomes: list[TierOutcome] = []
    with tempfile.TemporaryDirectory(prefix="narrator-tune-presets-") as tmp:
        for name, preset in presets.items():
            if tiers and name not in tiers:
                continue
            logger.info("=== Tier %s: %s (load limit %.2f) ===", name, preset.model,
                        float(preset.tune.get("max_load", DEFAULT_TIER_MAX_LOAD)))
            scratch = _scratch_config(narrator_config, preset, Path(tmp))
            try:
                outcome = run_search(
                    scratch,
                    models=[preset.model],
                    quick=quick,
                    budget_minutes=budget_minutes,
                    dry_run=False,  # writes the scratch copy only; that is how the winner is read
                    allow_download=allow_download,
                    tune_config=tune_config,
                    tuning_overrides=tier_tuning_overrides(preset, DEFAULT_SEARCH_SPACE),
                )
            except RuntimeError as exc:
                outcomes.append(TierOutcome(name, preset, None, f"search stopped: {exc}"))
                continue
            outcomes.append(tier_result(preset, outcome, max_latency_s, today))

    updated = dict(presets)
    for result in outcomes:
        if result.after is not None:
            updated[result.name] = result.after
    if not dry_run and any(r.after is not None for r in outcomes):
        if output.exists():
            shutil.copyfile(output, output.with_name(output.name + ".bak"))
        tmp_path = output.with_suffix(output.suffix + ".tmp")
        tmp_path.write_text(render_presets_yaml(updated), encoding="utf-8")
        tmp_path.replace(output)
    return outcomes


def format_outcomes(outcomes: list[TierOutcome]) -> list[str]:
    lines = []
    for r in outcomes:
        b = r.before
        if r.after is None:
            lines.append(f"  {r.name:<10} unchanged ({b.model}, step {b.step_ms}, passes {b.agreement_passes}): {r.reason}")
        else:
            a = r.after
            moved = ", ".join(
                f"{f} {getattr(b, f)} -> {getattr(a, f)}" for f in PRESET_FIELDS if getattr(b, f) != getattr(a, f)
            ) or "values confirmed"
            lines.append(f"  {r.name:<10} measured: {moved}. {a.note}")
    return lines
