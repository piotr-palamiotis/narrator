"""ASR performance presets (`asr.preset`), shipped in presets.yaml.

A preset names a hardware tier and supplies only the compute settings
(PRESET_FIELDS: model, step_ms, agreement_passes). Everything else -- the
processing unit (asr.unit), the language whitelist, the accuracy settings --
is never part of a preset.

Resolution happens in AsrConfig's `model_validator(mode="before")`, on the
raw dict from config.yaml: a covered field the file does not set is filled
from the preset; a field the file does set always wins. The rest of the app
only ever sees plain values.

Saving keeps the preset semantics (config.loader.save_config): a covered
field whose value equals the preset's is written as *absent*, not frozen
into an explicit value. That rule is by value, not by provenance, on
purpose: the GUI/web Save path rebuilds AsrConfig from a dump of the live
one plus the form (model, step_ms, ... always present), so "came from the
preset" cannot survive that round trip -- "equals the preset" does. The one
consequence: an explicit value that happens to equal the preset's is
treated as following the preset after the next save.

Tiers are listed most accurate first; `narrator --pick-preset`
(narrator.asr.preset_picker) picks the first one this machine can carry.
Rows are produced by `narrator-tune presets` (measured: true) and are
provisional until then (measured: false).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PRESETS_PATH = Path(__file__).with_name("presets.yaml")
CUSTOM = "custom"
# The only settings a preset covers (todo A5): the compute/latency knobs.
PRESET_FIELDS = ("model", "step_ms", "agreement_passes")


@dataclass(frozen=True)
class Preset:
    name: str
    model: str
    step_ms: int
    agreement_passes: int
    measured: bool = False
    note: str = ""
    # How `narrator-tune presets` searches this tier: unit, max_load,
    # search_space ({setting: [values]}). Not used at runtime.
    tune: dict = field(default_factory=dict)

    def values(self) -> dict:
        return {name: getattr(self, name) for name in PRESET_FIELDS}

    def to_row(self) -> dict:
        row: dict = {"measured": self.measured, **self.values()}
        if self.note:
            row["note"] = self.note
        if self.tune:
            row["tune"] = self.tune
        return row


def parse_presets(raw: dict) -> dict[str, Preset]:
    """{name: Preset} in file order (= tier order, most accurate first)."""
    tiers = (raw or {}).get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError("presets.yaml: expected a non-empty `tiers:` mapping")
    presets: dict[str, Preset] = {}
    for name, row in tiers.items():
        if name == CUSTOM:
            raise ValueError(f"presets.yaml: {CUSTOM!r} is reserved (it means: no preset)")
        if not isinstance(row, dict):
            raise ValueError(f"presets.yaml: tier {name!r} must be a mapping")
        missing = [f for f in PRESET_FIELDS if f not in row]
        unknown = sorted(set(row) - set(PRESET_FIELDS) - {"measured", "note", "tune"})
        if missing or unknown:
            raise ValueError(f"presets.yaml: tier {name!r} missing {missing} / unknown keys {unknown}")
        presets[str(name)] = Preset(
            name=str(name),
            model=str(row["model"]),
            step_ms=int(row["step_ms"]),
            agreement_passes=int(row["agreement_passes"]),
            measured=bool(row.get("measured", False)),
            note=str(row.get("note", "")),
            tune=dict(row.get("tune") or {}),
        )
    return presets


def load_presets(path: Path | None = None) -> dict[str, Preset]:
    if path is None:
        return dict(_load_default_presets())
    with Path(path).open("r", encoding="utf-8") as f:
        return parse_presets(yaml.safe_load(f))


@lru_cache(maxsize=1)
def _load_default_presets() -> tuple[tuple[str, Preset], ...]:
    # Cached: every AsrConfig validation with a named preset reads it.
    return tuple(load_presets(PRESETS_PATH).items())


def preset_names() -> list[str]:
    return list(load_presets())


def get_preset(name: str) -> Preset:
    presets = load_presets()
    if name not in presets:
        raise ValueError(f"unknown asr.preset {name!r}; known: {list(presets) + [CUSTOM]}")
    return presets[name]


def apply_preset(data: dict) -> dict:
    """The raw asr dict with the named preset's values filled in for every
    covered field it does not set itself (explicit values win)."""
    name = data.get("preset", CUSTOM)
    if name in (None, CUSTOM):
        return data
    preset = get_preset(str(name))
    return {**preset.values(), **data}


def preset_supplied_fields(asr) -> set[str]:
    """Covered fields of an AsrConfig that follow its preset (value equals
    the preset's). These are left out of a saved config.yaml."""
    if asr.preset == CUSTOM:
        return set()
    values = get_preset(asr.preset).values()
    return {name for name, value in values.items() if getattr(asr, name) == value}


def explicit_overrides(raw_asr: dict) -> dict:
    """{field: value} for covered fields that config.yaml sets explicitly to
    something other than its preset's value (they override the preset)."""
    name = raw_asr.get("preset", CUSTOM)
    if name in (None, CUSTOM):
        return {}
    values = get_preset(str(name)).values()
    return {f: raw_asr[f] for f in PRESET_FIELDS if f in raw_asr and raw_asr[f] != values[f]}


_HEADER = """\
# ASR performance presets (asr.preset in config.yaml); see
# narrator.config.presets. Tiers are listed MOST ACCURATE FIRST:
# `narrator --pick-preset` benchmarks this machine and picks the first tier
# it can carry (worker load <= 0.7, caption delay p50 <= 3 s).
#
# A preset covers only model, step_ms and agreement_passes. An explicit
# value in config.yaml always wins over the preset.
#
# measured: false = provisional values, never searched by
# `narrator-tune presets` (make tune-presets), which rewrites this file
# with measured: true rows. `tune:` tells that search how to tune the tier.
"""


def render_presets_yaml(presets: dict[str, Preset]) -> str:
    body = yaml.safe_dump(
        {"tiers": {name: p.to_row() for name, p in presets.items()}},
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    return _HEADER + "\n" + body
