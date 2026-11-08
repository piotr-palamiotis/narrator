"""Problem items: what the setup checks report and the dashboard shows.

A Problem is plain, immutable data (id, severity, subsystem, message, fixes)
so the checks stay pure functions and the web layer can serialize them as
they are. Fix action ids are either handled by the server
(`AppController.apply_fix`: "asr.retry", "asr.use_cpu", "asr.pin_unit",
"asr.unit:<unit id>", "audio.retry", "ndi.retry", "health.recheck") or start
with "ui:" and only navigate the web
UI (e.g. "ui:asr" opens the ASR tab so the operator can pick another unit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"  # a part of the app is not working
    WARNING = "warning"  # it works, but something will go wrong or is degraded
    INFO = "info"  # worth knowing, nothing to fix right now


class Subsystem(str, Enum):
    ASR = "asr"
    AUDIO = "audio"
    NDI = "ndi"
    WEB = "web"
    RECORDING = "recording"
    SETUP = "setup"


# Server-side fix actions (AppController.apply_fix).
FIX_ASR_RETRY = "asr.retry"
FIX_ASR_USE_CPU = "asr.use_cpu"
# Record the identity (name + PCI bus id) of the device now at the
# configured asr.unit index in asr.unit_identity.
FIX_ASR_PIN_UNIT = "asr.pin_unit"
FIX_AUDIO_RETRY = "audio.retry"
FIX_NDI_RETRY = "ndi.retry"
FIX_RECHECK = "health.recheck"
SERVER_FIXES = frozenset(
    {FIX_ASR_RETRY, FIX_ASR_USE_CPU, FIX_ASR_PIN_UNIT, FIX_AUDIO_RETRY, FIX_NDI_RETRY, FIX_RECHECK}
)
# "asr.unit:<unit id>": switch asr.unit to that unit (and record its identity).
FIX_ASR_UNIT_PREFIX = "asr.unit:"


def is_server_fix(action: str) -> bool:
    return action in SERVER_FIXES or (action.startswith(FIX_ASR_UNIT_PREFIX) and len(action) > len(FIX_ASR_UNIT_PREFIX))

# Client-side fixes: open a settings tab.
FIX_UI_ASR = "ui:asr"
FIX_UI_AUDIO = "ui:audio"
FIX_UI_NDI = "ui:ndi"
FIX_UI_RECORDINGS = "ui:recordings"


@dataclass(frozen=True)
class Fix:
    action: str
    label: str


@dataclass(frozen=True)
class Problem:
    id: str  # stable, dotted ("asr.unit.no_vulkan_build"): the UI keys rows on it
    severity: Severity
    subsystem: Subsystem
    message: str
    fixes: tuple[Fix, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "subsystem": self.subsystem.value,
            "message": self.message,
            "fixes": [{"action": f.action, "label": f.label} for f in self.fixes],
        }


_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


def sort_problems(problems: list[Problem]) -> list[Problem]:
    """Errors first, then warnings, then info; the first item per id wins
    (runtime problems are listed before the periodic checks' results)."""
    seen: set[str] = set()
    unique = []
    for problem in problems:
        if problem.id not in seen:
            seen.add(problem.id)
            unique.append(problem)
    return sorted(unique, key=lambda p: _SEVERITY_ORDER[p.severity])


def use_unit_fix(unit_id: str, name: str) -> Fix:
    """The "switch to this unit" fix (offered for a moved card, the same
    card through another backend, or the recommended unit)."""
    return Fix(FIX_ASR_UNIT_PREFIX + unit_id, f"Use {unit_id} ({name})")


def offered_fixes(problems: list[Problem]) -> set[str]:
    return {fix.action for problem in problems for fix in problem.fixes}
