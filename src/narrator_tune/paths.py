"""Where narrator-tune reads its inputs and keeps its working data.

Everything is anchored to the repository (this is a developer/operator tool
run from a checkout): reference recordings in scripts/test_audio/, the
tool's own settings in config/tune.yaml, and its cache and reports under
var/tune/.

Cache and reports used to live in scripts/autoconfigure_cache/ and
scripts/autoconfigure_reports/. Until the new directory exists, an existing
old one keeps being used (a 12-hour search's cached decodes are worth
keeping); move it to adopt the new layout:

    mkdir -p var/tune
    mv scripts/autoconfigure_cache var/tune/cache
    mv scripts/autoconfigure_reports var/tune/reports
"""
from __future__ import annotations

from pathlib import Path

from narrator.paths import REPO_ROOT as _APP_REPO_ROOT
from narrator.paths import var_dir as _app_var_dir

# One repo-root detection for app and tool (narrator.paths); the fallback is
# for a tool run from a non-editable install, where it still means "this
# checkout" by its own location.
REPO_ROOT = _APP_REPO_ROOT or Path(__file__).resolve().parents[2]
REAL_FIXTURES_DIR = REPO_ROOT / "scripts" / "test_audio"
CLIPS_DIR = REPO_ROOT / "scripts" / "test_clips"
TUNE_CONFIG_PATH = REPO_ROOT / "config" / "tune.yaml"

VAR_DIR = (_app_var_dir() if _APP_REPO_ROOT else REPO_ROOT / "var") / "tune"
LEGACY_CACHE_DIR = REPO_ROOT / "scripts" / "autoconfigure_cache"
LEGACY_REPORTS_DIR = REPO_ROOT / "scripts" / "autoconfigure_reports"


def _with_legacy_fallback(new: Path, legacy: Path) -> Path:
    """`new`, unless only the pre-narrator-tune location exists yet."""
    if not new.exists() and legacy.exists():
        return legacy
    return new


CACHE_DIR = _with_legacy_fallback(VAR_DIR / "cache", LEGACY_CACHE_DIR)
REPORTS_DIR = _with_legacy_fallback(VAR_DIR / "reports", LEGACY_REPORTS_DIR)


def legacy_dirs_in_use() -> list[tuple[Path, Path]]:
    """(legacy dir in use, where it should move) pairs, for a notice."""
    pairs = ((CACHE_DIR, VAR_DIR / "cache"), (REPORTS_DIR, VAR_DIR / "reports"))
    return [(used, wanted) for used, wanted in pairs if used != wanted]
