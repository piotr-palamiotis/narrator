"""Where narrator keeps what it writes: inside the repository checkout.

Layout under the repo root (all gitignored):

    config/config.yaml   the app config (passed with --config by the service)
    models/              whisper ggml models (downloads land here)
    var/                 runtime data: log/, tune/ (narrator-tune), web-ui.json
    third_party/         whisper.cpp / pywhispercpp source + build tree

The root is found by walking up from this file looking for the checkout
markers (pyproject.toml next to src/narrator), so it works for an editable
install (`pip install -e .`, what `make install` does) and not for a wheel in
site-packages or a PyInstaller build: there `repo_root()` is None and every
helper falls back to the platform user dirs (platformdirs), as before.
`NARRATOR_HOME` overrides the root explicitly (tests, odd deployments).

narrator_tune imports this too, so both agree on one repo root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "narrator"


def _find_repo_root() -> Path | None:
    override = os.environ.get("NARRATOR_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return None
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "narrator").is_dir():
            return parent
    return None


REPO_ROOT: Path | None = _find_repo_root()


def bundled_assets_dir() -> Path:
    """The shipped `assets/` folder: inside a PyInstaller bundle
    (`sys._MEIPASS`) or next to `src/` in a checkout. Independent of
    NARRATOR_HOME (that moves user data, not the shipped files)."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "assets"
    return Path(__file__).resolve().parents[2] / "assets"


def platform_config_path() -> Path:
    """The config file for a bare `narrator` run (no --config)."""
    return Path(user_config_dir(APP_NAME)) / "config.yaml"


def repo_config_path() -> Path | None:
    """<repo>/config/config.yaml, None outside a checkout."""
    return None if REPO_ROOT is None else REPO_ROOT / "config" / "config.yaml"


def var_dir() -> Path:
    """Runtime data root: <repo>/var, else the platform user data dir."""
    return Path(user_data_dir(APP_NAME)) if REPO_ROOT is None else REPO_ROOT / "var"


def log_dir() -> Path:
    return var_dir() / "log"


def reference_library_dir() -> Path | None:
    """<repo>/scripts/test_audio, the reference recordings narrator-tune
    learns from. None outside a checkout."""
    return None if REPO_ROOT is None else REPO_ROOT / "scripts" / "test_audio"


def third_party_dir() -> Path | None:
    return None if REPO_ROOT is None else REPO_ROOT / "third_party"


# --- whisper models --------------------------------------------------------


def models_dir() -> Path | None:
    """<repo>/models, None outside a checkout (pywhispercpp's own default)."""
    return None if REPO_ROOT is None else REPO_ROOT / "models"


def _legacy_models_dir() -> Path | None:
    """pywhispercpp's user-cache models dir, where models used to download."""
    try:
        from pywhispercpp.constants import MODELS_DIR
    except Exception:  # pragma: no cover - broken install
        return None
    return Path(MODELS_DIR)


def model_search_dirs() -> list[Path]:
    """Where a model is looked up, in order: the repo's models/, then the
    legacy user cache (so a multi-GB model downloaded before this layout
    existed is reused instead of fetched again)."""
    dirs = [d for d in (models_dir(), _legacy_models_dir()) if d is not None]
    return list(dict.fromkeys(dirs))


def _model_file_names(model: str) -> tuple[str, ...]:
    return (model, f"{model}.bin", f"ggml-{model}.bin")


def find_model_file(model: str, search_dirs: list[Path] | None = None) -> Path | None:
    """The local file `model` resolves to (a direct path, or a name found in
    the search dirs), None when it would have to be downloaded."""
    direct = Path(model).expanduser()
    if direct.is_file():
        return direct.resolve()
    for directory in model_search_dirs() if search_dirs is None else search_dirs:
        for name in _model_file_names(model):
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def model_load_args(model: str) -> tuple[str, str | None]:
    """(model, models_dir) to pass to pywhispercpp's Model(). A model found
    locally is passed as its full path (so no directory is consulted); an
    unknown name is passed through with models_dir set to <repo>/models, which
    is where pywhispercpp then downloads it. Uses Model's documented
    `models_dir` parameter, no monkeypatching of pywhispercpp.constants."""
    found = find_model_file(model)
    if found is not None:
        return str(found), None
    target = models_dir()
    return model, None if target is None else str(target)


def cached_model_names() -> list[str]:
    """Names (ggml- prefix stripped) of every model file in the search dirs."""
    names: set[str] = set()
    for directory in model_search_dirs():
        names.update(p.stem.removeprefix("ggml-") for p in directory.glob("*.bin"))
    return sorted(names)
