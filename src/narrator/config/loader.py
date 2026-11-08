"""Load, validate, save, and hot-reload the narrator YAML config.

The YAML file is the single source of truth. Whether it was edited by hand
or written by the GUI's Save button, changes flow through the same path:
file write -> watchdog detects -> validate -> atomic in-memory swap. This
keeps headless and GUI-driven operation behaviorally identical.
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import ValidationError
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from narrator.config.schema import RootConfig

logger = logging.getLogger(__name__)

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")


def load_config(path: Path) -> RootConfig:
    """Load and validate a config file. Raises ValidationError/OSError on failure."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RootConfig.model_validate(raw)


def config_to_yaml_data(config: RootConfig) -> dict:
    """The dict save_config writes. With a named asr.preset, covered fields
    whose value equals the preset's are left out, so the saved file keeps
    following the preset instead of freezing its current values into
    explicit overrides (see narrator.config.presets)."""
    from narrator.config.presets import preset_supplied_fields

    data = config.model_dump(mode="json")
    for name in preset_supplied_fields(config.asr):
        data["asr"].pop(name, None)
    return data


def _write_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def save_config(config: RootConfig, path: Path) -> None:
    """Serialize a config to YAML, writing atomically (write to temp file, then rename)."""
    data = config_to_yaml_data(config)
    _write_atomic(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def backup_config(path: Path) -> Path:
    """Copy `path` to `<name>.bak` next to it (the previous file, before a
    tool rewrites it); returns the backup path."""
    backup = path.with_name(path.name + ".bak")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def _edit_asr_preset_text(text: str, name: str) -> str | None:
    """`text` with `asr.preset` set to `name` by a line edit (comments and
    layout kept), or None when the file's shape is not the plain block style
    this handles."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if re.match(r"^asr:\s*(#.*)?$", ln.rstrip("\n"))), None)
    if start is None:
        sep = "" if not text or text.endswith("\n") else "\n"
        return f"{text}{sep}asr:\n  preset: {name}\n"
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#") and not lines[i][0].isspace():
            end = i
            break
    keys = [i for i in range(start + 1, end) if lines[i].strip() and not lines[i].strip().startswith("#")]
    indent = re.match(r"^(\s*)", lines[keys[0]]).group(1) if keys else "  "
    for i in keys:
        if re.match(rf"^{re.escape(indent)}preset:(\s|$)", lines[i]):
            lines[i] = f"{indent}preset: {name}\n"
            return "".join(lines)
    lines.insert(start + 1, f"{indent}preset: {name}\n")
    return "".join(lines)


def write_asr_preset(path: Path, name: str) -> Path:
    """Set only `asr.preset` in the config file at `path` (validated first;
    previous file kept as config.yaml.bak; written atomically, so a running
    app hot-reloads it). Comments are kept when the file is plain block
    YAML; otherwise the parsed file is re-dumped with only that key changed.
    Returns the backup path."""
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    expected = dict(raw)
    expected["asr"] = {**(raw.get("asr") or {}), "preset": name}
    RootConfig.model_validate(expected)  # never write a config that won't load
    new_text = _edit_asr_preset_text(text, name)
    try:
        ok = new_text is not None and yaml.safe_load(new_text) == expected
    except yaml.YAMLError:
        ok = False
    if not ok:
        new_text = yaml.safe_dump(expected, sort_keys=False, allow_unicode=True)
    backup = backup_config(path)
    _write_atomic(path, new_text)
    return backup


def ensure_config_exists(path: Path) -> None:
    """Copy the bundled defaults to `path` if no config file exists there yet."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULTS_PATH.read_text(encoding="utf-8"), encoding="utf-8")


class ConfigWatcher:
    """Watches a config file for changes and hands validated RootConfig snapshots
    to `on_change`. Invalid edits are logged and ignored (the last-good config
    stays in effect) rather than crashing the running app.
    """

    def __init__(self, path: Path, on_change: Callable[[RootConfig], None]) -> None:
        self._path = path.resolve()
        self._on_change = on_change
        self._lock = threading.Lock()
        self._observer: Observer | None = None

    def start(self) -> RootConfig:
        ensure_config_exists(self._path)
        initial = load_config(self._path)

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if Path(event.src_path).resolve() == watcher._path:
                    watcher._reload()

            def on_moved(self, event):
                if Path(event.dest_path).resolve() == watcher._path:
                    watcher._reload()

        self._observer = Observer()
        self._observer.schedule(_Handler(), str(self._path.parent), recursive=False)
        self._observer.start()
        return initial

    def _reload(self) -> None:
        with self._lock:
            try:
                config = load_config(self._path)
            except (ValidationError, OSError, yaml.YAMLError) as exc:
                logger.warning("Ignoring invalid config reload at %s: %s", self._path, exc)
                return
            self._on_change(config)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
