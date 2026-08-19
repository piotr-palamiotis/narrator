"""Load, validate, save, and hot-reload the narrator YAML config.

The YAML file is the single source of truth. Whether it was edited by hand
or written by the GUI's Save button, changes flow through the same path:
file write -> watchdog detects -> validate -> atomic in-memory swap. This
keeps headless and GUI-driven operation behaviorally identical.
"""
from __future__ import annotations

import logging
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


def save_config(config: RootConfig, path: Path) -> None:
    """Serialize a config to YAML, writing atomically (write to temp file, then rename)."""
    data = config.model_dump(mode="json")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    tmp_path.replace(path)


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
