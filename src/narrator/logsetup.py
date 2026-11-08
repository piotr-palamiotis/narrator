"""Optional rotating file log (`narrator --log-file`), under <repo>/var/log/.

Off by default: the systemd unit logs to journald via stderr, and a bare run
logs to the terminal. With the flag, the same records also go to
var/log/narrator.log (5 x 2 MB). The ASR worker is a separate spawned process
that cannot share a RotatingFileHandler safely, so it writes its own
narrator-asr.log, enabled through the NARRATOR_LOG_DIR environment variable
(inherited by the spawn).
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

from narrator.paths import log_dir

LOG_DIR_ENV = "NARRATOR_LOG_DIR"
_MAX_BYTES = 2_000_000
_BACKUPS = 5
_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _add_file_handler(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    logging.getLogger().addHandler(handler)
    return path


def enable_file_logging(directory: Path | None = None) -> Path:
    """Mirror the root logger into <directory>/narrator.log and make the ASR
    worker log into narrator-asr.log next to it.

    Also installs the stderr handler that logging.basicConfig would have (the
    controller's own basicConfig call is a no-op once the root logger has any
    handler, which would otherwise silence journald/terminal output), and
    sets the root level to INFO; the caller applies the configured level once
    the config is loaded.
    """
    directory = directory or log_dir()
    root = logging.getLogger()
    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        root.addHandler(stream)
    root.setLevel(logging.INFO)
    path = _add_file_handler(directory, "narrator.log")
    os.environ[LOG_DIR_ENV] = str(directory)
    return path


def attach_worker_file_log() -> None:
    """In the ASR worker process: log to narrator-asr.log when the parent
    enabled file logging."""
    directory = os.environ.get(LOG_DIR_ENV)
    if directory:
        try:
            _add_file_handler(Path(directory), "narrator-asr.log")
        except OSError:
            logging.getLogger(__name__).exception("cannot open the ASR worker log file")
