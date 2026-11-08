"""Recent log lines for the web UI's status view: a logging handler that keeps
the last N formatted records in memory, each with a sequence number so a
client can ask for "everything after N"."""
from __future__ import annotations

import collections
import logging
import threading

_FORMAT = "%(name)s: %(message)s"


class RingBufferHandler(logging.Handler):
    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self._records: collections.deque = collections.deque(maxlen=capacity)
        self._seq = 0
        self._lock_ring = threading.Lock()
        self.setFormatter(logging.Formatter(_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # never let logging break the caller
            return
        with self._lock_ring:
            self._seq += 1
            self._records.append(
                {"seq": self._seq, "time": record.created, "level": record.levelname, "message": message}
            )

    def since(self, seq: int = 0, limit: int = 500) -> list[dict]:
        with self._lock_ring:
            lines = [r for r in self._records if r["seq"] > seq]
        return lines[-limit:]

    @property
    def last_seq(self) -> int:
        return self._seq


_ring: RingBufferHandler | None = None


def install_log_ring(capacity: int = 500) -> RingBufferHandler:
    """Attach the ring buffer to the root logger (once) and return it.

    Also installs the stderr handler that logging.basicConfig would have:
    AppController.start's basicConfig call is a no-op once the root logger has
    any handler, which would otherwise silence terminal/journald output (the
    same rule as logsetup.enable_file_logging)."""
    global _ring
    root = logging.getLogger()
    if _ring is None:
        if not any(type(h) is logging.StreamHandler for h in root.handlers):
            stream = logging.StreamHandler()
            stream.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            root.addHandler(stream)
        if root.level == logging.WARNING:  # untouched default; caller sets the configured level
            root.setLevel(logging.INFO)
        _ring = RingBufferHandler(capacity)
        root.addHandler(_ring)
    return _ring


def log_ring() -> RingBufferHandler | None:
    return _ring

