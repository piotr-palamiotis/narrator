"""CaptionEvent: the message type carried from the ASR worker process to the
main process's caption buffer, over a multiprocessing.Queue.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class CaptionEventType(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"
    CLEAR = "clear"  # prune off-screen caption history (display untouched)
    WIPE = "wipe"  # true screen clear after a long silence lull (operator-equivalent)
    STATUS = "status"  # informational, e.g. resolved ASR backend/device string
    ERROR = "error"  # the worker is dying: why (shown by the setup checks' problem panel)


@dataclass(frozen=True)
class CaptionEvent:
    type: CaptionEventType
    text: str = ""
    timestamp: float = field(default_factory=time.monotonic)
    # FINAL only: when the captioned words were spoken, on the decoder's
    # clock (time.monotonic in the app, so comparable across the worker and
    # main processes). An estimate -- words are placed proportionally within
    # their decode window -- but free of the caption delay that `timestamp`
    # (emit time) carries. Used for transcript cue timing.
    start: float | None = None
    end: float | None = None
