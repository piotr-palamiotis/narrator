"""Transcript file of everything the ASR finalized (obs-localvocal's "save
to file"), written as SRT cues or plain text lines.

FINALs are commit-sized fragments (a few words each), so they are grouped
into readable cues: a cue is written out when its text ends a sentence and
has some length, when it grows past one subtitle's worth of text, when the
speaker pauses, or when the writer closes. Cue times come from the FINALs'
spoken-time estimates (CaptionEvent.start/end), relative to when the writer
was opened, falling back to the emit time.

Every write appends and flushes, so a crash loses at most the cue being
gathered. A write failure disables the transcript (logged once) and never
touches the caption path.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from narrator.asr.events import CaptionEvent
from narrator.config.schema import TranscriptConfig, TranscriptFormat
from narrator.paths import transcripts_dir

logger = logging.getLogger(__name__)

_MAX_CUE_CHARS = 84  # two 42-character subtitle lines
_MIN_SENTENCE_CUE_CHARS = 20  # a sentence end shorter than this joins the next one
_PAUSE_SPLIT_SECONDS = 1.5
_MIN_CUE_SECONDS = 0.5
_SENTENCE_END = (".", "?", "!", "…")


def default_transcript_directory() -> Path:
    return transcripts_dir()


def format_srt_time(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class TranscriptWriter:
    def __init__(
        self,
        config: TranscriptConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
        path: Path | None = None,
        origin: float | None = None,
    ) -> None:
        """`path` overrides the per-session file name (a recording session
        writes its live.srt here); `origin` is the clock value cue time 0
        corresponds to (default: now)."""
        self._format = config.format
        if path is None:
            directory = Path(config.directory).expanduser() if config.directory else default_transcript_directory()
            stamp = now().strftime("%Y-%m-%d_%H-%M-%S")
            path = directory / f"narrator-{stamp}.{config.format.value}"
        self.path = path
        self._origin = clock() if origin is None else origin
        self._cue_count = 0
        self._last_end = 0.0
        self._pending: list[str] = []
        self._pending_start = 0.0
        self._pending_end = 0.0
        self._failed = False

    def add_final(self, event: CaptionEvent) -> None:
        text = event.text.strip()
        if self._failed or not text:
            return
        end = (event.end if event.end is not None else event.timestamp) - self._origin
        start = event.start - self._origin if event.start is not None else end
        if self._pending and start - self._pending_end > _PAUSE_SPLIT_SECONDS:
            self._flush()
        if not self._pending:
            self._pending_start = start
        self._pending.append(text)
        self._pending_end = max(end, self._pending_end)
        joined = " ".join(self._pending)
        if len(joined) >= _MAX_CUE_CHARS or (
            joined.endswith(_SENTENCE_END) and len(joined) >= _MIN_SENTENCE_CUE_CHARS
        ):
            self._flush()

    def close(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        text = " ".join(self._pending)
        # SRT players need ordered, non-overlapping cues of visible length.
        start = max(self._pending_start, self._last_end)
        end = max(self._pending_end, start + _MIN_CUE_SECONDS)
        self._pending = []
        self._cue_count += 1
        self._last_end = end
        if self._format == TranscriptFormat.SRT:
            entry = f"{self._cue_count}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n"
        else:
            entry = text + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(entry)
        except OSError:
            logger.exception("transcript write to %s failed; transcript disabled", self.path)
            self._failed = True
