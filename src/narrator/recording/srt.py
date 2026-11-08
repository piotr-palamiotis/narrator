"""SRT parsing/validation for uploaded corrected subtitles, and writing.

An upload is validated strictly enough that what lands as corrected.srt is
a clean reference narrator-tune can score against: UTF-8, well-formed
timing lines, end >= start, cues in start order, at least one cue with text,
and nothing timed past the recording's end (a file for another recording
is the likeliest mistake). The stored file is re-serialized (renumbered,
LF line endings).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from narrator.captions.transcript import format_srt_time

MAX_SRT_BYTES = 4 * 1024 * 1024
# Cues may run a little past the audio (players round, editors pad).
_END_TOLERANCE_S = 5.0

_TIMING_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})(?:\s+.*)?$"
)


class SrtError(ValueError):
    """An uploaded SRT is not usable; the message says where and why."""


@dataclass(frozen=True)
class SrtCue:
    start: float
    end: float
    text: str


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    minutes, secs = int(m), int(s)
    if minutes >= 60 or secs >= 60:
        raise ValueError("minutes/seconds out of range")
    return int(h) * 3600 + minutes * 60 + secs + int(ms.ljust(3, "0")) / 1000.0


def parse_srt_text(data: bytes | str, *, duration_s: float | None = None) -> list[SrtCue]:
    """Parse and validate SRT content. Raises SrtError with a line number."""
    if isinstance(data, bytes):
        if len(data) > MAX_SRT_BYTES:
            raise SrtError(f"file is too large ({len(data) // 1024} KiB; at most {MAX_SRT_BYTES // 1024} KiB)")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SrtError(f"not UTF-8 text (byte {exc.start}); save the subtitles as UTF-8") from exc
    else:
        text = data.removeprefix("﻿")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    cues: list[SrtCue] = []
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        block_line = i + 1
        header = lines[i].strip()
        timing_line = header
        if not _TIMING_RE.match(header):
            # The usual shape: a cue number, then the timing line.
            if not header.isdigit():
                raise SrtError(f"line {block_line}: expected a cue number or a timing line, got {header[:60]!r}")
            i += 1
            if i >= n:
                raise SrtError(f"line {block_line}: cue {header} has no timing line")
            timing_line = lines[i].strip()
        match = _TIMING_RE.match(timing_line)
        if match is None:
            raise SrtError(f"line {i + 1}: expected 'HH:MM:SS,mmm --> HH:MM:SS,mmm', got {timing_line[:60]!r}")
        try:
            start = _seconds(*match.group(1, 2, 3, 4))
            end = _seconds(*match.group(5, 6, 7, 8))
        except ValueError as exc:
            raise SrtError(f"line {i + 1}: invalid time ({exc})") from exc
        if end < start:
            raise SrtError(f"line {i + 1}: the cue ends before it starts")
        if cues and start < cues[-1].start:
            raise SrtError(f"line {i + 1}: cues are out of order ({format_srt_time(start)} after {format_srt_time(cues[-1].start)})")
        if duration_s is not None and start > duration_s + _END_TOLERANCE_S:
            raise SrtError(
                f"line {i + 1}: cue at {format_srt_time(start)} is past the end of the recording "
                f"({format_srt_time(duration_s)}) -- is this the SRT of another recording?"
            )
        i += 1
        body: list[str] = []
        while i < n and lines[i].strip():
            body.append(lines[i].strip())
            i += 1
        cues.append(SrtCue(start, end, "\n".join(body)))
    if not any(cue.text.strip() for cue in cues):
        raise SrtError("no subtitle cues with text found")
    return cues


def format_srt(cues: list[SrtCue] | list[tuple[float, float, str]]) -> str:
    out = []
    for number, cue in enumerate(cues, start=1):
        start, end, text = (cue.start, cue.end, cue.text) if isinstance(cue, SrtCue) else cue
        out.append(f"{number}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n")
    return "\n".join(out)
