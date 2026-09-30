"""Rolling N-line caption display buffer.

Finalized transcript text streams in via `append_final_text`. Internally the
buffer keeps a small "pending" tail (the most recent wrapped line, which may
still be extended by the next chunk of text) plus a bounded history of fully
wrapped, completed lines. `visible_lines()` always returns the last
`max_lines` entries -- classic scroll-up captioning behavior.

Re-wrapping only happens incrementally on each `append_final_text` call
(cheap: operates on the pending tail + new text) except when the render
geometry itself changes (font size or box width), which triggers a full
rewrap of the retained history so on-screen text reflows correctly.
"""
from __future__ import annotations

from narrator.captions.wrap import TextWidthProvider, wrap_text

DEFAULT_HISTORY_CAP = 500


class CaptionBuffer:
    def __init__(self, max_lines: int, history_cap: int = DEFAULT_HISTORY_CAP) -> None:
        self._max_lines = max_lines
        self._history_cap = max(history_cap, max_lines)
        # Committed lines and pending tail are kept as a single tuple and always
        # replaced with one attribute assignment (atomic under the GIL), so
        # concurrent readers on other threads (NDI render loop, GUI preview poll)
        # never observe a transiently empty buffer mid-rewrap.
        self._state: tuple[list[str], str] = ([], "")
        self._font_path: str | None = None
        self._size_px: int | None = None
        self._box_width: float | None = None
        self._measure: TextWidthProvider | None = None

    def set_max_lines(self, max_lines: int) -> None:
        self._max_lines = max_lines
        self._history_cap = max(self._history_cap, max_lines)

    def configure_geometry(
        self, font_path: str, size_px: int, box_width: float, measure: TextWidthProvider
    ) -> None:
        """Update the wrap parameters. Triggers a full rewrap of retained text
        if font size or box width actually changed (a layout config change),
        so already-displayed captions reflow to match.
        """
        changed = (font_path, size_px, box_width) != (
            self._font_path,
            self._size_px,
            self._box_width,
        )
        self._font_path, self._size_px, self._box_width, self._measure = (
            font_path,
            size_px,
            box_width,
            measure,
        )
        if changed:
            self._rewrap_all()

    def _rewrap_all(self) -> None:
        committed, pending = self._state
        parts = list(committed)
        if pending:
            parts.append(pending)
        full_text = " ".join(parts).strip()
        self._state = self._wrapped_state([], "", full_text) if full_text else ([], "")

    def append_final_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._ingest(text)

    def _ingest(self, new_text: str) -> None:
        committed, pending = self._state
        self._state = self._wrapped_state(committed, pending, new_text)

    def _wrapped_state(
        self, base_committed: list[str], base_pending: str, new_text: str
    ) -> tuple[list[str], str]:
        if self._measure is None or self._font_path is None or self._size_px is None:
            raise RuntimeError("CaptionBuffer.configure_geometry() must be called before ingesting text")
        combined = f"{base_pending} {new_text}".strip() if base_pending else new_text
        wrapped = wrap_text(combined, self._font_path, self._size_px, self._box_width, self._measure)
        if not wrapped:
            return (base_committed, "")
        *complete, last = wrapped
        new_committed = base_committed + complete
        if len(new_committed) > self._history_cap:
            new_committed = new_committed[-self._history_cap :]
        return (new_committed, last)

    def prune_history(self) -> int:
        """Drop committed lines that are no longer displayable, keeping what is
        currently on screen exactly as it is.

        `visible_lines()` shows the last `max_lines` entries, so any committed
        line beyond that count is already invisible under the current layout
        settings and can be removed without disturbing the display. The pending
        tail is never touched -- it is part of the live caption. Returns the
        number of committed lines dropped (0 when there was nothing to prune).

        This is what the ASR worker's silence-timeout CLEAR event routes to, so
        the periodic screen-clear after `clear_after_silence_seconds` only
        trims memory, never the visible lower-third. A true screen wipe
        (Ctrl+N, the GUI's Clear button) still calls `clear()`.
        """
        committed, pending = self._state
        # The pending tail occupies a display slot of its own, so fewer
        # committed lines are visible while it exists.
        visible_committed = max(self._max_lines - (1 if pending else 0), 0)
        excess = len(committed) - visible_committed
        if excess <= 0:
            return 0
        self._state = (committed[excess:], pending)
        return excess

    def clear(self) -> None:
        self._state = ([], "")

    def snapshot_text(self) -> str:
        """Best-effort read-only join of all retained text (committed +
        pending), for mirroring into another buffer under different
        geometry (e.g. a GUI preview)."""
        committed, pending = self._state
        parts = list(committed)
        if pending:
            parts.append(pending)
        return " ".join(parts).strip()

    def visible_lines(self) -> list[str]:
        committed, pending = self._state
        lines = list(committed)
        if pending:
            lines.append(pending)
        return lines[-self._max_lines :]
