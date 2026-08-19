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
        self._committed_lines: list[str] = []
        self._pending_text: str = ""
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
        parts = list(self._committed_lines)
        if self._pending_text:
            parts.append(self._pending_text)
        full_text = " ".join(parts).strip()
        self._committed_lines = []
        self._pending_text = ""
        if full_text:
            self._ingest(full_text)

    def append_final_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._ingest(text)

    def _ingest(self, new_text: str) -> None:
        if self._measure is None or self._font_path is None or self._size_px is None:
            raise RuntimeError("CaptionBuffer.configure_geometry() must be called before ingesting text")
        combined = f"{self._pending_text} {new_text}".strip() if self._pending_text else new_text
        wrapped = wrap_text(combined, self._font_path, self._size_px, self._box_width, self._measure)
        if not wrapped:
            self._pending_text = ""
            return
        *complete, last = wrapped
        self._committed_lines.extend(complete)
        self._pending_text = last
        if len(self._committed_lines) > self._history_cap:
            self._committed_lines = self._committed_lines[-self._history_cap :]

    def clear(self) -> None:
        self._committed_lines = []
        self._pending_text = ""

    def snapshot_text(self) -> str:
        """Best-effort read-only join of all retained text (committed +
        pending), for mirroring into another buffer under different
        geometry (e.g. a GUI preview). Not synchronized against concurrent
        writers -- same caveat as visible_lines()."""
        parts = list(self._committed_lines)
        if self._pending_text:
            parts.append(self._pending_text)
        return " ".join(parts).strip()

    def visible_lines(self) -> list[str]:
        lines = list(self._committed_lines)
        if self._pending_text:
            lines.append(self._pending_text)
        return lines[-self._max_lines :]
