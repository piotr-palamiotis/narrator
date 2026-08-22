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
