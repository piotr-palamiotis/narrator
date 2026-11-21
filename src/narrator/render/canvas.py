"""RGBA canvas compositor: ties layout geometry, the rolling caption buffer,
and style overrides together, producing the frame sent to NDI.

Recompositing is skipped when neither the visible caption lines nor the
layout/style config have changed since the last call -- the render+NDI loop
still sends a frame every tick (see narrator.ndi.sender), it just resends the
cached bytes instead of redoing the (relatively expensive) Pillow drawing.

Caption state changes (a FINAL replacing the open partial, pacing reveals)
and frame reads happen under one lock, so the render thread never sees a
half-applied update -- e.g. neither the partial nor the FINAL that
replaces it, or both. With `layout.pacing` on, text passes through a
CaptionPacer (narrator.captions.pacer) on its way to the buffer.
"""
from __future__ import annotations

import threading
import time

from PIL import Image

from narrator.captions.buffer import CaptionBuffer
from narrator.captions.layout import compute_line_geometry, line_y_positions
from narrator.captions.pacer import CaptionPacer
from narrator.config.schema import CanvasConfig, LayoutConfig, StyleConfig, parse_rgba
from narrator.render.fonts import PillowFontMetrics, PillowTextWidth, load_font
from narrator.render.style import composite_line

_metrics = PillowFontMetrics()
_text_width = PillowTextWidth()


class CaptionRenderer:
    def __init__(self, canvas: CanvasConfig, layout: LayoutConfig, style: StyleConfig) -> None:
        self._buffer = CaptionBuffer(max_lines=layout.max_lines)
        self._canvas_cfg = canvas
        self._layout_cfg = layout
        self._style_cfg = style
        self._geometry = None
        self._cached_frame: Image.Image | None = None
        self._cached_lines: tuple[str, ...] | None = None
        self._partial_preview: str = ""
        self._lock = threading.RLock()
        self._clock = time.monotonic
        self._pacer: CaptionPacer | None = None
        self._apply_pacing()
        self._apply_geometry()

    def _apply_pacing(self) -> None:
        """Start/stop/switch the pacer for `layout.pacing`; text it still
        holds is shown at once when pacing goes off or changes mode."""
        mode = getattr(self._layout_cfg.pacing, "value", self._layout_cfg.pacing)
        if self._pacer is not None and self._pacer.mode == mode:
            return
        if self._pacer is not None:
            commits, tail = self._pacer.flush()
            for text in commits:
                self._buffer.append_final_text(text)
            self._partial_preview = tail
        self._pacer = CaptionPacer(mode) if mode != "off" else None

    def _apply_geometry(self) -> None:
        self._geometry = compute_line_geometry(self._canvas_cfg, self._layout_cfg, self._style_cfg, _metrics)
        self._buffer.set_max_lines(self._layout_cfg.max_lines)
        self._buffer.configure_geometry(
            self._style_cfg.font_family,
            self._geometry.font_size_px,
            self._geometry.box.width,
            _text_width,
        )
        self._cached_frame = None  # force recomposite: geometry/style changed
        self._cached_lines = None

    def update_config(self, canvas: CanvasConfig, layout: LayoutConfig, style: StyleConfig) -> None:
        with self._lock:
            self._canvas_cfg, self._layout_cfg, self._style_cfg = canvas, layout, style
            self._apply_pacing()
            self._apply_geometry()

    def append_final_text(self, text: str) -> None:
        """Commit a FINAL; it replaces the open partial in the same step (no
        frame shows both, or neither)."""
        with self._lock:
            if self._pacer is not None:
                self._pacer.push_final(text)
                return
            self._buffer.append_final_text(text)
            self._partial_preview = ""

    def prune_history(self) -> int:
        """Drop committed caption lines that can no longer be displayed under
        the current layout, without touching what is on screen. Used by the
        ASR worker's silence-timeout CLEAR event -- the display keeps showing
        the last captions; only off-screen history is trimmed. See
        CaptionBuffer.prune_history.
        """
        with self._lock:
            return self._buffer.prune_history()

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._partial_preview = ""
            if self._pacer is not None:
                self._pacer.reset()

    def set_partial_preview(self, text: str) -> None:
        """Show unconfirmed ASR hypothesis text appended after the committed
        buffer, for display only -- never stored in caption history. Used
        when `asr.show_partial_captions` is enabled; overwritten or cleared
        on the next partial/final/clear event. Shown inline, continuing the
        last caption line (CaptionBuffer.visible_lines_with_tail).
        """
        with self._lock:
            if self._pacer is not None:
                self._pacer.push_partial(text)
            else:
                self._partial_preview = text

    @property
    def resolved_font_size_px(self) -> int:
        return self._geometry.font_size_px

    @property
    def canvas_config(self) -> CanvasConfig:
        return self._canvas_cfg

    @property
    def layout_config(self) -> LayoutConfig:
        return self._layout_cfg

    @property
    def style_config(self) -> StyleConfig:
        return self._style_cfg

    def snapshot_text(self) -> str:
        return self._buffer.snapshot_text()

    def visible_lines(self) -> tuple[str, ...]:
        """The caption lines currently on screen (partial preview included)."""
        return self._visible_lines_with_preview()

    def _visible_lines_with_preview(self) -> tuple[str, ...]:
        with self._lock:
            if self._pacer is not None:
                commits, self._partial_preview = self._pacer.advance(self._clock())
                for text in commits:
                    self._buffer.append_final_text(text)
            return tuple(self._buffer.visible_lines_with_tail(self._partial_preview))

    def get_frame(self) -> Image.Image:
        """Return the current RGBA frame, recompositing only if the visible
        caption lines (including any partial preview) changed since the last call.
        """
        lines = self._visible_lines_with_preview()
        if self._cached_frame is not None and lines == self._cached_lines:
            return self._cached_frame

        canvas_cfg, layout_cfg, style_cfg, geometry = (
            self._canvas_cfg,
            self._layout_cfg,
            self._style_cfg,
            self._geometry,
        )
        image = Image.new(
            "RGBA",
            (canvas_cfg.width_px, canvas_cfg.height_px),
            parse_rgba(canvas_cfg.background_color),
        )
        font = load_font(style_cfg.font_family, geometry.font_size_px)
        slot_tops = line_y_positions(geometry, len(lines), layout_cfg.vertical_align_within_box.value)
        for text, slot_top in zip(lines, slot_tops):
            composite_line(
                canvas=image,
                text=text,
                font=font,
                box_left=geometry.box.left,
                box_right=geometry.box.right,
                slot_top=slot_top,
                slot_height=geometry.per_line_height,
                canvas_height_px=canvas_cfg.height_px,
                style=style_cfg,
                text_align=layout_cfg.text_align,
            )

        self._cached_frame = image
        self._cached_lines = lines
        return image
