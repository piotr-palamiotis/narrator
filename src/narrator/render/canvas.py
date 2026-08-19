"""RGBA canvas compositor: ties layout geometry, the rolling caption buffer,
and style overrides together, producing the frame sent to NDI.

Recompositing is skipped when neither the visible caption lines nor the
layout/style config have changed since the last call -- the render+NDI loop
still sends a frame every tick (see narrator.ndi.sender), it just resends the
cached bytes instead of redoing the (relatively expensive) Pillow drawing.
"""
from __future__ import annotations

from PIL import Image

from narrator.captions.buffer import CaptionBuffer
from narrator.captions.layout import compute_line_geometry, line_y_positions
from narrator.captions.wrap import wrap_text
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
        self._apply_geometry()

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
        self._canvas_cfg, self._layout_cfg, self._style_cfg = canvas, layout, style
        self._apply_geometry()

    def append_final_text(self, text: str) -> None:
        self._buffer.append_final_text(text)

    def clear(self) -> None:
        self._buffer.clear()
        self._partial_preview = ""

    def set_partial_preview(self, text: str) -> None:
        """Show unconfirmed ASR hypothesis text appended after the committed
        buffer, for display only -- never stored in caption history. Used
        when `asr.show_partial_captions` is enabled; overwritten or cleared
        on the next partial/final/clear event.
        """
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

    def _visible_lines_with_preview(self) -> tuple[str, ...]:
        committed = self._buffer.visible_lines()
        if not self._partial_preview:
            return tuple(committed)
        preview_lines = wrap_text(
            self._partial_preview,
            self._style_cfg.font_family,
            self._geometry.font_size_px,
            self._geometry.box.width,
            _text_width,
        )
        combined = committed + preview_lines
        return tuple(combined[-self._layout_cfg.max_lines :])

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
