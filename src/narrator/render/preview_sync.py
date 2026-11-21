"""Preview frames of the caption canvas for the web UI.

Whenever the requested canvas/layout/style match what the running app's real
renderer is using, `PreviewSync` hands back that renderer's own `get_frame()`
output directly -- the exact framebuffer being composited for the NDI
stream, including in-progress partial captions.

Only while the operator has unsaved display edits (a "draft") does it fall
back to a private `CaptionRenderer`, fully decoupled from the live broadcast,
so editing fields can preview a "what if" layout without touching what is on
air. That fallback mirrors the live caption buffer's text and re-renders it
through the same geometry/wrap pipeline, so margin, font and line-count edits
reflow the previewed text instead of deleting it.

(Moved out of the retired Tk GUI, narrator.gui.preview.)
"""
from __future__ import annotations

from PIL import Image

from narrator.config.schema import CanvasConfig, LayoutConfig, StyleConfig
from narrator.render.canvas import CaptionRenderer


class PreviewSync:
    def __init__(self, canvas: CanvasConfig, layout: LayoutConfig, style: StyleConfig) -> None:
        self._renderer = CaptionRenderer(canvas, layout, style)
        self._last_source_text = ""

    def refresh(
        self,
        canvas: CanvasConfig,
        layout: LayoutConfig,
        style: StyleConfig,
        live_renderer: CaptionRenderer | None,
        sample_text: str = "",
    ) -> Image.Image:
        """`sample_text` (non-empty) previews that text instead of the live
        captions -- for styling while nothing is on air; never broadcast."""
        if not sample_text and live_renderer is not None and (canvas, layout, style) == (
            live_renderer.canvas_config,
            live_renderer.layout_config,
            live_renderer.style_config,
        ):
            return live_renderer.get_frame()

        if (canvas, layout, style) != (
            self._renderer.canvas_config,
            self._renderer.layout_config,
            self._renderer.style_config,
        ):  # update_config forces a full re-layout; skip it when unchanged
            self._renderer.update_config(canvas, layout, style)

        source_text = sample_text
        if not sample_text and live_renderer is not None:
            try:
                source_text = live_renderer.snapshot_text()
            except Exception:
                source_text = ""

        if source_text != self._last_source_text:
            if source_text.startswith(self._last_source_text) and self._last_source_text:
                delta = source_text[len(self._last_source_text) :].strip()
                if delta:
                    self._renderer.append_final_text(delta)
            else:
                self._renderer.clear()
                if source_text:
                    self._renderer.append_final_text(source_text)
            self._last_source_text = source_text

        return self._renderer.get_frame()
