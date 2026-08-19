"""Live preview of the rendered caption canvas, shown inside the GUI.

Whenever the form matches the config the running app's real renderer is
already using, `PreviewSync` hands back that renderer's own `get_frame()`
output directly -- the exact same framebuffer being composited for the NDI
stream, pixel for pixel, including in-progress partial captions.

Only while the operator has unsaved edits pending (the form no longer
matches the live config) does it fall back to a private `CaptionRenderer`,
fully decoupled from the live broadcast, so editing fields can preview a
"what if" layout without mutating what's currently on air. That fallback
mirrors whatever text the live app's caption buffer currently holds, falling
back further to a bundled sample paragraph when the app hasn't produced any
captions yet, and re-renders through the same geometry/wrap pipeline the
real renderer uses so margin, font, and line-count edits reflow the
previewed text instead of deleting it.

`ScaledImageCanvas` is a small `tk.Canvas` that keeps the last frame it was
given and re-fits it (preserving aspect ratio, letterboxed as needed) on
every resize.
"""
from __future__ import annotations

import tkinter as tk

from PIL import Image, ImageTk

from narrator.config.schema import CanvasConfig, LayoutConfig, StyleConfig
from narrator.render.canvas import CaptionRenderer

PREVIEW_BG = "#1c1c1c"

SAMPLE_TEXT = (
    "This is a live preview of the caption canvas. It uses placeholder text "
    "so you can see how margins, font, and line count settings affect the "
    "layout before any real captions arrive. As you adjust the controls "
    "around this canvas, this text reflows to match -- it is never cleared "
    "just because the layout changed. Once the app starts transcribing real "
    "speech, this preview will switch over and mirror the actual live "
    "captions instead. Try changing the max line count, the margins on each "
    "side, or the font below to see the effect immediately, without needing "
    "to save first."
)


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
    ) -> Image.Image:
        if live_renderer is not None and (canvas, layout, style) == (
            live_renderer.canvas_config,
            live_renderer.layout_config,
            live_renderer.style_config,
        ):
            return live_renderer.get_frame()

        self._renderer.update_config(canvas, layout, style)

        source_text = ""
        if live_renderer is not None:
            try:
                source_text = live_renderer.snapshot_text()
            except Exception:
                source_text = ""
        if not source_text:
            source_text = SAMPLE_TEXT

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


PREVIEW_BORDER = "#4a4a4a"


class ScaledImageCanvas(tk.Canvas):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(
            parent,
            bg=PREVIEW_BG,
            highlightthickness=1,
            highlightbackground=PREVIEW_BORDER,
        )
        self._pil_image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self.bind("<Configure>", self._redraw)

    def show(self, image: Image.Image) -> None:
        self._pil_image = image
        self._redraw()

    def _redraw(self, _event: object = None) -> None:
        if self._pil_image is None:
            return
        avail_w, avail_h = self.winfo_width(), self.winfo_height()
        if avail_w <= 1 or avail_h <= 1:
            return
        img_w, img_h = self._pil_image.size
        scale = min(avail_w / img_w, avail_h / img_h)
        new_w, new_h = max(1, round(img_w * scale)), max(1, round(img_h * scale))
        resized = self._pil_image.resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.delete("all")
        cx, cy = avail_w // 2, avail_h // 2
        self.create_image(cx, cy, image=self._photo, anchor="center")
        self.create_rectangle(
            cx - new_w // 2, cy - new_h // 2, cx + new_w // 2, cy + new_h // 2,
            outline=PREVIEW_BORDER,
        )
