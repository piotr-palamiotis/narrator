"""Tkinter operator GUI: a form bound to the config schema. Every field edit
is autosaved to the YAML config file (debounced -- see _AUTOSAVE_DEBOUNCE_MS)
so changes persist even if the operator closes the app without pressing
Save; Save itself just flushes immediately and confirms. The running app
picks up the write via its config watcher (see
app_controller.AppController._on_config_changed) -- the GUI never mutates
in-memory state directly, so headless and GUI-driven operation stay
behaviorally identical (see the plan's "Config Schema & Operator Control
Surface" rationale).
"""
from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import sv_ttk
from pydantic import ValidationError

from narrator.audio.devices import list_input_devices
from narrator.config.loader import load_config, save_config
from narrator.config.schema import (
    AsrConfig,
    BackgroundBandStyle,
    CanvasConfig,
    GuiConfig,
    LayoutConfig,
    NdiConfig,
    RootConfig,
    ShadowStyle,
    StrokeStyle,
    StyleConfig,
    StyleOverrides,
)
from narrator.gui.preview import PreviewSync, ScaledImageCanvas
from narrator.gui.tray import start_tray_icon
from narrator.gui.widgets import (
    checkbox_field,
    compact_percent_field,
    dropdown_field,
    float_field,
    font_picker_field,
    int_field,
    percent_field,
    rgba_color_field,
    text_field,
)

logger = logging.getLogger(__name__)

# Bound at the root window level (bind_all) so it works no matter which
# widget currently has focus, including the manual caption entry.
_CLEAR_SHORTCUT = "<Control-l>"

# Field attribute names whose value affects the rendered canvas -- edits to
# these trigger a debounced live preview refresh (in addition to the autosave
# every bound field gets). Audio/ASR and NDI fields are listed separately in
# _AUTOSAVE_ONLY_FIELD_NAMES below since they don't affect rendering.
_PREVIEW_FIELD_NAMES = [
    "canvas_width", "canvas_height", "canvas_bg",
    "max_lines", "margin_top", "margin_bottom", "margin_left", "margin_right",
    "interline", "text_align",
    "font_family", "font_color", "min_font_px", "line_fill_ratio",
    "band_enabled", "band_shape", "band_color", "band_radius", "band_padding",
    "shadow_enabled", "shadow_offset_x", "shadow_offset_y", "shadow_blur", "shadow_color",
    "stroke_enabled", "stroke_width", "stroke_color",
]

# Fields that don't affect the rendered canvas but still need to be persisted
# (and, via the config watcher, applied live) as soon as they change.
_AUTOSAVE_ONLY_FIELD_NAMES = [
    "audio_device",
    "asr_model", "asr_language", "asr_device", "asr_show_partial",
    "ndi_name", "ndi_fps",
]

_AUTOSAVE_DEBOUNCE_MS = 800

# Font files available in the "Font file" dropdown, scanned from assets/fonts/.
_FONTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
_FONT_EXTENSIONS = {".ttf", ".otf"}


def _list_font_files() -> list[str]:
    if not _FONTS_DIR.is_dir():
        return []
    return sorted(
        f"assets/fonts/{p.name}"
        for p in _FONTS_DIR.iterdir()
        if p.suffix.lower() in _FONT_EXTENSIONS
    )


class NarratorGUI:
    def __init__(self, root: tk.Tk, controller, config_path: Path) -> None:
        self.root = root
        self.controller = controller
        self.config_path = config_path
        self.config = load_config(config_path)
        self._debounce_id: str | None = None
        self._autosave_debounce_id: str | None = None

        root.title("Narrator - Live Caption Settings")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        paned = ttk.PanedWindow(root, orient="vertical")
        paned.pack(fill="both", expand=True, padx=8, pady=8)

        top = ttk.Frame(paned)
        bottom = ttk.Frame(paned)
        paned.add(top, weight=1)
        paned.add(bottom, weight=1)

        self._build_preview_pane(top)
        self._build_caption_mode_row(top)
        self._build_basic_font_row(bottom)

        notebook = ttk.Notebook(bottom)
        notebook.pack(fill="both", expand=True, pady=(0, 8))

        self._build_canvas_tab(notebook)
        self._build_layout_tab(notebook)
        self._build_style_tab(notebook)
        self._build_audio_asr_tab(notebook)
        self._build_ndi_tab(notebook)

        button_bar = ttk.Frame(bottom)
        button_bar.pack(fill="x")
        ttk.Button(button_bar, text="Save", command=self._on_save).pack(side="left")
        ttk.Button(button_bar, text="Clear captions (Ctrl+L)", command=self._on_clear).pack(side="left", padx=(6, 0))
        self.status_var = tk.StringVar(value="ASR backend: (starting...)")
        ttk.Label(button_bar, textvariable=self.status_var).pack(side="right")

        for name in _PREVIEW_FIELD_NAMES + _AUTOSAVE_ONLY_FIELD_NAMES:
            getattr(self, name).trace_add("write", self._on_field_changed)

        self._preview_sync = PreviewSync(self.config.canvas, self.config.layout, self.config.style)
        self.root.after(50, lambda: paned.sashpos(0, max(self.root.winfo_height() // 2, 1)))

        self.root.bind_all(_CLEAR_SHORTCUT, self._on_clear)

        self._tray = None
        try:
            self._tray = start_tray_icon(
                on_show=self._request_show_from_tray,
                on_clear=self._request_clear_from_tray,
                on_quit=self._request_quit_from_tray,
            )
        except Exception:
            logger.warning("Failed to start system tray icon; closing the window will quit Narrator", exc_info=True)
            self._tray = None

        self._poll_status()
        self._refresh_preview()
        self._poll_preview_text()

    def _build_preview_pane(self, parent: ttk.Frame) -> None:
        # Margin controls are bare spinboxes (no text labels) tucked right
        # against the edge of the preview they affect, so the preview canvas
        # -- row/column 1 -- gets the rest of the pane's space.
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        layout = self.config.layout

        top_frame = ttk.Frame(parent)
        top_frame.grid(row=0, column=1, sticky="n")
        self.margin_top = compact_percent_field(top_frame, layout.margin_top_pct)

        left_frame = ttk.Frame(parent)
        left_frame.grid(row=1, column=0, sticky="w", padx=(0, 2))
        self.margin_left = compact_percent_field(left_frame, layout.margin_left_pct)

        self.preview_canvas = ScaledImageCanvas(parent)
        self.preview_canvas.grid(row=1, column=1, sticky="nsew")

        right_frame = ttk.Frame(parent)
        right_frame.grid(row=1, column=2, sticky="e", padx=(2, 0))
        self.margin_right = compact_percent_field(right_frame, layout.margin_right_pct)

        bottom_frame = ttk.Frame(parent)
        bottom_frame.grid(row=2, column=1, sticky="s")
        self.margin_bottom = compact_percent_field(bottom_frame, layout.margin_bottom_pct)

    def _build_basic_font_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 8))
        style = self.config.style

        path_frame = ttk.Frame(row)
        path_frame.pack(side="left", padx=(0, 16))
        font_options = _list_font_files()
        if style.font_family not in font_options:
            font_options = sorted({*font_options, style.font_family})
        self.font_family = font_picker_field(path_frame, 0, "Font file", style.font_family, font_options, _FONTS_DIR)

        color_frame = ttk.Frame(row)
        color_frame.pack(side="left")
        self.font_color = rgba_color_field(color_frame, 0, "Font color (RGBA)", style.font_color)

    def _on_field_changed(self, *_args) -> None:
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
        self._debounce_id = self.root.after(50, self._refresh_preview)

        # Autosaving runs on its own, longer debounce -- separate from the
        # preview's fast one -- so it doesn't write (and trigger a live
        # ASR/NDI restart) on every keystroke or spinbox tick while a value
        # is still actively being adjusted.
        if self._autosave_debounce_id is not None:
            self.root.after_cancel(self._autosave_debounce_id)
        self._autosave_debounce_id = self.root.after(_AUTOSAVE_DEBOUNCE_MS, self._autosave)

    def _gather_preview_config(self) -> tuple[CanvasConfig, LayoutConfig, StyleConfig] | None:
        try:
            canvas = CanvasConfig(
                width_px=self.canvas_width.get(),
                height_px=self.canvas_height.get(),
                background_color=self.canvas_bg.get(),
            )
            layout = LayoutConfig(
                max_lines=self.max_lines.get(),
                margin_top_pct=self.margin_top.get(),
                margin_bottom_pct=self.margin_bottom.get(),
                margin_left_pct=self.margin_left.get(),
                margin_right_pct=self.margin_right.get(),
                interline_pct=self.interline.get(),
                text_align=self.text_align.get(),
                vertical_align_within_box=self.config.layout.vertical_align_within_box,
            )
            style = StyleConfig(
                font_family=self.font_family.get(),
                font_color=self.font_color.get(),
                min_font_px=self.min_font_px.get(),
                line_fill_ratio=self.line_fill_ratio.get(),
                overrides=StyleOverrides(
                    background_band=BackgroundBandStyle(
                        enabled=self.band_enabled.get(),
                        shape=self.band_shape.get(),
                        color=self.band_color.get(),
                        corner_radius_pct=self.band_radius.get(),
                        padding_pct=self.band_padding.get(),
                    ),
                    shadow=ShadowStyle(
                        enabled=self.shadow_enabled.get(),
                        offset_x_pct=self.shadow_offset_x.get(),
                        offset_y_pct=self.shadow_offset_y.get(),
                        blur_px=self.shadow_blur.get(),
                        color=self.shadow_color.get(),
                    ),
                    stroke=StrokeStyle(
                        enabled=self.stroke_enabled.get(),
                        width_px=self.stroke_width.get(),
                        color=self.stroke_color.get(),
                    ),
                ),
            )
        except (ValidationError, tk.TclError):
            return None
        return canvas, layout, style

    def _refresh_preview(self) -> None:
        self._debounce_id = None
        cfgs = self._gather_preview_config()
        if cfgs is None:
            return
        live_renderer = getattr(self.controller, "_renderer", None)
        image = self._preview_sync.refresh(*cfgs, live_renderer)
        self.preview_canvas.show(image)

    def _poll_preview_text(self) -> None:
        self._refresh_preview()
        self.root.after(400, self._poll_preview_text)

    def _poll_status(self) -> None:
        backend = getattr(self.controller, "resolved_asr_backend", "unresolved")
        self.status_var.set(f"ASR backend: {backend}")
        self.root.after(1000, self._poll_status)

    def _build_canvas_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Canvas")
        c = self.config.canvas
        self.canvas_width = int_field(tab, 0, "Width (px)", c.width_px, 1, 8000)
        self.canvas_height = int_field(tab, 1, "Height (px)", c.height_px, 1, 8000)
        self.canvas_bg = rgba_color_field(tab, 2, "Background color (RGBA)", c.background_color)
        ttk.Label(
            tab,
            text="Note: width/height changes restart the NDI stream (briefly interrupting it) to apply.",
            foreground="gray",
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2))

    def _build_layout_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Layout")
        layout = self.config.layout
        self.max_lines = int_field(tab, 0, "Max lines", layout.max_lines, 1, 20)
        self.interline = percent_field(tab, 1, "Interline (% of canvas height)", layout.interline_pct)
        self.text_align = dropdown_field(tab, 2, "Text align", layout.text_align.value, ["left", "center", "right"])
        ttk.Label(tab, text="Font size is never set directly -- it's derived from these values.", foreground="gray").grid(
            row=3, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )
        ttk.Label(tab, text="Margins are set around the preview canvas above.", foreground="gray").grid(
            row=4, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 2)
        )

    def _build_style_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Style")
        style = self.config.style
        self.min_font_px = int_field(tab, 0, "Min font size (px floor)", style.min_font_px, 1, 400)
        self.line_fill_ratio = float_field(tab, 1, "Line fill ratio", style.line_fill_ratio, 0.1, 1.0, 0.05)
        ttk.Label(tab, text="Font file path/color are set in the row below the preview canvas.", foreground="gray").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 2)
        )

        band = style.overrides.background_band
        ttk.Separator(tab).grid(row=3, column=0, columnspan=4, sticky="ew", pady=6)
        self.band_enabled = checkbox_field(tab, 4, "Background band enabled", band.enabled)
        self.band_shape = dropdown_field(tab, 5, "Band shape", band.shape.value, ["rectangle", "pill"])
        self.band_color = rgba_color_field(tab, 6, "Band color (RGBA)", band.color)
        self.band_radius = float_field(tab, 7, "Band corner radius (%)", band.corner_radius_pct, 0.0, 50.0, 0.5)
        self.band_padding = float_field(tab, 8, "Band padding (% of canvas height)", band.padding_pct, 0.0, 20.0, 0.5)

        shadow = style.overrides.shadow
        ttk.Separator(tab).grid(row=9, column=0, columnspan=4, sticky="ew", pady=6)
        self.shadow_enabled = checkbox_field(tab, 10, "Shadow enabled", shadow.enabled)
        self.shadow_offset_x = float_field(tab, 11, "Shadow offset X (%)", shadow.offset_x_pct, -10.0, 10.0, 0.1)
        self.shadow_offset_y = float_field(tab, 12, "Shadow offset Y (%)", shadow.offset_y_pct, -10.0, 10.0, 0.1)
        self.shadow_blur = int_field(tab, 13, "Shadow blur (px)", shadow.blur_px, 0, 100)
        self.shadow_color = rgba_color_field(tab, 14, "Shadow color (RGBA)", shadow.color)

        stroke = style.overrides.stroke
        ttk.Separator(tab).grid(row=15, column=0, columnspan=4, sticky="ew", pady=6)
        self.stroke_enabled = checkbox_field(tab, 16, "Stroke (outline) enabled", stroke.enabled)
        self.stroke_width = int_field(tab, 17, "Stroke width (px)", stroke.width_px, 0, 50)
        self.stroke_color = rgba_color_field(tab, 18, "Stroke color (RGBA)", stroke.color)

    def _build_audio_asr_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Audio / ASR")
        audio = self.config.audio
        try:
            device_names = ["default"] + [d.name for d in list_input_devices()]
        except Exception:
            device_names = ["default"]
        self.audio_device = dropdown_field(tab, 0, "Microphone", audio.device, device_names)

        asr = self.config.asr
        ttk.Separator(tab).grid(row=1, column=0, columnspan=4, sticky="ew", pady=6)
        self.asr_model = text_field(tab, 2, "Whisper model", asr.model, width=30)
        self.asr_language = text_field(tab, 3, "Language (ISO code or 'auto')", asr.language, width=10)
        self.asr_device = dropdown_field(tab, 4, "ASR device", asr.device.value, ["auto", "vulkan", "cpu"])
        self.asr_show_partial = checkbox_field(tab, 5, "Show partial (unconfirmed) captions", asr.show_partial_captions)
        ttk.Label(tab, text="Audio/ASR model+device changes restart the ASR worker automatically.", foreground="gray").grid(
            row=6, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )

    def _build_ndi_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="NDI")
        ndi = self.config.ndi
        self.ndi_name = text_field(tab, 0, "NDI source name", ndi.source_name, width=30)
        self.ndi_fps = float_field(tab, 1, "Frame rate (fps)", ndi.fps, 1.0, 120.0, 1.0)
        ttk.Label(tab, text="Changes restart the NDI stream automatically (briefly interrupting it).", foreground="gray").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )

    def _build_config_from_form(self) -> RootConfig:
        return RootConfig(
            app=self.config.app,
            audio=self.config.audio.model_copy(update={"device": self.audio_device.get()}),
            asr=AsrConfig(
                model=self.asr_model.get(),
                language=self.asr_language.get(),
                device=self.asr_device.get(),
                vad_threshold=self.config.asr.vad_threshold,
                step_ms=self.config.asr.step_ms,
                agreement_passes=self.config.asr.agreement_passes,
                end_silence_ms=self.config.asr.end_silence_ms,
                max_context_seconds=self.config.asr.max_context_seconds,
                clear_after_silence_seconds=self.config.asr.clear_after_silence_seconds,
                show_partial_captions=self.asr_show_partial.get(),
            ),
            canvas=CanvasConfig(
                width_px=self.canvas_width.get(),
                height_px=self.canvas_height.get(),
                background_color=self.canvas_bg.get(),
            ),
            layout=LayoutConfig(
                max_lines=self.max_lines.get(),
                margin_top_pct=self.margin_top.get(),
                margin_bottom_pct=self.margin_bottom.get(),
                margin_left_pct=self.margin_left.get(),
                margin_right_pct=self.margin_right.get(),
                interline_pct=self.interline.get(),
                text_align=self.text_align.get(),
                vertical_align_within_box=self.config.layout.vertical_align_within_box,
            ),
            style=StyleConfig(
                font_family=self.font_family.get(),
                font_color=self.font_color.get(),
                min_font_px=self.min_font_px.get(),
                line_fill_ratio=self.line_fill_ratio.get(),
                overrides=StyleOverrides(
                    background_band=BackgroundBandStyle(
                        enabled=self.band_enabled.get(),
                        shape=self.band_shape.get(),
                        color=self.band_color.get(),
                        corner_radius_pct=self.band_radius.get(),
                        padding_pct=self.band_padding.get(),
                    ),
                    shadow=ShadowStyle(
                        enabled=self.shadow_enabled.get(),
                        offset_x_pct=self.shadow_offset_x.get(),
                        offset_y_pct=self.shadow_offset_y.get(),
                        blur_px=self.shadow_blur.get(),
                        color=self.shadow_color.get(),
                    ),
                    stroke=StrokeStyle(
                        enabled=self.stroke_enabled.get(),
                        width_px=self.stroke_width.get(),
                        color=self.stroke_color.get(),
                    ),
                ),
            ),
            ndi=NdiConfig(source_name=self.ndi_name.get(), fps=self.ndi_fps.get()),
            gui=GuiConfig(enabled=self.config.gui.enabled),
        )

    def _persist(self, new_config: RootConfig) -> None:
        save_config(new_config, self.config_path)
        self.config = new_config

    def _autosave(self) -> None:
        """Writes whatever is currently in the form to the config file, so a
        change survives even if the operator never presses Save and just
        closes the app -- the config file, not the bundled defaults, is the
        single source of truth for the running app (see config/loader.py);
        defaults are only ever used to seed that file the first time it's
        created.
        """
        self._autosave_debounce_id = None
        try:
            new_config = self._build_config_from_form()
        except (ValidationError, tk.TclError):
            return
        if new_config == self.config:
            return
        self._persist(new_config)

    def _on_save(self) -> None:
        if self._autosave_debounce_id is not None:
            self.root.after_cancel(self._autosave_debounce_id)
            self._autosave_debounce_id = None
        try:
            new_config = self._build_config_from_form()
        except ValidationError as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        self._persist(new_config)
        messagebox.showinfo("Saved", f"Configuration saved to {self.config_path}")

    def _on_close(self) -> None:
        """The window's X button. Hides to the tray instead of quitting when a
        tray icon is running -- see build_tray_icon/_request_quit_from_tray
        for the actual quit path -- so the app keeps capturing/transcribing/
        sending NDI in the background. Falls back to quitting outright if the
        tray icon failed to start (e.g. no tray host on this desktop), so the
        window always stays closable.
        """
        if self._tray is not None:
            self.root.withdraw()
            return
        self._shutdown()

    def _shutdown(self) -> None:
        if self._autosave_debounce_id is not None:
            self.root.after_cancel(self._autosave_debounce_id)
            self._autosave()
        if self._tray is not None:
            # Some tray backends can take a few seconds to tear down (e.g.
            # waiting on a setup thread that's blocked because no systray
            # host ever docked the icon) -- do it on a daemon thread so
            # Quit closes the window immediately instead of freezing it.
            threading.Thread(target=self._tray.stop, daemon=True).start()
        self.root.destroy()

    def _show_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    # The following three run on pystray's own thread (menu item callbacks),
    # not the Tk thread -- Tkinter is not thread-safe, so every one of them
    # must marshal back onto the Tk event loop via root.after(0, ...) rather
    # than touching widgets directly.
    def _request_show_from_tray(self) -> None:
        self.root.after(0, self._show_window)

    def _request_clear_from_tray(self) -> None:
        self.root.after(0, self._on_clear)

    def _request_quit_from_tray(self) -> None:
        self.root.after(0, self._shutdown)

    def _build_caption_mode_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))

        self.caption_mode_var = tk.StringVar(value="auto")
        ttk.Label(row, text="Captions:").pack(side="left", padx=(0, 4))
        ttk.Radiobutton(
            row, text="Auto", variable=self.caption_mode_var, value="auto", command=self._on_caption_mode_changed
        ).pack(side="left")
        ttk.Radiobutton(
            row, text="Manual", variable=self.caption_mode_var, value="manual", command=self._on_caption_mode_changed
        ).pack(side="left")

        # A plain single-line Entry can't hold a newline no matter what the
        # operator presses, so "no new line can be created" falls out for
        # free -- Enter is bound below to submit the line instead of doing
        # nothing (Entry's default behavior for Return).
        self.manual_entry_var = tk.StringVar()
        self.manual_entry = ttk.Entry(row, textvariable=self.manual_entry_var)
        self.manual_entry.bind("<Return>", self._on_manual_submit)
        # Not gridded/packed here -- shown only in manual mode, see
        # _on_caption_mode_changed.

    def _on_caption_mode_changed(self) -> None:
        mode = self.caption_mode_var.get()
        self.controller.set_caption_mode(mode)
        if mode == "manual":
            self.manual_entry.pack(side="left", fill="x", expand=True, padx=(12, 0))
            self.manual_entry.focus_set()
        else:
            self.manual_entry.pack_forget()

    def _on_manual_submit(self, _event: object = None) -> None:
        self.controller.submit_manual_text(self.manual_entry_var.get())
        self.manual_entry_var.set("")

    def _on_clear(self, _event: object = None) -> None:
        self.controller.clear_captions()


def run_gui(controller, config_path: Path) -> None:
    root = tk.Tk()
    sv_ttk.set_theme("dark")
    NarratorGUI(root, controller, config_path)
    root.mainloop()
