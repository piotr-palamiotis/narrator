"""Tkinter operator GUI: a form bound to the config schema. Every field edit
is autosaved to the YAML config file (debounced -- see _AUTOSAVE_DEBOUNCE_MS)
so changes persist even if the operator closes the app without pressing
Save; Save itself just flushes immediately and confirms. The running app
picks up the write via its config watcher (see
app_controller.AppController._on_config_changed) -- the GUI never mutates
in-memory state directly, so headless and GUI-driven operation stay
behaviorally identical (see the plan's "Config Schema & Operator Control
Surface" rationale).

Ctrl+N (bound at the root window level, see _CLEAR_SHORTCUT) clears the
current captions at any time; the Audio/ASR tab also shows a live mic input
level meter and the ASR worker's Active/Inactive state next to the device
picker."""
from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import sv_ttk
from pydantic import ValidationError

from narrator.audio.devices import list_input_devices
from narrator.asr.whisper_languages import language_display_name, whisper_language_codes
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
    CONTROL_WIDTH,
    dropdown_field,
    float_field,
    font_picker_field,
    int_field,
    mic_level_field,
    percent_field,
    rgba_color_field,
    scrollable_tab,
    text_field,
)

logger = logging.getLogger(__name__)

# Bound at the root window level (bind_all) so it works no matter which
# widget currently has focus, including the manual caption entry.
_CLEAR_SHORTCUT = "<Control-n>"

# Field attribute names whose value affects the rendered canvas -- edits to
# these trigger a debounced live preview refresh (in addition to the autosave
# every bound field gets). Audio/ASR and NDI fields are listed separately in
# _AUTOSAVE_ONLY_FIELD_NAMES below since they don't affect rendering.
_PREVIEW_FIELD_NAMES = [
    "canvas_width", "canvas_height", "canvas_bg",
    "max_lines", "margin_top", "margin_bottom", "margin_left", "margin_right",
    "interline", "text_align", "vertical_align",
    "font_family", "font_color",
    "band_enabled", "band_shape", "band_color", "band_radius", "band_padding",
    "shadow_enabled", "shadow_offset_x", "shadow_offset_y", "shadow_blur", "shadow_color",
    "stroke_enabled", "stroke_width", "stroke_color",
]

# Fields that don't affect the rendered canvas but still need to be persisted
# (and, via the config watcher, applied live) as soon as they change.
_AUTOSAVE_ONLY_FIELD_NAMES = [
    "audio_device",
    "asr_model", "asr_device", "asr_show_partial",
    "asr_prune_after", "asr_wipe_after",
    "ndi_name", "ndi_fps",
]

_AUTOSAVE_DEBOUNCE_MS = 800


def model_dropdown_options(current: str) -> list[str]:
    """Options for the Whisper model dropdown: every model name pywhispercpp
    can download on demand (its canonical list), plus any model already in
    the local cache (resolvable without a download), so a custom name or a
    direct path to a ggml file -- which config also accepts -- stays
    selectable instead of being silently lost on the next autosave. The
    dropdown stays editable (see dropdown_field) for the same reason.
    """
    try:
        from pywhispercpp.constants import AVAILABLE_MODELS, MODELS_DIR

        cached = [p.stem.removeprefix("ggml-") for p in sorted(Path(MODELS_DIR).glob("*.bin"))]
        options = sorted({*AVAILABLE_MODELS, *cached})
    except Exception:  # pragma: no cover - only on broken installs
        options = ["large-v3", "medium", "small", "base", "tiny"]
    if current and current not in options:
        options.insert(0, current)
    return options


def final_whitelist_display(codes: list[str]) -> str:
    """Comma-joined codes for the listbox's backing var; de-duplicated, order
    preserved (the first entry carries the most weight in ties and reads as
    the 'primary' language)."""
    seen: list[str] = []
    for code in codes:
        if code not in seen:
            seen.append(code)
    return ", ".join(seen)


def final_whitelist_parse(text: str) -> list[str]:
    """Inverse of final_whitelist_display: split, strip, drop empties."""
    return [code for code in (part.strip() for part in text.split(",")) if code]

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

        notebook = ttk.Notebook(bottom)
        notebook.pack(fill="both", expand=True, pady=(0, 8))

        self._build_canvas_tab(notebook)
        self._build_layout_tab(notebook)
        self._build_style_tab(notebook)
        self._build_audio_asr_tab(notebook)
        self._build_ndi_tab(notebook)

        for name in _PREVIEW_FIELD_NAMES + _AUTOSAVE_ONLY_FIELD_NAMES:
            getattr(self, name).trace_add("write", self._on_field_changed)

        self._preview_sync = PreviewSync(self.config.canvas, self.config.layout, self.config.style)
        self.root.after(50, lambda: paned.sashpos(0, max(self.root.winfo_height() // 2, 1)))

        self.root.bind_all(_CLEAR_SHORTCUT, self._on_clear)

        self._tray_menu = self._build_tray_popup_menu()
        self._tray = None
        try:
            self._tray = start_tray_icon(
                on_show=self._request_show_from_tray,
                on_quit=self._request_quit_from_tray,
                on_activate=self._request_tray_menu,
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
                vertical_align_within_box=self.vertical_align.get(),
            )
            style = StyleConfig(
                font_family=self.font_family.get(),
                font_color=self.font_color.get(),
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
        _canvas, layout, _style = cfgs
        self.preview_canvas.show(image, layout)

    def _poll_preview_text(self) -> None:
        self._refresh_preview()
        self.root.after(400, self._poll_preview_text)

    def _poll_status(self) -> None:
        """Keeps the ASR worker's Active/Inactive indicator (next to the ASR
        device picker on the Audio/ASR tab) current. "Active" means a live
        worker *process* exists and hasn't exited.
        """
        process = getattr(self.controller, "_asr_process", None)
        active = process is not None and process.is_alive()
        self.asr_status_var.set("● Active" if active else "○ Inactive")
        self.asr_status_label.configure(foreground="pale green" if active else "#e06c75")
        self.root.after(1000, self._poll_status)

    def _current_mic_level(self) -> float:
        """Latest RMS level from the live mic capture, or 0 when there is
        none (headless pieces, capture not started yet, or mid-restart --
        _restart_audio_capture swaps the MicCapture object itself, hence the
        attribute read instead of caching the instance).
        """
        mic = getattr(self.controller, "_mic", None)
        return mic.level if mic is not None else 0.0

    def _poll_mic_level(self) -> None:
        self.mic_level.refresh()
        self.root.after(100, self._poll_mic_level)

    def _build_canvas_tab(self, notebook: ttk.Notebook) -> None:
        tab = scrollable_tab(notebook, "Canvas")
        c = self.config.canvas
        self.canvas_width = int_field(tab, 0, "Width (px)", c.width_px, 1, 8000)
        self.canvas_height = int_field(tab, 1, "Height (px)", c.height_px, 1, 8000)

    def _build_layout_tab(self, notebook: ttk.Notebook) -> None:
        tab = scrollable_tab(notebook, "Layout")
        layout = self.config.layout
        self.max_lines = int_field(tab, 0, "Max lines", layout.max_lines, 1, 20)
        self.interline = percent_field(tab, 1, "Interline (% of canvas height)", layout.interline_pct)
        self.text_align = dropdown_field(tab, 2, "Text align", layout.text_align.value, ["left", "center", "right"])
        self.vertical_align = dropdown_field(
            tab, 3, "Vertical align", layout.vertical_align_within_box.value, ["top", "center", "bottom"]
        )

    def _build_style_tab(self, notebook: ttk.Notebook) -> None:
        tab = scrollable_tab(notebook, "Style")
        tab.grid_columnconfigure(0, weight=1, uniform="style_col")
        tab.grid_columnconfigure(1, weight=1, uniform="style_col")

        left = ttk.Frame(tab)
        left.grid(row=0, column=0, sticky="new", padx=(0, 8))
        right = ttk.Frame(tab)
        right.grid(row=0, column=1, sticky="new")

        style = self.config.style
        canvas = self.config.canvas

        font_group = ttk.LabelFrame(left, text="Font")
        font_group.pack(fill="x", pady=(0, 8))
        font_options = _list_font_files()
        if style.font_family not in font_options:
            font_options = sorted({*font_options, style.font_family})
        self.font_family = font_picker_field(font_group, 0, "Font file", style.font_family, font_options, _FONTS_DIR)
        self.font_color = rgba_color_field(font_group, 1, "Font color (RGBA)", style.font_color)

        bg_group = ttk.LabelFrame(left, text="Stream background")
        bg_group.pack(fill="x", pady=(0, 8))
        self.canvas_bg = rgba_color_field(bg_group, 0, "Background color (RGBA)", canvas.background_color)

        band = style.overrides.background_band
        band_group = ttk.LabelFrame(left, text="Background band")
        band_group.pack(fill="x", pady=(0, 8))
        self.band_enabled = checkbox_field(band_group, 0, "Background band enabled", band.enabled)
        self.band_shape = dropdown_field(band_group, 1, "Band shape", band.shape.value, ["rectangle", "pill"])
        self.band_color = rgba_color_field(band_group, 2, "Band color (RGBA)", band.color)
        self.band_radius = float_field(band_group, 3, "Band corner radius (%)", band.corner_radius_pct, 0.0, 50.0, 0.5)
        self.band_padding = float_field(band_group, 4, "Band padding (% of canvas height)", band.padding_pct, 0.0, 20.0, 0.5)

        shadow = style.overrides.shadow
        shadow_group = ttk.LabelFrame(right, text="Shadow")
        shadow_group.pack(fill="x", pady=(0, 8))
        self.shadow_enabled = checkbox_field(shadow_group, 0, "Shadow enabled", shadow.enabled)
        self.shadow_offset_x = float_field(shadow_group, 1, "Shadow offset X (%)", shadow.offset_x_pct, -10.0, 10.0, 0.1)
        self.shadow_offset_y = float_field(shadow_group, 2, "Shadow offset Y (%)", shadow.offset_y_pct, -10.0, 10.0, 0.1)
        self.shadow_blur = int_field(shadow_group, 3, "Shadow blur (px)", shadow.blur_px, 0, 100)
        self.shadow_color = rgba_color_field(shadow_group, 4, "Shadow color (RGBA)", shadow.color)

        stroke = style.overrides.stroke
        stroke_group = ttk.LabelFrame(right, text="Stroke")
        stroke_group.pack(fill="x", pady=(0, 8))
        self.stroke_enabled = checkbox_field(stroke_group, 0, "Stroke (outline) enabled", stroke.enabled)
        self.stroke_width = int_field(stroke_group, 1, "Stroke width (px)", stroke.width_px, 0, 50)
        self.stroke_color = rgba_color_field(stroke_group, 2, "Stroke color (RGBA)", stroke.color)

    def _build_audio_asr_tab(self, notebook: ttk.Notebook) -> None:
        tab = scrollable_tab(notebook, "Audio / ASR")
        tab.grid_columnconfigure(0, weight=1)

        mode_row = ttk.Frame(tab)
        mode_row.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 8))
        self.caption_mode_var = tk.StringVar(value="auto")
        ttk.Label(mode_row, text="Mode:").pack(side="left", padx=(0, 4))
        ttk.Radiobutton(
            mode_row, text="Auto (speech-to-text)", variable=self.caption_mode_var, value="auto",
            command=self._on_caption_mode_changed,
        ).pack(side="left")
        ttk.Radiobutton(
            mode_row, text="Manual", variable=self.caption_mode_var, value="manual",
            command=self._on_caption_mode_changed,
        ).pack(side="left")

        # These two frames share row 1 -- only one is ever gridded at a time
        # (see _on_caption_mode_changed), so switching to Manual clears the
        # auto settings out of the way and gives the entry the space instead.
        self._auto_settings_frame = ttk.Frame(tab)
        self._auto_settings_frame.grid(row=1, column=0, sticky="ew")
        auto = self._auto_settings_frame
        audio = self.config.audio
        try:
            device_names = ["default"] + [d.name for d in list_input_devices()]
        except Exception:
            device_names = ["default"]
        self.audio_device = dropdown_field(auto, 0, "Microphone", audio.device, device_names)

        # A live input-level meter: the fastest, friendliest way to answer
        # "is my mic even on?" without leaving the settings window.
        ttk.Label(auto, text="Input level").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.mic_level = mic_level_field(auto, get_level=self._current_mic_level)
        self.mic_level.grid(row=1, column=1, sticky="w", padx=4, pady=2)
        self._poll_mic_level()

        ttk.Separator(auto).grid(row=3, column=0, columnspan=4, sticky="ew", pady=6)
        asr = self.config.asr
        self.asr_model = dropdown_field(auto, 4, "Whisper model", asr.model, model_dropdown_options(asr.model))
        self.asr_device = dropdown_field(auto, 5, "ASR device", asr.device.value, ["auto", "vulkan", "cpu"])
        self.asr_status_var = tk.StringVar(value="○ Inactive")
        self.asr_status_label = ttk.Label(auto, textvariable=self.asr_status_var, foreground="gray")
        self.asr_status_label.grid(row=5, column=2, columnspan=2, sticky="w", padx=4)
        self.asr_show_partial = checkbox_field(auto, 6, "Show partial captions", asr.show_partial_captions)
        # Caption retention: prune only ever trims off-screen history; wipe is
        # the true screen clear (0 disables it). Both reset on any speech.
        self.asr_prune_after = float_field(
            auto, 7, "Prune history after silence (s)",
            asr.prune_history_after_silence_seconds, 1.0, 3600.0, 1.0,
        )
        self.asr_wipe_after = float_field(
            auto, 8, "Wipe screen after silence (s, 0 = off)",
            asr.wipe_screen_after_silence_seconds, 0.0, 3600.0, 1.0,
        )
        self._build_language_whitelist_editor(auto, start_row=10)
        ttk.Label(auto, text="Ctrl+N clears the captions at any time.", foreground="gray").grid(
            row=14, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2)
        )

        self._manual_settings_frame = ttk.Frame(tab)
        self._manual_settings_frame.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            self._manual_settings_frame,
            text="Type a caption, press Enter to send.",
            foreground="gray",
        ).pack(anchor="w", padx=4, pady=(0, 6))
        # A plain single-line Entry can't hold a newline no matter what the
        # operator presses, so "no new line can be created" falls out for
        # free -- Enter is bound below to submit the line instead of doing
        # nothing (Entry's default behavior for Return).
        self.manual_entry_var = tk.StringVar()
        self.manual_entry = ttk.Entry(self._manual_settings_frame, textvariable=self.manual_entry_var, width=CONTROL_WIDTH)
        self.manual_entry.pack(padx=4, anchor="w")
        self.manual_entry.bind("<Return>", self._on_manual_submit)
        self._manual_settings_frame.grid_remove()

    def _build_language_whitelist_editor(self, parent: ttk.Frame, start_row: int) -> None:
        """Language whitelist editor: restricts which languages the decoder
        may use (empty = whisper decides freely). A dropdown of every
        language this whisper build supports plus an Add button fills the
        list; Remove takes the selected entry out. Changes autosave like any
        other field (the var trace fires on every mutation).
        """
        row = start_row
        ttk.Separator(parent).grid(row=row, column=0, columnspan=4, sticky="ew", pady=6)
        row += 1
        ttk.Label(parent, text="Languages").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )
        row += 1

        codes = list(whisper_language_codes())

        picker_frame = ttk.Frame(parent)
        picker_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=4, pady=2)
        self._whitelist_var = tk.StringVar(value=final_whitelist_display(self.config.asr.language_whitelist))

        add_var = tk.StringVar()
        add_dropdown = ttk.Combobox(picker_frame, textvariable=add_var, values=[
            language_display_name(code) for code in codes
        ], state="readonly", width=CONTROL_WIDTH)
        add_dropdown.pack(side="left")
        self._whitelist_code_by_label = {
            language_display_name(code): code for code in codes
        }

        def _add() -> None:
            code = self._whitelist_code_by_label.get(add_var.get())
            if code is None:
                return
            current = final_whitelist_parse(self._whitelist_var.get())
            if code in current:
                return
            current.append(code)
            self._whitelist_var.set(final_whitelist_display(current))

        ttk.Button(picker_frame, text="Add", command=_add).pack(side="left", padx=(6, 0))

        row += 1
        list_frame = ttk.Frame(parent)
        list_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=4, pady=(2, 2))
        self._whitelist_listbox = tk.Listbox(list_frame, height=3, width=CONTROL_WIDTH, selectmode="browse")
        self._whitelist_listbox.pack(side="left")

        def _remove() -> None:
            selection = self._whitelist_listbox.curselection()
            if not selection:
                return
            current = final_whitelist_parse(self._whitelist_var.get())
            del current[selection[0]]
            self._whitelist_var.set(final_whitelist_display(current))

        ttk.Button(list_frame, text="Remove", command=_remove).pack(side="left", padx=(6, 0))

        # Sync listbox <-> var, and let the autosave machinery see changes.
        def _sync_listbox(*_args) -> None:
            self._whitelist_listbox.delete(0, "end")
            for code in final_whitelist_parse(self._whitelist_var.get()):
                self._whitelist_listbox.insert("end", language_display_name(code))

        self._whitelist_var.trace_add("write", _sync_listbox)
        self._whitelist_var.trace_add("write", self._on_field_changed)
        _sync_listbox()

    def _build_ndi_tab(self, notebook: ttk.Notebook) -> None:
        tab = scrollable_tab(notebook, "NDI")
        ndi = self.config.ndi
        self.ndi_name = text_field(tab, 0, "NDI source name", ndi.source_name)
        self.ndi_fps = float_field(tab, 1, "Frame rate (fps)", ndi.fps, 1.0, 120.0, 1.0)

    def _build_config_from_form(self) -> RootConfig:
        return RootConfig(
            app=self.config.app,
            audio=self.config.audio.model_copy(update={"device": self.audio_device.get()}),
            asr=AsrConfig(
                model=self.asr_model.get(),
                device=self.asr_device.get(),
                vad_threshold=self.config.asr.vad_threshold,
                step_ms=self.config.asr.step_ms,
                agreement_passes=self.config.asr.agreement_passes,
                end_silence_ms=self.config.asr.end_silence_ms,
                max_context_seconds=self.config.asr.max_context_seconds,
                prune_history_after_silence_seconds=self.asr_prune_after.get(),
                wipe_screen_after_silence_seconds=self.asr_wipe_after.get(),
                show_partial_captions=self.asr_show_partial.get(),
                language_whitelist=final_whitelist_parse(self._whitelist_var.get()),
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
                vertical_align_within_box=self.vertical_align.get(),
            ),
            style=StyleConfig(
                font_family=self.font_family.get(),
                font_color=self.font_color.get(),
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

    def _build_tray_popup_menu(self) -> tk.Menu:
        """A Tk menu standing in for a native tray context menu.

        pystray's Xlib/Xorg fallback backend (used when no AppIndicator/GTK
        bindings are importable -- see gui/tray.py) can't display its own
        menu and only ever delivers a plain click, so a real right-click
        context menu isn't reachable on that backend. This menu is popped up
        from that one click instead (see `_show_tray_menu`), giving the
        operator the same show/close options regardless of backend. Hiding
        the window is done via the window's own close (X) button -- see
        `_on_close` -- not from this menu.
        """
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Show Narrator", command=self._show_window)
        menu.add_separator()
        menu.add_command(label="Close Narrator", command=self._shutdown)
        return menu

    def _show_tray_menu(self) -> None:
        x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
        try:
            self._tray_menu.tk_popup(x, y)
        finally:
            self._tray_menu.grab_release()

    # The following three run on pystray's own thread (menu item/activation
    # callbacks), not the Tk thread -- Tkinter is not thread-safe, so every
    # one of them must marshal back onto the Tk event loop via
    # root.after(0, ...) rather than touching widgets directly.
    def _request_show_from_tray(self) -> None:
        self.root.after(0, self._show_window)

    def _request_quit_from_tray(self) -> None:
        self.root.after(0, self._shutdown)

    def _request_tray_menu(self) -> None:
        self.root.after(0, self._show_tray_menu)

    def _on_caption_mode_changed(self) -> None:
        mode = self.caption_mode_var.get()
        self.controller.set_caption_mode(mode)
        if mode == "manual":
            self._auto_settings_frame.grid_remove()
            self._manual_settings_frame.grid()
            self.manual_entry.focus_set()
        else:
            self._manual_settings_frame.grid_remove()
            self._auto_settings_frame.grid()

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
