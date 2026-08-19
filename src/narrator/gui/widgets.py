"""Small reusable Tkinter form widgets shared across gui/app.py's config tabs."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

from narrator.config.schema import parse_rgba

# Fallback colors for the font picker's thumbnails, used only if the active
# ttk theme doesn't define the style options they're normally read from
# (gui/app.py always applies sv_ttk's dark theme, so this is just a safety
# net -- see sv_ttk's dark.tcl "colors" array for where these came from).
_FONT_PICKER_BG = "#1c1c1c"
_FONT_PICKER_FG = "#fafafa"


def labeled_row(parent: tk.Widget, row: int, label: str) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)


def percent_field(parent: tk.Widget, row: int, label: str, initial: float) -> tk.DoubleVar:
    labeled_row(parent, row, label)
    var = tk.DoubleVar(value=initial)
    spin = ttk.Spinbox(parent, from_=0.0, to=100.0, increment=0.5, textvariable=var, width=8)
    spin.grid(row=row, column=1, sticky="w", padx=4, pady=2)
    return var


def compact_percent_field(parent: tk.Widget, initial: float) -> tk.DoubleVar:
    """A bare percent spinbox with no label row, for spots (like around the
    preview canvas) where every pixel is better spent on the thing being
    previewed than on repeating what the control obviously does.
    """
    var = tk.DoubleVar(value=initial)
    spin = ttk.Spinbox(parent, from_=0.0, to=100.0, increment=0.5, textvariable=var, width=5)
    spin.pack()
    return var


def int_field(parent: tk.Widget, row: int, label: str, initial: int, from_: int = 0, to: int = 10000) -> tk.IntVar:
    labeled_row(parent, row, label)
    var = tk.IntVar(value=initial)
    spin = ttk.Spinbox(parent, from_=from_, to=to, textvariable=var, width=8)
    spin.grid(row=row, column=1, sticky="w", padx=4, pady=2)
    return var


def float_field(parent: tk.Widget, row: int, label: str, initial: float, from_: float = 0.0, to: float = 1000.0, increment: float = 0.1) -> tk.DoubleVar:
    labeled_row(parent, row, label)
    var = tk.DoubleVar(value=initial)
    spin = ttk.Spinbox(parent, from_=from_, to=to, increment=increment, textvariable=var, width=8)
    spin.grid(row=row, column=1, sticky="w", padx=4, pady=2)
    return var


def text_field(parent: tk.Widget, row: int, label: str, initial: str, width: int = 30) -> tk.StringVar:
    labeled_row(parent, row, label)
    var = tk.StringVar(value=initial)
    ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="w", padx=4, pady=2)
    return var


def dropdown_field(parent: tk.Widget, row: int, label: str, initial: str, options: list[str]) -> tk.StringVar:
    labeled_row(parent, row, label)
    var = tk.StringVar(value=initial)
    ttk.Combobox(parent, textvariable=var, values=options, state="readonly", width=27).grid(
        row=row, column=1, sticky="w", padx=4, pady=2
    )
    return var


def font_picker_field(
    parent: tk.Widget, row: int, label: str, initial: str, options: list[str], fonts_dir: Path
) -> tk.StringVar:
    """A font dropdown whose entries are rendered in their own typeface --
    just the font's name, not its file path -- instead of listed as plain
    text, so the operator can see what a font looks like before picking it.

    `options` are paths as stored in config (e.g. "assets/fonts/Foo.ttf",
    relative to the project root); `fonts_dir` is that directory, used to
    resolve a path to an actual file to rasterize for the preview.
    """
    labeled_row(parent, row, label)
    var = tk.StringVar(value=initial)

    style = ttk.Style()
    bg = style.lookup("TEntry", "fieldbackground") or _FONT_PICKER_BG
    fg = style.lookup("TEntry", "foreground") or _FONT_PICKER_FG

    thumbs: list[ImageTk.PhotoImage] = []  # kept alive for as long as the menu/button are

    def _resolve(path: str) -> Path | None:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        candidate = fonts_dir / candidate.name
        return candidate if candidate.is_file() else None

    def _render(path: str, height: int = 26) -> ImageTk.PhotoImage:
        name = Path(path).stem
        font_file = _resolve(path)
        try:
            font = ImageFont.truetype(str(font_file), height - 8) if font_file else ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()
        left, top, right, bottom = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), name, font=font)
        img = Image.new("RGBA", (right - left + 16, height), bg)
        ImageDraw.Draw(img).text((8 - left, (height - (bottom - top)) // 2 - top), name, font=font, fill=fg)
        photo = ImageTk.PhotoImage(img)
        thumbs.append(photo)
        return photo

    def _select(path: str) -> None:
        var.set(path)
        button.configure(image=_render(path))

    menu = tk.Menu(parent, tearoff=0, background=bg, foreground=fg, activebackground=fg, activeforeground=bg)
    for path in options:
        menu.add_command(image=_render(path), command=lambda p=path: _select(p))

    button = ttk.Menubutton(parent, image=_render(initial), menu=menu)
    button.grid(row=row, column=1, sticky="w", padx=4, pady=2)
    return var


def checkbox_field(parent: tk.Widget, row: int, label: str, initial: bool) -> tk.BooleanVar:
    var = tk.BooleanVar(value=initial)
    ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2)
    return var


def _ask_opacity(parent: tk.Widget, hex_rgb: str, initial_alpha: int) -> int | None:
    """A modal popup with a slider for the alpha channel, shown right after
    the system color picker returns (which has no alpha support of its
    own). Returns the chosen alpha (0-255), or None if cancelled.
    """
    top = tk.Toplevel(parent)
    top.title("Opacity")
    top.transient(parent.winfo_toplevel())
    top.resizable(False, False)

    result: dict[str, int | None] = {"alpha": None}

    frame = ttk.Frame(top, padding=10)
    frame.pack(fill="both", expand=True)

    preview = tk.Label(frame, text="  ", relief="sunken", width=4, height=2)
    preview.grid(row=0, column=0, rowspan=2, padx=(0, 10))

    r, g, b = int(hex_rgb[1:3], 16), int(hex_rgb[3:5], 16), int(hex_rgb[5:7], 16)
    pct_var = tk.DoubleVar(value=round(initial_alpha / 255 * 100))
    pct_label = ttk.Label(frame, text="")

    def _update_preview(*_args) -> None:
        pct = pct_var.get()
        alpha = pct / 100
        # tk can't render real alpha; approximate by blending against a
        # mid-gray backdrop so the preview still hints at translucency.
        blended = tuple(round(c * alpha + 128 * (1 - alpha)) for c in (r, g, b))
        preview.configure(bg="#{:02x}{:02x}{:02x}".format(*blended))
        pct_label.configure(text=f"{pct:.0f}%")

    ttk.Label(frame, text="Opacity").grid(row=0, column=1, sticky="w")
    ttk.Scale(frame, from_=0, to=100, variable=pct_var, orient="horizontal", length=180, command=_update_preview).grid(
        row=1, column=1, sticky="ew"
    )
    pct_label.grid(row=1, column=2, padx=(6, 0))

    def _ok() -> None:
        result["alpha"] = round(pct_var.get() / 100 * 255)
        top.destroy()

    def _cancel() -> None:
        top.destroy()

    btns = ttk.Frame(frame)
    btns.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky="e")
    ttk.Button(btns, text="Cancel", command=_cancel).pack(side="right", padx=(6, 0))
    ttk.Button(btns, text="OK", command=_ok).pack(side="right")

    _update_preview()
    top.protocol("WM_DELETE_WINDOW", _cancel)
    top.grab_set()
    top.focus_set()
    top.wait_window()
    return result["alpha"]


def rgba_color_field(parent: tk.Widget, row: int, label: str, initial: str) -> tk.StringVar:
    """A hex-text entry (source of truth, editable directly as '#RRGGBBAA')
    plus a swatch that doubles as its own button: clicking it opens the
    system color picker for RGB, then a follow-up popup to set the alpha
    channel (Tkinter's built-in chooser has no alpha support).
    """
    labeled_row(parent, row, label)
    var = tk.StringVar(value=initial)
    entry = ttk.Entry(parent, textvariable=var, width=10)
    entry.grid(row=row, column=1, sticky="w", padx=4, pady=2)

    def _refresh_swatch(*_args) -> None:
        try:
            r, g, b, _a = parse_rgba(var.get())
            swatch.configure(bg=f"#{r:02x}{g:02x}{b:02x}")
        except ValueError:
            pass

    def _pick_color() -> None:
        try:
            r, g, b, a = parse_rgba(var.get())
        except ValueError:
            r, g, b, a = 255, 255, 255, 255
        _rgb, hex_str = colorchooser.askcolor(color=f"#{r:02x}{g:02x}{b:02x}", parent=parent)
        if not hex_str:
            return
        new_alpha = _ask_opacity(parent, hex_str, a)
        if new_alpha is None:
            return
        var.set(f"{hex_str.upper()}{new_alpha:02X}")
        _refresh_swatch()

    swatch = tk.Button(parent, text="  ", relief="raised", bd=1, width=3, cursor="hand2", command=_pick_color)
    swatch.grid(row=row, column=2, sticky="w", padx=4, pady=2)

    var.trace_add("write", _refresh_swatch)
    _refresh_swatch()
    return var
