"""Small reusable Tkinter form widgets shared across gui/app.py's config tabs."""
from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser, ttk

from narrator.config.schema import parse_rgba


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


def checkbox_field(parent: tk.Widget, row: int, label: str, initial: bool) -> tk.BooleanVar:
    var = tk.BooleanVar(value=initial)
    ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2)
    return var


def rgba_color_field(parent: tk.Widget, row: int, label: str, initial: str) -> tk.StringVar:
    """A hex-text entry (source of truth, editable directly as '#RRGGBBAA')
    plus a swatch button that opens the system color picker for RGB and
    preserves the existing alpha value (Tkinter's built-in chooser has no
    alpha channel support).
    """
    labeled_row(parent, row, label)
    var = tk.StringVar(value=initial)
    entry = ttk.Entry(parent, textvariable=var, width=10)
    entry.grid(row=row, column=1, sticky="w", padx=4, pady=2)

    swatch = tk.Label(parent, text="  ", relief="sunken", width=3)
    swatch.grid(row=row, column=2, sticky="w", padx=4, pady=2)

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
        _rgb, hex_str = colorchooser.askcolor(color=f"#{r:02x}{g:02x}{b:02x}")
        if hex_str:
            var.set(f"{hex_str.upper()}{a:02X}")
            _refresh_swatch()

    ttk.Button(parent, text="Pick", command=_pick_color, width=5).grid(row=row, column=3, sticky="w", padx=4, pady=2)
    var.trace_add("write", _refresh_swatch)
    _refresh_swatch()
    return var
