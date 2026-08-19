"""Pure geometry functions: text-box computation and font-size derivation.

Font size is never a config value. It is derived once from canvas
dimensions, margins, line count, and interline percentage, and is re-derived
only when one of those layout-affecting values changes -- never per
transcript update, so captions don't visually resize as words stream in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from narrator.config.schema import CanvasConfig, LayoutConfig, StyleConfig


@dataclass(frozen=True)
class TextBox:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass(frozen=True)
class LineGeometry:
    """Result of the layout algorithm: everything the renderer needs to place text."""

    box: TextBox
    font_size_px: int
    per_line_height: float
    gap_px: float
    max_lines: int


class FontMetricsProvider(Protocol):
    """Abstraction over Pillow's `ImageFont` so layout.py has no hard Pillow
    dependency and stays unit-testable with a fake.
    """

    def ascent_descent(self, font_path: str, size_px: int) -> tuple[int, int]:
        """Return (ascent, descent) in pixels for `font_path` at `size_px`."""
        ...


def compute_text_box(canvas: CanvasConfig, layout: LayoutConfig) -> TextBox:
    width, height = canvas.width_px, canvas.height_px
    top = height * layout.margin_top_pct / 100.0
    bottom = height * (1.0 - layout.margin_bottom_pct / 100.0)
    left = width * layout.margin_left_pct / 100.0
    right = width * (1.0 - layout.margin_right_pct / 100.0)
    return TextBox(left=left, top=top, right=right, bottom=bottom)


def compute_per_line_height(
    canvas: CanvasConfig, layout: LayoutConfig, box: TextBox
) -> tuple[float, float]:
    """Returns (per_line_height, gap_px). Raises ValueError if the interline
    gaps alone consume the whole box (should already be rejected by config
    validation, but layout.py is defensive on its own since it may be called
    with ad-hoc geometry in tests).
    """
    gap_px = canvas.height_px * layout.interline_pct / 100.0
    gap_count = max(layout.max_lines - 1, 0)
    total_gap_px = gap_px * gap_count
    usable_height = box.height - total_gap_px
    if usable_height <= 0:
        raise ValueError(
            f"interline gaps ({total_gap_px:.1f}px) consume the entire text box "
            f"height ({box.height:.1f}px)"
        )
    return usable_height / layout.max_lines, gap_px


MIN_FONT_PX_FLOOR = 1
MAX_FONT_PX_CEILING = 400


def derive_font_size_px(
    per_line_height: float,
    style: StyleConfig,
    metrics: FontMetricsProvider,
    max_font_px: int = MAX_FONT_PX_CEILING,
) -> int:
    """Binary-search the largest integer font size whose (ascent + descent)
    fits within `per_line_height`. Clamps to `MIN_FONT_PX_FLOOR` if even the
    smallest size doesn't fit -- an impossible layout (from the operator's
    other geometry settings) degrades to a 1px font rather than crashing.
    """
    floor = MIN_FONT_PX_FLOOR
    target = per_line_height

    if target <= 0:
        return floor

    lo, hi = floor, max(floor, max_font_px)
    best = floor

    def fits(size: int) -> bool:
        ascent, descent = metrics.ascent_descent(style.font_family, size)
        return (ascent + descent) <= target

    if fits(lo):
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
    # else: even 1px doesn't fit -- clamp to the floor rather than crash;
    # an impossibly tight layout is the operator's other settings' fault.

    return best


def compute_line_geometry(
    canvas: CanvasConfig,
    layout: LayoutConfig,
    style: StyleConfig,
    metrics: FontMetricsProvider,
) -> LineGeometry:
    box = compute_text_box(canvas, layout)
    per_line_height, gap_px = compute_per_line_height(canvas, layout, box)
    font_size_px = derive_font_size_px(per_line_height, style, metrics)
    return LineGeometry(
        box=box,
        font_size_px=font_size_px,
        per_line_height=per_line_height,
        gap_px=gap_px,
        max_lines=layout.max_lines,
    )


def line_y_positions(geometry: LineGeometry, active_line_count: int, vertical_align: str) -> list[float]:
    """Top-y pixel offset (within the canvas) for each of `active_line_count`
    active lines (<= geometry.max_lines), honoring vertical_align_within_box
    when fewer than max_lines are currently populated.
    """
    n = geometry.max_lines
    slot_top = [
        geometry.box.top + i * (geometry.per_line_height + geometry.gap_px) for i in range(n)
    ]
    if active_line_count >= n:
        return slot_top
    if vertical_align == "top":
        return slot_top[:active_line_count]
    if vertical_align == "center":
        empty = n - active_line_count
        start = empty // 2
        return slot_top[start : start + active_line_count]
    # bottom (default): fill the last `active_line_count` slots
    return slot_top[n - active_line_count :]
