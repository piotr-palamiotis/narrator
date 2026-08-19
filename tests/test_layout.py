import pytest

from narrator.captions.layout import (
    compute_line_geometry,
    compute_per_line_height,
    compute_text_box,
    derive_font_size_px,
    line_y_positions,
)
from narrator.config.schema import CanvasConfig, LayoutConfig, StyleConfig


class FakeMetrics:
    """Deterministic fake: ascent+descent == size_px (i.e. metrics scale 1:1
    with font size), so expected results are easy to compute by hand.
    """

    def ascent_descent(self, font_path: str, size_px: int) -> tuple[int, int]:
        return int(size_px * 0.8), int(size_px * 0.2)


def make_canvas(width=1000, height=1000):
    return CanvasConfig(width_px=width, height_px=height, background_color="#00000000")


def test_compute_text_box_basic():
    canvas = make_canvas(1000, 1000)
    layout = LayoutConfig(
        max_lines=2, margin_top_pct=10, margin_bottom_pct=10, margin_left_pct=20, margin_right_pct=20
    )
    box = compute_text_box(canvas, layout)
    assert box.top == 100
    assert box.bottom == 900
    assert box.left == 200
    assert box.right == 800
    assert box.height == 800
    assert box.width == 600


def test_per_line_height_with_interline():
    canvas = make_canvas(1000, 1000)
    layout = LayoutConfig(max_lines=3, margin_top_pct=0, margin_bottom_pct=0, interline_pct=1.0)
    box = compute_text_box(canvas, layout)
    per_line, gap = compute_per_line_height(canvas, layout, box)
    # box height = 1000, gap = 1% of 1000 = 10px, 2 gaps -> 20px total
    assert gap == 10
    assert per_line == pytest.approx((1000 - 20) / 3)


def test_single_line_has_no_gap():
    canvas = make_canvas(1000, 1000)
    layout = LayoutConfig(max_lines=1, margin_top_pct=0, margin_bottom_pct=0, interline_pct=5.0)
    box = compute_text_box(canvas, layout)
    per_line, gap = compute_per_line_height(canvas, layout, box)
    assert per_line == 1000  # no gaps subtracted since gap_count = max(0, 0) = 0


def test_zero_interline():
    canvas = make_canvas(1000, 1000)
    layout = LayoutConfig(max_lines=4, margin_top_pct=0, margin_bottom_pct=0, interline_pct=0.0)
    box = compute_text_box(canvas, layout)
    per_line, gap = compute_per_line_height(canvas, layout, box)
    assert gap == 0
    assert per_line == 250


def test_interline_consumes_whole_box_raises():
    canvas = make_canvas(1000, 1000)
    layout = LayoutConfig(max_lines=5, margin_top_pct=0, margin_bottom_pct=0, interline_pct=30.0)
    box = compute_text_box(canvas, layout)
    with pytest.raises(ValueError):
        compute_per_line_height(canvas, layout, box)


def test_derive_font_size_matches_target():
    style = StyleConfig()
    metrics = FakeMetrics()
    size = derive_font_size_px(100.0, style, metrics)
    # The binary search must pick the LARGEST size that still fits: confirm
    # `size` fits and `size + 1` does not, rather than hardcoding a value
    # (int-truncated metrics mean the exact boundary isn't a round number).
    ascent, descent = metrics.ascent_descent(style.font_family, size)
    assert (ascent + descent) <= 100.0
    next_ascent, next_descent = metrics.ascent_descent(style.font_family, size + 1)
    assert (next_ascent + next_descent) > 100.0


def test_derive_font_size_clamps_to_floor_when_box_too_small():
    style = StyleConfig()
    metrics = FakeMetrics()
    size = derive_font_size_px(0.0, style, metrics)
    assert size == 1  # clamped to the 1px floor rather than going lower or crashing


def test_compute_line_geometry_end_to_end():
    canvas = make_canvas(1920, 1080)
    layout = LayoutConfig(max_lines=2, margin_top_pct=10, margin_bottom_pct=5, interline_pct=1.0)
    style = StyleConfig()
    geometry = compute_line_geometry(canvas, layout, style, FakeMetrics())
    assert geometry.max_lines == 2
    assert geometry.font_size_px > 0
    assert geometry.box.height == pytest.approx(1080 * 0.85)


def test_line_y_positions_bottom_alignment_when_partial():
    canvas = make_canvas(1000, 1000)
    layout = LayoutConfig(max_lines=3, margin_top_pct=0, margin_bottom_pct=0, interline_pct=0)
    style = StyleConfig()
    geometry = compute_line_geometry(canvas, layout, style, FakeMetrics())
    positions_full = line_y_positions(geometry, 3, "bottom")
    positions_partial = line_y_positions(geometry, 1, "bottom")
    # With bottom alignment, a single active line should occupy the LAST slot.
    assert positions_partial == [positions_full[-1]]


def test_line_y_positions_top_alignment_when_partial():
    canvas = make_canvas(1000, 1000)
    layout = LayoutConfig(max_lines=3, margin_top_pct=0, margin_bottom_pct=0, interline_pct=0)
    style = StyleConfig()
    geometry = compute_line_geometry(canvas, layout, style, FakeMetrics())
    positions_full = line_y_positions(geometry, 3, "top")
    positions_partial = line_y_positions(geometry, 1, "top")
    assert positions_partial == [positions_full[0]]
