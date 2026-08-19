"""Manual verification script (build phase 6/7): renders a few caption
scenarios to PNG files under scripts/output/ so layout, wrapping, and style
overrides can be checked visually without any audio/ASR/NDI dependency.

Usage: python scripts/render_preview.py
"""
from __future__ import annotations

from pathlib import Path

from narrator.config.schema import (
    BackgroundBandStyle,
    CanvasConfig,
    LayoutConfig,
    ShadowStyle,
    StrokeStyle,
    StyleConfig,
    StyleOverrides,
)
from narrator.render.canvas import CaptionRenderer

OUT_DIR = Path(__file__).with_name("output")


def save(renderer: CaptionRenderer, name: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    # Composite over a checkerboard so alpha/transparency is visible in the PNG.
    frame = renderer.get_frame()
    checker = _checkerboard(frame.size)
    checker.alpha_composite(frame)
    checker.save(path)
    print(f"wrote {path} (font_size_px={renderer.resolved_font_size_px})")


def _checkerboard(size, tile=20):
    from PIL import Image

    w, h = size
    img = Image.new("RGBA", size, (255, 255, 255, 255))
    px = img.load()
    for y in range(h):
        for x in range(w):
            if (x // tile + y // tile) % 2 == 0:
                px[x, y] = (200, 200, 200, 255)
    return img


def main() -> None:
    canvas = CanvasConfig(width_px=1280, height_px=360, background_color="#00000000")

    # Scenario 1: default 2-line pill style, Polish text with diacritics, long enough to wrap.
    layout = LayoutConfig(max_lines=2, margin_top_pct=10, margin_bottom_pct=10, margin_left_pct=5, margin_right_pct=5, interline_pct=2.0)
    style = StyleConfig()
    r1 = CaptionRenderer(canvas, layout, style)
    r1.append_final_text("Zażółć gęślą jaźń, to jest test polskich napisów na żywo strumieniowanych przez NDI.")
    save(r1, "01_default_pill")

    # Scenario 2: high-contrast "yellow box" broadcast look, 3 lines, no shadow.
    layout2 = LayoutConfig(max_lines=3, margin_top_pct=5, margin_bottom_pct=5, margin_left_pct=5, margin_right_pct=5, interline_pct=1.5)
    style2 = StyleConfig(
        font_color="#000000FF",
        overrides=StyleOverrides(
            background_band=BackgroundBandStyle(enabled=True, shape="rectangle", color="#FFD400FF", padding_pct=0.8),
            shadow=ShadowStyle(enabled=False),
            stroke=StrokeStyle(enabled=False),
        ),
    )
    r2 = CaptionRenderer(canvas, layout2, style2)
    r2.append_final_text("To jest przykladowy dlugi tekst ktory powinien zawinac sie na kilka linii aby sprawdzic przewijanie napisow.")
    save(r2, "02_yellow_box_3lines")

    # Scenario 3: shadow + stroke, no band, 1 line, centered text.
    layout3 = LayoutConfig(max_lines=1, margin_top_pct=30, margin_bottom_pct=30, margin_left_pct=5, margin_right_pct=5, text_align="center")
    style3 = StyleConfig(
        overrides=StyleOverrides(
            background_band=BackgroundBandStyle(enabled=False),
            shadow=ShadowStyle(enabled=True, offset_x_pct=0.5, offset_y_pct=0.5, blur_px=6),
            stroke=StrokeStyle(enabled=True, width_px=3),
        )
    )
    r3 = CaptionRenderer(canvas, layout3, style3)
    r3.append_final_text("Jedna linia, wysoki kontrast")
    save(r3, "03_shadow_stroke_centered")

    # Scenario 4: scrolling behavior -- feed more lines than max_lines, confirm only last N shown.
    layout4 = LayoutConfig(max_lines=2, margin_top_pct=10, margin_bottom_pct=10, margin_left_pct=5, margin_right_pct=5, interline_pct=1.0)
    r4 = CaptionRenderer(canvas, layout4, StyleConfig())
    for i in range(1, 6):
        r4.append_final_text(f"To jest linia numer {i} ktora powinna przewijac sie w gore.")
    save(r4, "04_scrolling_last_n")


if __name__ == "__main__":
    main()
