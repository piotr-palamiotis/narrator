"""Font loading/caching and the Pillow-backed adapters for the geometry protocols
used by narrator.captions.layout (FontMetricsProvider) and
narrator.captions.wrap (TextWidthProvider).
"""
from __future__ import annotations

from pathlib import Path

from PIL import ImageFont

_FALLBACK_FONT = str(Path(__file__).resolve().parents[3] / "assets" / "fonts" / "NotoSans-Regular.ttf")

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def load_font(font_path: str, size_px: int) -> ImageFont.FreeTypeFont:
    key = (font_path, size_px)
    cached = _font_cache.get(key)
    if cached is not None:
        return cached
    try:
        font = ImageFont.truetype(font_path, size_px)
    except OSError:
        font = ImageFont.truetype(_FALLBACK_FONT, size_px)
    _font_cache[key] = font
    return font


class PillowFontMetrics:
    """Implements narrator.captions.layout.FontMetricsProvider."""

    def ascent_descent(self, font_path: str, size_px: int) -> tuple[int, int]:
        font = load_font(font_path, size_px)
        return font.getmetrics()


class PillowTextWidth:
    """Implements narrator.captions.wrap.TextWidthProvider."""

    def text_width(self, text: str, font_path: str, size_px: int) -> float:
        font = load_font(font_path, size_px)
        return font.getlength(text)
