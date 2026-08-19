"""Per-line style-override compositing: background band -> shadow -> stroke+fill text.

Three composable blocks cover the requested legibility effects: a solid-color
"box" or rounded "pill" behind each line (the classic broadcast caption look,
also usable for a high-contrast yellow-boxed style), a blurred drop shadow,
and a text outline/stroke. They are drawn in that order, back to front, onto
the caller-supplied RGBA canvas for each active display line.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from narrator.config.schema import StyleConfig, TextAlign, parse_rgba


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, stroke_width: int):
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height, bbox


def _horizontal_anchor(text_align: TextAlign, box_left: float, box_right: float, text_width: float) -> float:
    if text_align == TextAlign.CENTER:
        return box_left + (box_right - box_left - text_width) / 2.0
    if text_align == TextAlign.RIGHT:
        return box_right - text_width
    return box_left  # LEFT


def composite_line(
    canvas: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    box_left: float,
    box_right: float,
    slot_top: float,
    slot_height: float,
    canvas_height_px: int,
    style: StyleConfig,
    text_align: TextAlign,
) -> None:
    """Draw one caption line, with all enabled style overrides, onto `canvas` in place."""
    overrides = style.overrides
    stroke_width = overrides.stroke.width_px if overrides.stroke.enabled else 0

    draw = ImageDraw.Draw(canvas)
    text_width, text_height, bbox = _measure(draw, text, font, stroke_width)

    x = _horizontal_anchor(text_align, box_left, box_right, text_width)
    y = slot_top + (slot_height - text_height) / 2.0
    # Compensate for the font's natural top/left bearing so (x, y) is the
    # actual top-left corner of the rendered glyphs, not the layout origin.
    draw_x = x - bbox[0]
    draw_y = y - bbox[1]

    if overrides.background_band.enabled:
        band = overrides.background_band
        padding_px = canvas_height_px * band.padding_pct / 100.0
        rect = (
            x - padding_px,
            slot_top,
            x + text_width + padding_px,
            slot_top + slot_height,
        )
        color = parse_rgba(band.color)
        if band.shape.value == "pill":
            radius = canvas_height_px * band.corner_radius_pct / 100.0
            draw.rounded_rectangle(rect, radius=radius, fill=color)
        else:
            draw.rectangle(rect, fill=color)

    if overrides.shadow.enabled:
        shadow = overrides.shadow
        shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        offset_x = canvas.width * shadow.offset_x_pct / 100.0
        offset_y = canvas_height_px * shadow.offset_y_pct / 100.0
        shadow_draw.text(
            (draw_x + offset_x, draw_y + offset_y),
            text,
            font=font,
            fill=parse_rgba(shadow.color),
        )
        if shadow.blur_px > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow.blur_px))
        canvas.alpha_composite(shadow_layer)
        draw = ImageDraw.Draw(canvas)  # re-bind after alpha_composite mutated canvas

    if overrides.stroke.enabled:
        draw.text(
            (draw_x, draw_y),
            text,
            font=font,
            fill=parse_rgba(style.font_color),
            stroke_width=overrides.stroke.width_px,
            stroke_fill=parse_rgba(overrides.stroke.color),
        )
    else:
        draw.text((draw_x, draw_y), text, font=font, fill=parse_rgba(style.font_color))
