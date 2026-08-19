"""Word-wrap text to a fixed pixel width at a fixed font size.

Runs continuously as new transcript text arrives (unlike layout.py's font-size
derivation, which only runs when layout-affecting config changes). Long single
words that don't fit even alone are force-split at the character level rather
than shrinking the font further, since font size is a layout-only concept here.
"""
from __future__ import annotations

from typing import Protocol


class TextWidthProvider(Protocol):
    def text_width(self, text: str, font_path: str, size_px: int) -> float:
        """Return the rendered pixel width of `text` at `size_px`."""
        ...


def _split_long_word(word: str, font_path: str, size_px: int, box_width: float, measure: TextWidthProvider) -> list[str]:
    """Character-level split of a single word that doesn't fit box_width on its own."""
    if not word:
        return [word]
    chunks: list[str] = []
    current = ""
    for ch in word:
        candidate = current + ch
        if current and measure.text_width(candidate, font_path, size_px) > box_width:
            chunks.append(current)
            current = ch
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [word]


def wrap_text(
    text: str,
    font_path: str,
    size_px: int,
    box_width: float,
    measure: TextWidthProvider,
) -> list[str]:
    """Greedy word-wrap `text` into lines that fit within `box_width` pixels."""
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""

    def emit(line: str) -> None:
        if line:
            lines.append(line)

    for word in words:
        # Word alone doesn't fit even in an empty line: force character-split.
        if measure.text_width(word, font_path, size_px) > box_width:
            emit(current)
            current = ""
            for chunk in _split_long_word(word, font_path, size_px, box_width, measure):
                lines.append(chunk)
            continue

        candidate = f"{current} {word}" if current else word
        if measure.text_width(candidate, font_path, size_px) <= box_width:
            current = candidate
        else:
            emit(current)
            current = word

    emit(current)
    return lines
