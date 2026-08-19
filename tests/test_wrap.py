from narrator.captions.buffer import CaptionBuffer
from narrator.captions.wrap import wrap_text


class FixedWidthMeasure:
    """1 char == 1 pixel unit, deterministic and easy to reason about."""

    def text_width(self, text: str, font_path: str, size_px: int) -> float:
        return len(text)


def test_wrap_basic_word_boundaries():
    lines = wrap_text("the quick brown fox", "font", 10, box_width=10, measure=FixedWidthMeasure())
    assert lines == ["the quick", "brown fox"]


def test_wrap_empty_text():
    assert wrap_text("", "font", 10, box_width=10, measure=FixedWidthMeasure()) == []
    assert wrap_text("   ", "font", 10, box_width=10, measure=FixedWidthMeasure()) == []


def test_wrap_single_long_word_forces_char_split():
    lines = wrap_text("supercalifragilistic", "font", 10, box_width=5, measure=FixedWidthMeasure())
    assert "".join(lines) == "supercalifragilistic"
    assert all(len(line) <= 5 for line in lines)


def test_wrap_long_word_mixed_with_normal_words():
    lines = wrap_text("go supercalifragilistic now", "font", 10, box_width=5, measure=FixedWidthMeasure())
    assert "".join(lines) == "gosupercalifragilisticnow" or lines[0] == "go"
    # "go" fits, long word forces split, "now" fits on its own line
    assert lines[0] == "go"
    assert lines[-1] == "now"


def test_wrap_word_exactly_fits_box_width():
    lines = wrap_text("hello", "font", 10, box_width=5, measure=FixedWidthMeasure())
    assert lines == ["hello"]


def test_caption_buffer_rolls_last_n_lines():
    buf = CaptionBuffer(max_lines=2)
    buf.configure_geometry("font", 10, box_width=10, measure=FixedWidthMeasure())
    buf.append_final_text("the quick brown fox jumps")
    visible = buf.visible_lines()
    assert len(visible) <= 2


def test_caption_buffer_clear():
    buf = CaptionBuffer(max_lines=2)
    buf.configure_geometry("font", 10, box_width=10, measure=FixedWidthMeasure())
    buf.append_final_text("hello world")
    assert buf.visible_lines() != []
    buf.clear()
    assert buf.visible_lines() == []


def test_caption_buffer_incremental_append_continues_pending_line():
    buf = CaptionBuffer(max_lines=5)
    buf.configure_geometry("font", 10, box_width=11, measure=FixedWidthMeasure())
    buf.append_final_text("hello")
    buf.append_final_text("world")
    # "hello world" is 11 chars, which fits box_width=11 -> should combine onto one line.
    assert buf.visible_lines() == ["hello world"]


def test_caption_buffer_rewraps_on_geometry_change():
    buf = CaptionBuffer(max_lines=5)
    buf.configure_geometry("font", 10, box_width=20, measure=FixedWidthMeasure())
    buf.append_final_text("the quick brown fox")
    assert buf.visible_lines() == ["the quick brown fox"]
    # Shrinking box_width should force a rewrap of existing text into more lines.
    buf.configure_geometry("font", 10, box_width=10, measure=FixedWidthMeasure())
    visible = buf.visible_lines()
    assert len(visible) > 1
    assert " ".join(visible) == "the quick brown fox"
