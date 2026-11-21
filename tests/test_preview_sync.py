"""PreviewSync's sample caption: shown in the preview only, never on air."""
from narrator.config.schema import CanvasConfig, LayoutConfig, StyleConfig
from narrator.render.canvas import CaptionRenderer
from narrator.render.preview_sync import PreviewSync


def _configs():
    return CanvasConfig(width_px=320, height_px=180), LayoutConfig(), StyleConfig()


def test_sample_text_is_previewed_without_touching_the_live_renderer():
    canvas, layout, style = _configs()
    live = CaptionRenderer(canvas, layout, style)
    sync = PreviewSync(canvas, layout, style)

    # No sample: identical settings hand back the live renderer's own frame.
    assert sync.refresh(canvas, layout, style, live) is live.get_frame()

    frame = sync.refresh(canvas, layout, style, live, sample_text="Przykładowy napis")
    assert frame is not live.get_frame()
    assert frame.getbbox() is not None  # something was drawn
    assert live.snapshot_text() == ""  # nothing went on air

    # Turning the sample off goes back to the live frame.
    assert sync.refresh(canvas, layout, style, live) is live.get_frame()
