from narrator.app_controller import AppController
from narrator.asr.events import CaptionEvent, CaptionEventType


class FakeRenderer:
    def __init__(self) -> None:
        self.final_texts: list[str] = []
        self.partial_preview: str | None = None
        self.cleared = False

    def append_final_text(self, text: str) -> None:
        self.final_texts.append(text)

    def set_partial_preview(self, text: str) -> None:
        self.partial_preview = text

    def clear(self) -> None:
        self.cleared = True
        self.final_texts = []
        self.partial_preview = None


def make_controller() -> tuple[AppController, FakeRenderer]:
    # AppController.__init__ wires up real audio/NDI/config-watching infra;
    # this exercises only the caption-mode/manual/clear logic, which only
    # needs `_renderer` and `_caption_mode`, so we bypass __init__ entirely.
    controller = AppController.__new__(AppController)
    renderer = FakeRenderer()
    controller._renderer = renderer
    controller._caption_mode = "auto"
    controller.resolved_asr_backend = "unresolved"
    return controller, renderer


def test_auto_mode_applies_asr_final_and_partial_events():
    controller, renderer = make_controller()

    controller._handle_caption_event(CaptionEvent(CaptionEventType.PARTIAL, "hell"))
    controller._handle_caption_event(CaptionEvent(CaptionEventType.FINAL, "hello"))

    assert renderer.final_texts == ["hello"]


def test_manual_mode_ignores_asr_final_partial_and_clear_events():
    controller, renderer = make_controller()
    controller.set_caption_mode("manual")

    controller._handle_caption_event(CaptionEvent(CaptionEventType.PARTIAL, "hell"))
    controller._handle_caption_event(CaptionEvent(CaptionEventType.FINAL, "hello"))
    controller._handle_caption_event(CaptionEvent(CaptionEventType.CLEAR))

    assert renderer.final_texts == []
    assert renderer.partial_preview is None
    assert not renderer.cleared


def test_manual_mode_still_applies_status_events():
    controller, renderer = make_controller()
    controller.set_caption_mode("manual")

    controller._handle_caption_event(CaptionEvent(CaptionEventType.STATUS, "CPU"))

    assert controller.resolved_asr_backend == "CPU"


def test_submit_manual_text_appends_and_clears_partial_preview():
    controller, renderer = make_controller()
    controller.set_caption_mode("manual")

    controller.submit_manual_text("  operator typed this  ")

    assert renderer.final_texts == ["operator typed this"]
    assert renderer.partial_preview == ""


def test_submit_manual_text_ignores_blank_input():
    controller, renderer = make_controller()

    controller.submit_manual_text("   ")

    assert renderer.final_texts == []


def test_clear_captions_works_in_either_mode():
    controller, renderer = make_controller()

    controller.clear_captions()

    assert renderer.cleared is True


def test_set_caption_mode_rejects_unknown_mode():
    controller, _renderer = make_controller()
    try:
        controller.set_caption_mode("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an invalid caption mode")
