import numpy as np

from narrator.asr.events import CaptionEventType
from narrator.asr.streaming_decoder import StreamingDecoder
from narrator.config.schema import AsrConfig


class FakeVad:
    """Returns exactly one probability per feed() call, controlled by the test.
    Each call is treated as one ~32ms VAD window for silence/speech timing.
    """

    def __init__(self) -> None:
        self.level = 0.0

    def process(self, frame):
        return [self.level]


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self.resolved_device_description = "CPU (fake)"
        self._responses = iter(responses)

    def transcribe(self, audio):
        text = next(self._responses)
        return [(0.0, 1.0, text)] if text else []


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


FRAME = np.zeros(480, dtype=np.float32)  # 30ms @ 16kHz, content irrelevant (FakeVad ignores it)


def make_decoder(responses, config=None, events=None):
    events = events if events is not None else []
    config = config or AsrConfig(step_ms=100, agreement_passes=2, end_silence_ms=300, clear_after_silence_seconds=5)
    clock = FakeClock()
    decoder = StreamingDecoder(
        FakeBackend(responses),
        FakeVad(),
        config,
        on_event=events.append,
        clock=clock,
        min_decode_seconds=0.01,  # tiny: isolates step_ms timing from window-duration gating in tests
    )
    return decoder, clock, events


def feed_decode_step(decoder, clock, config):
    """Advance the clock just past `step_ms` and feed one speech frame,
    triggering exactly one decode step (and, on the very first call, entering
    speech). Deterministic alternative to simulating real audio duration.
    """
    clock.advance(config.step_ms / 1000.0 + 0.001)
    decoder._vad.level = 1.0
    decoder.feed(FRAME)


def feed_silence(decoder, n_frames):
    decoder._vad.level = 0.0
    for _ in range(n_frames):
        decoder.feed(FRAME)


def test_status_event_emitted_on_construction():
    _, _, events = make_decoder(["hello"])
    assert events[0].type == CaptionEventType.STATUS
    assert events[0].text == "CPU (fake)"


def test_commits_prefix_after_agreement_passes():
    config = AsrConfig(step_ms=100, agreement_passes=2, end_silence_ms=300, clear_after_silence_seconds=5)
    decoder, clock, events = make_decoder(
        ["hello world", "hello world there", "hello world there now"], config=config
    )

    # Step 1: decode "hello world". 1 hypothesis in history (< agreement_passes=2) -> no commit yet.
    feed_decode_step(decoder, clock, config)
    # Step 2: decode "hello world there". Agreed prefix of the last 2 hypotheses
    #   (["hello","world"], ["hello","world","there"]) is 2 -> commit "hello world".
    feed_decode_step(decoder, clock, config)
    # Step 3: decode "hello world there now". Agreed prefix of the last 2 hypotheses
    #   (["hello","world","there"], ["hello","world","there","now"]) is 3 -> commit "there".
    feed_decode_step(decoder, clock, config)

    finals = [e for e in events if e.type == CaptionEventType.FINAL]
    assert [f.text for f in finals] == ["hello world", "there"]


def test_force_finalizes_on_end_silence():
    config = AsrConfig(step_ms=100, agreement_passes=99, end_silence_ms=100, clear_after_silence_seconds=5)
    decoder, clock, events = make_decoder(["partial hypothesis", "partial hypothesis final"], config=config)

    feed_decode_step(decoder, clock, config)  # decode step; agreement_passes=99 so nothing commits yet
    feed_silence(decoder, 5)  # 5 * 32ms = 160ms >= end_silence_ms=100ms -> force-finalize

    finals = [e for e in events if e.type == CaptionEventType.FINAL]
    assert len(finals) == 1
    assert finals[0].text == "partial hypothesis final"


def test_clear_emitted_after_prolonged_silence():
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=50, clear_after_silence_seconds=1.0)
    decoder, clock, events = make_decoder(["hi"], config=config)

    feed_decode_step(decoder, clock, config)  # decode "hi"; agreement_passes=1 commits immediately
    feed_silence(decoder, 2)  # 2 * 32ms = 64ms >= end_silence_ms=50ms -> force-finalize (no-op text-wise)
    assert any(e.type == CaptionEventType.FINAL for e in events)

    # Now well past clear_after_silence_seconds=1.0 with no further speech.
    clock.advance(2.0)
    feed_silence(decoder, 1)
    assert any(e.type == CaptionEventType.CLEAR for e in events)


def test_no_duplicate_clear_events():
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=50, clear_after_silence_seconds=0.5)
    decoder, clock, events = make_decoder(["hi"], config=config)

    feed_decode_step(decoder, clock, config)
    feed_silence(decoder, 2)
    clock.advance(1.0)
    feed_silence(decoder, 3)
    clock.advance(1.0)
    feed_silence(decoder, 3)
    clear_events = [e for e in events if e.type == CaptionEventType.CLEAR]
    assert len(clear_events) == 1
