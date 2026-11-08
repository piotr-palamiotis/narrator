"""The segmenter decoder (narrator.asr.segmenter, `asr.decoder: segmenter`):
VAD-cut segments decoded once (FINAL), optional whole-segment peeks
(PARTIAL), and the guards it shares with local agreement -- no-loss
rollover, repetition collapse, boundary dedupe, whitelist lock, silence
timers, drain-then-tick ingest. GPU-free: fake VAD, backend and clock."""
from __future__ import annotations

import numpy as np
import pytest

from narrator.asr.backend import DevicePolicyError, SegmentResult
from narrator.asr.decoder_common import create_decoder
from narrator.asr.events import CaptionEventType
from narrator.asr.segmenter import Segmenter
from narrator.asr.streaming_decoder import StreamingDecoder
from narrator.asr.text_settings import TextSettings
from narrator.captions.blocklist import Blocklist
from narrator.config.schema import AsrConfig

FRAME_SAMPLES = 480  # 30 ms @ 16 kHz
FRAME = np.zeros(FRAME_SAMPLES, dtype=np.float32)
CHUNK = 8000  # 0.5 s of audio per word in the word-encoded fake


class FakeVad:
    """One probability per feed() call (one ~32 ms window), set by the test."""

    def __init__(self) -> None:
        self.level = 0.0
        self.calls = 0
        self.resets = 0

    def process(self, frame):
        self.calls += 1
        return [self.level]

    def reset_state(self) -> None:
        self.resets += 1


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SegBackend:
    """decode_segment returns scripted texts (or `text_fn(audio)`); every
    call is recorded. `peek_cost`/`final_cost` advance the clock, like a
    real decode does on the live monotonic clock."""

    resolved_device_description = "CPU (fake)"

    def __init__(self, responses=None, text_fn=None, clock=None) -> None:
        self._responses = iter(responses or [])
        self._text_fn = text_fn
        self.clock = clock
        self.calls: list[dict] = []
        self.detect_calls = 0
        self.detected = [("pl", 0.99, 0.9)]  # cycled through, last one repeats
        self.peek_cost = 0.0
        self.final_cost = 0.0
        self.fail_with: Exception | None = None

    def decode_segment(self, audio, *, prompt=None, language=None, drop_trailing_period=False, peek=False):
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append(
            {"seconds": len(audio) / 16000, "prompt": prompt, "language": language,
             "drop_trailing_period": drop_trailing_period, "peek": peek}
        )
        if self.clock is not None:
            self.clock.advance(self.peek_cost if peek else self.final_cost)
        text = self._text_fn(audio) if self._text_fn else next(self._responses, "")
        return SegmentResult(text=text, language=language)

    def detect_language(self, audio):
        self.detect_calls += 1
        result = self.detected[min(self.detect_calls - 1, len(self.detected) - 1)]
        return result

    @property
    def finals(self):
        return [c for c in self.calls if not c["peek"]]

    @property
    def peeks(self):
        return [c for c in self.calls if c["peek"]]


def word_text(audio) -> str:
    """Word-encoded audio -> words: every 0.5 s chunk of the source holds
    (id + 1) / 1000; a word is heard when at least half its chunk is in the
    segment, so a word cut by a boundary lands in exactly one segment."""
    ids = np.rint(np.asarray(audio) * 1000).astype(int) - 1
    values, first, counts = np.unique(ids, return_index=True, return_counts=True)
    heard = [int(v) for f, v, c in sorted(zip(first, values, counts)) if v >= 0 and c >= CHUNK // 2]
    return " ".join(f"w{i}" for i in heard)


def encoded_speech(seconds: float) -> np.ndarray:
    n = int(seconds * 16000)
    return ((np.arange(n) // CHUNK) + 1).astype(np.float32) / 1000.0


def seg_config(**overrides) -> AsrConfig:
    values = dict(
        decoder="segmenter",
        language_whitelist=["pl"],
        end_silence_ms=300,
        max_segment_ms=15000,
        prune_history_after_silence_seconds=5,
        show_partial_captions=False,
        context_sentences=0,
    )
    values.update(overrides)
    return AsrConfig(**values)


def make(config=None, responses=None, text_fn=None, text_settings=None):
    config = config or seg_config()
    clock = FakeClock()
    backend = SegBackend(responses, text_fn, clock)
    vad = FakeVad()
    events = []
    seg = Segmenter(backend, vad, config, events.append, clock=clock, text_settings=text_settings)
    return seg, backend, vad, clock, events


def run_frames(seg, clock, n, level=None, audio=None):
    """Feed n frames (zeros, or consecutive slices of `audio`) drain-then-
    tick like the worker; `level` sets the fake VAD for all of them."""
    if level is not None:
        seg._vad.level = level
    for k in range(n):
        frame = FRAME if audio is None else audio[k * FRAME_SAMPLES : (k + 1) * FRAME_SAMPLES]
        clock.advance(0.03)
        seg.feed(frame, run_decode_step=False)
        seg.tick()


def run_audio(seg, clock, audio, levels=None):
    for k, i in enumerate(range(0, len(audio), FRAME_SAMPLES)):
        if levels is not None:
            seg._vad.level = levels(k)
        clock.advance(0.03)
        seg.feed(audio[i : i + FRAME_SAMPLES], run_decode_step=False)
        seg.tick()


def of_type(events, kind):
    return [e for e in events if e.type is kind]


def final_texts(events):
    return [e.text for e in of_type(events, CaptionEventType.FINAL)]


def final_words(events):
    return " ".join(final_texts(events)).split()


# ---------------------------------------------------------------------------
# selection and the basic segment lifecycle
# ---------------------------------------------------------------------------


def test_create_decoder_picks_the_configured_decoder():
    backend = SegBackend()
    seg = create_decoder(backend, FakeVad(), seg_config(), lambda e: None)
    assert isinstance(seg, Segmenter)
    la = create_decoder(SegBackend(), FakeVad(), AsrConfig(language_whitelist=["pl"]), lambda e: None)
    assert isinstance(la, StreamingDecoder)


def test_status_event_on_construction():
    _, _, _, _, events = make()
    assert events[0].type is CaptionEventType.STATUS and events[0].text == "CPU (fake)"


def test_a_segment_is_decoded_once_when_speech_ends():
    seg, backend, _, clock, events = make(responses=["dzień dobry"])
    run_frames(seg, clock, 40, level=1.0)  # 1.2 s of speech
    assert backend.calls == []  # nothing decoded while the segment is open (partials off)
    run_frames(seg, clock, 12, level=0.0)  # >= end_silence_ms
    assert final_texts(events) == ["dzień dobry"]
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert not call["peek"] and not call["drop_trailing_period"]  # a pause is a sentence end
    assert call["language"] == "pl"
    # speech + the silence that closed it (10 x 32 ms windows), plus no pre-roll (none buffered)
    assert call["seconds"] == pytest.approx((40 + 10) * 0.03, abs=0.001)
    assert of_type(events, CaptionEventType.PARTIAL) == []


def test_silence_only_never_decodes():
    seg, backend, _, clock, events = make(responses=["x"])
    run_frames(seg, clock, 200, level=0.0)
    assert backend.calls == [] and final_texts(events) == []


def test_speech_start_pulls_in_100ms_of_preroll():
    seg, _, _, clock, _ = make()
    run_frames(seg, clock, 20, level=0.0)
    seg._vad.level = 1.0
    clock.advance(0.03)
    seg.feed(FRAME, run_decode_step=False)
    # >= 100 ms of pre-roll in whole frames (4 x 30 ms) + the onset frame
    assert seg._window.duration_seconds == pytest.approx(0.15)


def test_final_carries_spoken_audio_times():
    seg, _, _, clock, events = make(responses=["słowa"])
    run_frames(seg, clock, 34, level=0.0)  # ~1 s of silence
    onset = clock.now + 0.03
    run_frames(seg, clock, 67, level=1.0)  # ~2 s of speech
    speech_end = clock.now
    run_frames(seg, clock, 12, level=0.0)
    final = of_type(events, CaptionEventType.FINAL)[0]
    assert final.start == pytest.approx(onset - 0.03 - 0.12, abs=0.02)  # frame start minus pre-roll
    assert final.end == pytest.approx(speech_end, abs=0.05)
    assert final.start < final.end < final.timestamp + 1e9


def test_drain_without_tick_never_peeks_and_tick_runs_one():
    config = seg_config(show_partial_captions=True, partial_latency_ms=300)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "coś")
    seg._vad.level = 1.0
    for _ in range(60):  # 1.8 s drained in one go
        clock.advance(0.03)
        seg.feed(FRAME, run_decode_step=False)
    assert backend.peeks == []
    seg.tick()
    assert len(backend.peeks) == 1
    assert backend.peeks[0]["seconds"] == pytest.approx(1.8)  # the whole open segment


def test_device_policy_error_propagates():
    seg, backend, _, clock, _ = make(responses=["x"])
    backend.fail_with = DevicePolicyError("gpu gone")
    run_frames(seg, clock, 30, level=1.0)
    with pytest.raises(DevicePolicyError):
        run_frames(seg, clock, 12, level=0.0)


def test_other_decode_errors_skip_the_segment():
    seg, backend, _, clock, events = make(responses=["x"])
    backend.fail_with = RuntimeError("boom")
    run_frames(seg, clock, 30, level=1.0)
    run_frames(seg, clock, 12, level=0.0)
    assert final_texts(events) == []
    backend.fail_with = None
    run_frames(seg, clock, 30, level=1.0)
    run_frames(seg, clock, 12, level=0.0)
    assert final_texts(events) == ["x"]


# ---------------------------------------------------------------------------
# segment cap: no-loss rollover, pause preference, Silero reset
# ---------------------------------------------------------------------------


def test_long_speech_loses_no_words_across_segment_caps():
    """The guarantee of the local-agreement no-loss rollover test, against
    the segmenter: 31 s of unbroken speech cut at a 10 s cap must caption
    every word exactly once."""
    config = seg_config(max_segment_ms=10000, end_silence_ms=600)
    seg, backend, _, clock, events = make(config, text_fn=word_text)
    seg._vad.level = 1.0
    run_audio(seg, clock, encoded_speech(31.0))  # 62 words
    seg._vad.level = 0.0
    run_audio(seg, clock, np.zeros(16000, dtype=np.float32))
    assert final_words(events) == [f"w{i}" for i in range(62)]
    assert seg.rollover_count == 3
    assert max(c["seconds"] for c in backend.finals) <= 10.05
    # a cut mid-speech is not a sentence end: whisper's "." there is dropped
    assert [c["drop_trailing_period"] for c in backend.finals] == [True, True, True, False]


def test_slow_decodes_still_lose_no_words():
    """Drain-then-tick: decodes that take longer than the audio they cover
    (here every peek costs 1 s of clock) never drop ingested audio."""
    config = seg_config(max_segment_ms=6000, end_silence_ms=600, show_partial_captions=True, partial_latency_ms=500)
    seg, backend, _, clock, events = make(config, text_fn=word_text)
    backend.peek_cost = 1.0
    backend.final_cost = 0.5
    seg._vad.level = 1.0
    run_audio(seg, clock, encoded_speech(20.0))
    seg._vad.level = 0.0
    run_audio(seg, clock, np.zeros(16000, dtype=np.float32))
    assert final_words(events) == [f"w{i}" for i in range(40)]
    assert backend.peeks  # partials did run


def test_cap_prefers_a_pause_once_the_segment_is_mostly_full():
    config = seg_config(max_segment_ms=10000, end_silence_ms=2000)
    seg, _, _, clock, _ = make(config, text_fn=word_text)
    cut_at = []

    def levels(k):
        t = k * 0.03
        if seg.rollover_count and not cut_at:
            cut_at.append(t)
        return 0.0 if (4.0 <= t < 4.3 or 7.0 <= t < 7.3) else 1.0

    run_audio(seg, clock, encoded_speech(9.0), levels=levels)
    assert seg.rollover_count == 1
    assert 7.1 <= cut_at[0] <= 7.3  # the second pause (segment > 6 s), not the first


def test_no_cut_below_the_cap():
    config = seg_config(max_segment_ms=10000, end_silence_ms=600)
    seg, backend, _, clock, _ = make(config, text_fn=word_text)
    seg._vad.level = 1.0
    run_audio(seg, clock, encoded_speech(5.5))
    assert seg.rollover_count == 0 and backend.calls == []


def test_silero_state_reset_on_each_transition_into_non_speech():
    seg, _, vad, clock, _ = make(responses=["a", "b"])
    run_frames(seg, clock, 30, level=1.0)
    run_frames(seg, clock, 50, level=0.0)
    assert vad.resets == 1  # once per transition, not per silent window
    run_frames(seg, clock, 30, level=1.0)
    run_frames(seg, clock, 50, level=0.0)
    assert vad.resets == 2


def test_real_silero_vad_keeps_its_quiet_reset_backstop():
    from narrator.asr.vad import SileroVad

    vad = SileroVad(threshold=0.5)
    assert vad._quiet_windows_for_reset > 0  # the 10 s quiet reset stays on
    vad._quiet_windows = 5
    vad.reset_state()
    assert vad._quiet_windows == 0 and not vad._state.any()


# ---------------------------------------------------------------------------
# partials (peeks)
# ---------------------------------------------------------------------------


def test_partials_replace_each_other_and_the_final_replaces_them():
    config = seg_config(show_partial_captions=True, partial_latency_ms=300)
    seg, backend, _, clock, events = make(config, text_fn=word_text)
    seg._vad.level = 1.0
    run_audio(seg, clock, encoded_speech(4.0))
    partials = [e.text for e in of_type(events, CaptionEventType.PARTIAL)]
    assert len(partials) >= 3
    # each partial is the WHOLE open segment so far (replace, not append)
    for text in partials:
        assert text.split()[0] == "w0"
    assert all(len(a.split()) <= len(b.split()) for a, b in zip(partials, partials[1:]))
    assert all(c["drop_trailing_period"] for c in backend.peeks)
    seg._vad.level = 0.0
    run_audio(seg, clock, np.zeros(8000, dtype=np.float32))
    assert final_texts(events) == ["w0 w1 w2 w3 w4 w5 w6 w7"]
    assert events[-1].type in (CaptionEventType.FINAL, CaptionEventType.CLEAR)
    after_final = events[[e.type for e in events].index(CaptionEventType.FINAL):]
    assert of_type(after_final, CaptionEventType.PARTIAL) == []


def test_partials_off_means_no_peek_decodes_at_all():
    seg, backend, _, clock, events = make(seg_config(show_partial_captions=False), text_fn=word_text)
    seg._vad.level = 1.0
    run_audio(seg, clock, encoded_speech(6.0))
    assert backend.peeks == [] and seg.peek_count == 0
    assert of_type(events, CaptionEventType.PARTIAL) == []


def test_peek_interval_grows_with_segment_length():
    config = seg_config(show_partial_captions=True, partial_latency_ms=1000, max_segment_ms=30000)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "x")
    run_frames(seg, clock, 700, level=1.0)  # 21 s
    held = [c["seconds"] for c in backend.peeks]
    gaps = [b - a for a, b in zip(held, held[1:])]
    assert gaps[0] >= 1.0
    assert gaps[-1] > gaps[0] + 1.0  # 1 s + 0.1 s per held second


def test_slow_peeks_give_way():
    """A peek waits until twice the previous peek's cost has passed since it
    started: partials take at most about half the worker."""
    config = seg_config(show_partial_captions=True, partial_latency_ms=300, max_segment_ms=30000)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "x")
    backend.peek_cost = 1.0  # each peek holds the (live) clock for 1 s
    seg._vad.level = 1.0
    starts = []
    for _ in range(300):
        clock.advance(0.03)
        seg.feed(FRAME, run_decode_step=False)
        before, t = len(backend.peeks), clock.now
        seg.tick()
        if len(backend.peeks) > before:
            starts.append(t)
    assert seg.peeks_skipped_busy > 0
    assert len(starts) >= 3
    assert all(b - a >= 2.0 for a, b in zip(starts, starts[1:]))


def test_peek_cost_is_not_read_stale_across_segments():
    """A peek just before its segment closes must not be costed at the
    first peek of a much later segment (that would read as a huge cost and
    starve the new segment's partials)."""
    config = seg_config(show_partial_captions=True, partial_latency_ms=300)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "x")
    run_frames(seg, clock, 40, level=1.0)
    run_frames(seg, clock, 300, level=0.0)  # 9 s of silence
    before = len(backend.peeks)
    run_frames(seg, clock, 60, level=1.0)  # 1.8 s of new speech
    assert len(backend.peeks) - before >= 2


def test_partial_cleared_when_the_final_comes_out_empty():
    config = seg_config(show_partial_captions=True, partial_latency_ms=300)
    texts = iter(["coś", "coś", "coś"])
    seg, backend, _, clock, events = make(config, text_fn=lambda a: next(texts, ""))
    run_frames(seg, clock, 40, level=1.0)
    assert of_type(events, CaptionEventType.PARTIAL)
    run_frames(seg, clock, 12, level=0.0)
    assert final_texts(events) == []
    assert of_type(events, CaptionEventType.PARTIAL)[-1].text == ""  # no stale partial left on air


# ---------------------------------------------------------------------------
# text pipeline and boundary dedupe
# ---------------------------------------------------------------------------


def test_text_pipeline_runs_on_partials_and_finals():
    config = seg_config(
        show_partial_captions=True,
        partial_latency_ms=300,
        text_filters=[{"pattern": r"\s*napisy wykonane.*", "replacement": ""}],
    )
    settings = TextSettings(blocklist=Blocklist.from_entries(["kurwa"]))
    seg, _, _, clock, events = make(
        config, text_fn=lambda a: "to jest jest kurwa test napisy wykonane przez", text_settings=settings
    )
    run_frames(seg, clock, 40, level=1.0)
    run_frames(seg, clock, 12, level=0.0)
    texts = [e.text for e in events if e.type in (CaptionEventType.PARTIAL, CaptionEventType.FINAL) and e.text]
    assert texts
    for text in texts:
        assert "kurwa" not in text and "napisy" not in text and "jest jest" not in text
        assert text.startswith("to jest k") and text.endswith("test")


def test_text_settings_hot_reload():
    seg, _, _, clock, events = make(responses=["kurwa mać", "kurwa mać"])
    run_frames(seg, clock, 30, level=1.0)
    run_frames(seg, clock, 12, level=0.0)
    seg.set_text_settings(TextSettings(blocklist=Blocklist.from_entries(["kurwa"])))
    run_frames(seg, clock, 30, level=1.0)
    run_frames(seg, clock, 12, level=0.0)
    first, second = final_texts(events)
    assert first == "kurwa mać" and "kurwa" not in second


def test_repeated_boundary_word_dropped_across_finals():
    seg, _, _, clock, events = make(responses=["mamy 15", "15 tysięcy"])
    for _ in range(2):
        run_frames(seg, clock, 30, level=1.0)
        run_frames(seg, clock, 12, level=0.0)
    assert final_texts(events) == ["mamy 15", "tysięcy"]


def test_peek_shows_the_boundary_dedupe_without_advancing_it():
    config = seg_config(show_partial_captions=True, partial_latency_ms=300)
    phase = {"text": "a b c"}
    seg, _, _, clock, events = make(config, text_fn=lambda a: phase["text"])
    speak_segment(seg, clock)
    assert final_texts(events)[-1] == "a b c"
    phase["text"] = "c d"
    run_frames(seg, clock, 40, level=1.0)
    partials = [e.text for e in of_type(events, CaptionEventType.PARTIAL) if e.text]
    assert partials[-1] == "d"  # shown without the doubled word ...
    run_frames(seg, clock, 12, level=0.0)
    assert final_texts(events)[-1] == "d"  # ... and the FINAL still drops it (state not advanced by peeks)


@pytest.mark.parametrize("text", ["", ".", " ... ", "!"])
def test_empty_or_punctuation_only_text_makes_no_event(text):
    seg, _, _, clock, events = make(responses=[text])
    run_frames(seg, clock, 30, level=1.0)
    run_frames(seg, clock, 12, level=0.0)
    assert final_texts(events) == []


# ---------------------------------------------------------------------------
# context sentences (whisper prompt)
# ---------------------------------------------------------------------------


def speak_segment(seg, clock, frames=30):
    run_frames(seg, clock, frames, level=1.0)
    run_frames(seg, clock, 12, level=0.0)


def test_context_is_the_last_n_finals():
    seg, backend, _, clock, _ = make(seg_config(context_sentences=2), responses=["jeden.", "dwa.", "trzy.", "cztery."])
    for _ in range(4):
        speak_segment(seg, clock)
    assert [c["prompt"] for c in backend.finals] == [None, "jeden.", "jeden. dwa.", "dwa. trzy."]


def test_no_context_when_off():
    seg, backend, _, clock, _ = make(seg_config(context_sentences=0), responses=["jeden.", "dwa."])
    speak_segment(seg, clock)
    speak_segment(seg, clock)
    assert [c["prompt"] for c in backend.finals] == [None, None]


def test_context_reset_by_a_wipe():
    config = seg_config(context_sentences=1, wipe_screen_after_silence_seconds=2.0)
    seg, backend, _, clock, events = make(config, responses=["jeden.", "dwa."])
    speak_segment(seg, clock)
    run_frames(seg, clock, 100, level=0.0)  # 3 s: WIPE
    assert of_type(events, CaptionEventType.WIPE)
    speak_segment(seg, clock)
    assert [c["prompt"] for c in backend.finals] == [None, None]


def test_context_not_carried_into_another_language():
    config = seg_config(context_sentences=1, language_whitelist=["pl", "en"])
    seg, backend, _, clock, _ = make(config, responses=["jeden.", "two.", "three."])
    backend.detected = [("pl", 0.9, 0.8), ("en", 0.9, 0.8), ("en", 0.9, 0.8)]
    for _ in range(3):
        speak_segment(seg, clock, frames=60)
    assert [c["language"] for c in backend.finals] == ["pl", "en", "en"]
    assert [c["prompt"] for c in backend.finals] == [None, None, "two."]


# ---------------------------------------------------------------------------
# silence timers (prune / wipe), unchanged semantics
# ---------------------------------------------------------------------------


def test_prune_once_per_lull_and_no_wipe_by_default():
    config = seg_config(prune_history_after_silence_seconds=1.0)
    seg, _, _, clock, events = make(config, responses=["a", "b"])
    speak_segment(seg, clock)
    run_frames(seg, clock, 300, level=0.0)  # 9 s
    assert len(of_type(events, CaptionEventType.CLEAR)) == 1
    assert of_type(events, CaptionEventType.WIPE) == []
    speak_segment(seg, clock)
    run_frames(seg, clock, 100, level=0.0)
    assert len(of_type(events, CaptionEventType.CLEAR)) == 2


def test_wipe_after_the_configured_lull():
    config = seg_config(prune_history_after_silence_seconds=1.0, wipe_screen_after_silence_seconds=4.0)
    seg, _, _, clock, events = make(config, responses=["a"])
    speak_segment(seg, clock)
    run_frames(seg, clock, 100, level=0.0)  # 3 s
    assert of_type(events, CaptionEventType.WIPE) == []
    run_frames(seg, clock, 100, level=0.0)
    assert len(of_type(events, CaptionEventType.WIPE)) == 1


def test_no_timers_before_any_speech():
    config = seg_config(prune_history_after_silence_seconds=1.0)
    seg, _, _, clock, events = make(config)
    run_frames(seg, clock, 300, level=0.0)
    assert of_type(events, CaptionEventType.CLEAR) == []


# ---------------------------------------------------------------------------
# hybrid and fixed modes
# ---------------------------------------------------------------------------


def test_hybrid_closes_every_segment_duration_and_skips_speechless_audio():
    config = seg_config(segmentation_mode="hybrid", segment_duration_ms=1000, max_segment_ms=15000)
    seg, backend, _, clock, events = make(config, responses=["a", "b", "c"])
    run_frames(seg, clock, 100, level=0.0)  # 3 s of silence: kept to its last 250 ms, never decoded
    assert backend.calls == []
    assert seg._window.duration_seconds <= 0.25 + 1e-9
    run_frames(seg, clock, 70, level=1.0)  # 2.1 s of speech
    assert len(backend.finals) == 2
    assert all(c["seconds"] == pytest.approx(1.0, abs=0.04) for c in backend.finals)
    assert final_texts(events) == ["a", "b"]


def test_hybrid_peeks_only_after_speech_was_heard():
    config = seg_config(segmentation_mode="hybrid", segment_duration_ms=5000, show_partial_captions=True,
                        partial_latency_ms=300)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "x")
    run_frames(seg, clock, 60, level=0.0)
    assert backend.peeks == []
    run_frames(seg, clock, 40, level=1.0)
    assert backend.peeks


def test_fixed_mode_uses_no_vad():
    config = seg_config(segmentation_mode="fixed", segment_duration_ms=1000, max_segment_ms=15000)
    seg, backend, vad, clock, events = make(config, text_fn=lambda a: "x")
    run_frames(seg, clock, 100, level=0.0)  # silence or not, fixed segments are decoded
    assert vad.calls == 0
    assert len(backend.finals) == 2  # 2 x 1.02 s
    seg.flush()  # the open 0.96 s remainder
    assert len(backend.finals) == 3
    assert of_type(events, CaptionEventType.CLEAR) == []  # no VAD, no silence timers


# ---------------------------------------------------------------------------
# language whitelist on segments (ported from the local-agreement tests)
# ---------------------------------------------------------------------------


def test_single_entry_whitelist_is_forced_without_detection():
    config = seg_config(show_partial_captions=True, partial_latency_ms=300)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "x")
    speak_segment(seg, clock, frames=60)
    assert backend.detect_calls == 0
    assert {c["language"] for c in backend.calls} == {"pl"}


def test_detection_waits_for_1_5_s_and_locks_once_per_segment():
    config = seg_config(language_whitelist=["pl", "en"], show_partial_captions=True, partial_latency_ms=300)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "x")
    run_frames(seg, clock, 45, level=1.0)  # 1.35 s: peeks due, but no lock yet
    assert backend.detect_calls == 0 and backend.peeks == []
    run_frames(seg, clock, 60, level=1.0)
    assert backend.detect_calls == 1
    assert backend.peeks and backend.peeks[0]["seconds"] >= 1.5
    run_frames(seg, clock, 12, level=0.0)
    assert backend.detect_calls == 1  # the FINAL reuses the lock
    assert {c["language"] for c in backend.calls} == {"pl"}


def test_ambiguous_margin_defers_the_lock_and_skips_peeks():
    config = seg_config(language_whitelist=["pl", "en"], show_partial_captions=True, partial_latency_ms=300)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "x")
    backend.detected = [("pl", 0.4, 0.05), ("pl", 0.45, 0.1), ("en", 0.8, 0.6)]
    run_frames(seg, clock, 52, level=1.0)  # 1.56 s: the first detection, ambiguous
    assert backend.detect_calls == 1 and backend.peeks == []
    run_frames(seg, clock, 100, level=1.0)  # re-detected on the growing segment until decisive
    assert backend.detect_calls == 3
    assert backend.peeks and {c["language"] for c in backend.peeks} == {"en"}


def test_ambiguous_finalize_keeps_the_previous_segment_language():
    config = seg_config(language_whitelist=["pl", "en"])
    seg, backend, _, clock, _ = make(config, responses=["jeden", "dwa"])
    backend.detected = [("en", 0.9, 0.8), ("pl", 0.4, 0.05), ("pl", 0.4, 0.05)]
    speak_segment(seg, clock, frames=60)
    speak_segment(seg, clock, frames=60)
    assert [c["language"] for c in backend.finals] == ["en", "en"]


def test_ambiguous_first_segment_uses_the_top_candidate():
    config = seg_config(language_whitelist=["pl", "en"])
    seg, backend, _, clock, events = make(config, responses=["jeden"])
    backend.detected = [("pl", 0.4, 0.05)]
    speak_segment(seg, clock, frames=60)
    assert [c["language"] for c in backend.finals] == ["pl"]
    assert final_texts(events) == ["jeden"]


def test_detection_failure_with_no_history_drops_the_segment_without_crashing():
    config = seg_config(language_whitelist=["pl", "en"])
    seg, backend, _, clock, events = make(config, responses=["jeden", "dwa"])
    backend.detect_language = lambda audio: None
    speak_segment(seg, clock, frames=60)
    assert final_texts(events) == [] and backend.calls == []


def test_capped_segment_keeps_its_lock_but_a_pause_rearms_it():
    config = seg_config(language_whitelist=["pl", "en"], max_segment_ms=1000, end_silence_ms=300)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "słowo")
    run_frames(seg, clock, 84, level=1.0)  # 2.5 s unbroken: two cuts at the cap
    assert seg.rollover_count == 2
    assert backend.detect_calls == 1  # decided once, carried through both cuts
    run_frames(seg, clock, 12, level=0.0)  # a real pause
    run_frames(seg, clock, 30, level=1.0)
    run_frames(seg, clock, 12, level=0.0)
    assert backend.detect_calls == 2


def test_hybrid_segments_decide_their_language_afresh():
    config = seg_config(language_whitelist=["pl", "en"], segmentation_mode="hybrid", segment_duration_ms=2000)
    seg, backend, _, clock, _ = make(config, text_fn=lambda a: "x")
    run_frames(seg, clock, 210, level=1.0)  # 6.3 s: three 2 s segments
    assert len(backend.finals) == 3 and backend.detect_calls == 3


# ---------------------------------------------------------------------------
# stream_sim drives the segmenter
# ---------------------------------------------------------------------------


class _ScheduledVad:
    def __init__(self, speech_frames: int) -> None:
        self.speech_frames = speech_frames
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        return [1.0 if self.calls <= self.speech_frames else 0.0]


def test_stream_sim_replays_the_segmenter_deterministically():
    from narrator.asr.stream_sim import DecodeCostModel, simulate_stream

    config = seg_config(show_partial_captions=True, partial_latency_ms=500, end_silence_ms=300)
    cost = DecodeCostModel(base_s=0.2, per_audio_s=0.05, peek_base_s=0.1, peek_per_audio_s=0.02)
    audio = encoded_speech(6.0)
    runs = [
        simulate_stream(SegBackend(text_fn=word_text), config, audio, cost, vad=_ScheduledVad(200))
        for _ in range(2)
    ]
    first = runs[0]
    assert first.to_dict() == runs[1].to_dict()
    assert first.text.split() == [f"w{i}" for i in range(12)]
    assert first.decode_calls == 1 and first.peek_calls >= 3
    assert first.partials and all(index == 0 for _, index, _ in first.partials)
    final_at = first.finals[0][0]
    # the FINAL is stamped at its decode's completion: after speech end + end silence
    assert final_at >= 6.0 + 0.3
    window = 6.0 + 0.32
    peeks = first.peek_calls
    expected_busy_min = cost.decode_seconds(window) + peeks * cost.decode_seconds(1.0, peek=True)
    assert first.busy_seconds >= expected_busy_min - 1e-6
    visible = first.first_visible_times()[0]
    assert visible[0] < final_at  # partials put words on screen before the FINAL


def test_stream_sim_flushes_an_open_hybrid_segment():
    from narrator.asr.stream_sim import DecodeCostModel, simulate_stream

    config = seg_config(segmentation_mode="hybrid", segment_duration_ms=10000, end_silence_ms=300)
    result = simulate_stream(
        SegBackend(text_fn=word_text), config, encoded_speech(3.0), DecodeCostModel(0.1, 0.01),
        vad=_ScheduledVad(100),
    )
    assert result.text.split() == [f"w{i}" for i in range(6)]


# ---------------------------------------------------------------------------
# the ASR worker loop drives the segmenter (drain-then-tick, live text settings)
# ---------------------------------------------------------------------------


def test_worker_runs_the_configured_segmenter_with_live_text_settings(monkeypatch):
    import queue
    import threading
    import time

    from narrator.asr import worker

    class _Backend(SegBackend):
        def __init__(self, config):
            super().__init__(text_fn=lambda a: "kurwa mać")
            self.vocabulary = None

        def set_vocabulary(self, vocabulary):
            self.vocabulary = vocabulary

        def close(self):
            pass

    class _Vad:
        """Speech while frames are non-zero."""

        def __init__(self, threshold):
            pass

        def process(self, frame):
            return [1.0 if np.any(frame) else 0.0]

        def reset_state(self):
            pass

    monkeypatch.setattr(worker, "create_backend", _Backend)
    monkeypatch.setattr(worker, "SileroVad", _Vad)
    audio_q, event_q, control_q = queue.Queue(), queue.Queue(), queue.Queue()
    stop = threading.Event()
    config = seg_config(end_silence_ms=300)
    thread = threading.Thread(
        target=worker._run, args=(config, audio_q, event_q, stop, TextSettings(), control_q), daemon=True
    )
    thread.start()

    def utterance():
        for _ in range(30):
            audio_q.put(np.full(FRAME_SAMPLES, 0.1, dtype=np.float32))
        for _ in range(15):
            audio_q.put(np.zeros(FRAME_SAMPLES, dtype=np.float32))

    def wait_finals(n):
        finals = []
        deadline = time.monotonic() + 5
        while len(finals) < n and time.monotonic() < deadline:
            try:
                event = event_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if event.type is CaptionEventType.FINAL:
                finals.append(event.text)
        return finals

    utterance()
    first = wait_finals(1)
    control_q.put(TextSettings(vocabulary=("Kowalski",), blocklist=Blocklist.from_entries(["kurwa"])))
    time.sleep(0.7)  # applied at the top of the worker's next loop (its queue wait is 0.5 s)
    utterance()
    second = wait_finals(1)
    stop.set()
    thread.join(timeout=5)
    assert first == ["kurwa mać"]
    assert second and "kurwa" not in second[0]
    assert worker._current_backend.vocabulary == ["Kowalski"]
