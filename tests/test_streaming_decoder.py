import numpy as np
import pytest

from narrator.asr.backend import _restrict_to_whitelist
from narrator.asr.events import CaptionEventType
from narrator.asr.streaming_decoder import StreamingDecoder, _collapse_repetitions
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
        self.language_calls: list[str | None] = []  # language kwarg per transcribe
        self.detect_calls: int = 0
        self.detected_language = "pl"  # returned by detect_language
        self.detected_probability = 0.99
        self.detect_fails = False  # simulate detection errors (returns None)
        self.detected_margin = 0.9  # decisive by default

    def detect_language(self, audio):
        self.detect_calls += 1
        if self.detect_fails:
            return None
        return (self.detected_language, self.detected_probability, self.detected_margin)

    def transcribe(self, audio, language=None):
        self.language_calls.append(language)
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
    config = config or AsrConfig(
        step_ms=100,
        agreement_passes=2,
        end_silence_ms=300,
        prune_history_after_silence_seconds=5,
        # Permanent-forced (single entry): the agreement/commit tests only
        # care about text, not language selection, and must not depend on
        # detection timing.
        language_whitelist=["pl"],
    )
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
    config = AsrConfig(step_ms=100, agreement_passes=2, end_silence_ms=300, prune_history_after_silence_seconds=5, language_whitelist=["pl"])
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
    config = AsrConfig(step_ms=100, agreement_passes=99, end_silence_ms=100, prune_history_after_silence_seconds=5, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(["partial hypothesis", "partial hypothesis final"], config=config)

    feed_decode_step(decoder, clock, config)  # decode step; agreement_passes=99 so nothing commits yet
    feed_silence(decoder, 5)  # 5 * 32ms = 160ms >= end_silence_ms=100ms -> force-finalize

    finals = [e for e in events if e.type == CaptionEventType.FINAL]
    assert len(finals) == 1
    assert finals[0].text == "partial hypothesis final"


def test_clear_emitted_after_prolonged_silence():
    """Past the prune threshold, CLEAR fires exactly once per silence stretch;
    with wiping disabled (the default), no WIPE ever fires, however long the
    silence lasts."""
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=50, prune_history_after_silence_seconds=1.0)
    decoder, clock, events = make_decoder(["hi"], config=config)

    feed_decode_step(decoder, clock, config)  # decode "hi"; agreement_passes=1 commits immediately
    feed_silence(decoder, 2)  # 2 * 32ms = 64ms >= end_silence_ms=50ms -> force-finalize (no-op text-wise)
    assert any(e.type == CaptionEventType.FINAL for e in events)

    # Now well past prune_history_after_silence_seconds=1.0 with no further speech.
    clock.advance(2.0)
    feed_silence(decoder, 1)
    assert any(e.type == CaptionEventType.CLEAR for e in events)
    assert not any(e.type == CaptionEventType.WIPE for e in events)  # wipe disabled by default

    clock.advance(60.0)
    feed_silence(decoder, 1)
    assert not any(e.type == CaptionEventType.WIPE for e in events)  # even a very long lull


def test_no_duplicate_clear_events():
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=50, prune_history_after_silence_seconds=0.5)
    decoder, clock, events = make_decoder(["hi"], config=config)

    feed_decode_step(decoder, clock, config)
    feed_silence(decoder, 2)
    clock.advance(1.0)
    feed_silence(decoder, 3)
    clock.advance(1.0)
    feed_silence(decoder, 3)
    clear_events = [e for e in events if e.type == CaptionEventType.CLEAR]
    assert len(clear_events) == 1


def test_wipe_emitted_once_after_configured_lull():
    """With wipe_screen_after_silence_seconds enabled, a long-enough silence
    produces exactly one WIPE -- and only after it, never before the prune
    threshold just because both thresholds were crossed in one step."""
    config = AsrConfig(
        step_ms=100, agreement_passes=1, end_silence_ms=50,
        prune_history_after_silence_seconds=0.5, wipe_screen_after_silence_seconds=2.0,
    )
    decoder, clock, events = make_decoder(["hi"], config=config)

    feed_decode_step(decoder, clock, config)
    feed_silence(decoder, 2)  # force-finalize; idle timer starts at last speech

    clock.advance(1.0)  # past prune (0.5s), well short of wipe (2.0s)
    feed_silence(decoder, 1)
    assert any(e.type == CaptionEventType.CLEAR for e in events)
    assert not any(e.type == CaptionEventType.WIPE for e in events)

    clock.advance(1.5)  # total 2.5s idle >= 2.0s
    feed_silence(decoder, 1)
    wipe_events = [e for e in events if e.type == CaptionEventType.WIPE]
    assert len(wipe_events) == 1

    clock.advance(5.0)
    feed_silence(decoder, 1)
    wipe_events = [e for e in events if e.type == CaptionEventType.WIPE]
    assert len(wipe_events) == 1  # no duplicates on continued silence


def test_wipe_disabled_by_default_even_after_long_silence():
    """wipe_screen_after_silence_seconds=0 (the default) must never wipe: the
    operator decides when the screen goes blank."""
    config = AsrConfig(
        step_ms=100, agreement_passes=1, end_silence_ms=50,
        prune_history_after_silence_seconds=0.5, wipe_screen_after_silence_seconds=0,
    )
    decoder, clock, events = make_decoder(["hi"], config=config)

    feed_decode_step(decoder, clock, config)
    feed_silence(decoder, 2)
    clock.advance(600.0)
    feed_silence(decoder, 1)

    assert any(e.type == CaptionEventType.CLEAR for e in events)  # pruning still runs
    assert not any(e.type == CaptionEventType.WIPE for e in events)


def test_new_speech_resets_wipe_timer():
    """Any speech between lulls re-arms both the prune and wipe timers."""
    config = AsrConfig(
        step_ms=100, agreement_passes=1, end_silence_ms=50,
        prune_history_after_silence_seconds=0.5, wipe_screen_after_silence_seconds=2.0,
    )
    decoder, clock, events = make_decoder(["hi", "ho"], config=config)

    feed_decode_step(decoder, clock, config)
    clock.advance(1.0)  # speech again before any threshold
    feed_decode_step(decoder, clock, config)
    clock.advance(1.0)
    feed_silence(decoder, 1)

    assert not any(e.type == CaptionEventType.WIPE for e in events)
    clock.advance(1.5)
    feed_silence(decoder, 1)
    assert not any(e.type == CaptionEventType.WIPE for e in events)  # only 1.5s idle since last speech
    clock.advance(1.0)  # 2.5s idle now
    feed_silence(decoder, 1)
    assert len([e for e in events if e.type == CaptionEventType.WIPE]) == 1


def test_feed_without_decode_step_ingests_but_does_not_transcribe():
    """The ASR worker drains queued audio with run_decode_step=False so that
    ingest keeps pace with the microphone while inference runs at its own
    cadence. Ingesting must not consume a backend response.
    """
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300)
    decoder, clock, events = make_decoder(["should not be consumed"], config=config)

    decoder._vad.level = 1.0
    for _ in range(10):
        clock.advance(config.step_ms / 1000.0 + 0.001)
        decoder.feed(FRAME, run_decode_step=False)

    assert [e for e in events if e.type == CaptionEventType.FINAL] == []
    # ...and the audio really was ingested, not discarded
    assert len(decoder._window) == 10 * len(FRAME)


def test_tick_runs_one_decode_step_after_draining():
    """Drain-then-tick must produce exactly one decode for a whole batch of
    frames, which is what stops the audio queue from backing up."""
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(["jeden", "dwa"], config=config)

    decoder._vad.level = 1.0
    for _ in range(20):
        clock.advance(0.05)
        decoder.feed(FRAME, run_decode_step=False)
    decoder.tick()

    finals = [e.text for e in events if e.type == CaptionEventType.FINAL]
    assert finals == ["jeden"]  # exactly one decode consumed, not twenty


def test_tick_is_a_noop_when_not_in_speech():
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300)
    decoder, clock, events = make_decoder([], config=config)

    decoder._vad.level = 0.0
    decoder.feed(FRAME, run_decode_step=False)
    clock.advance(10.0)
    decoder.tick()  # must not call the backend (its response list is empty)

    assert [e for e in events if e.type == CaptionEventType.FINAL] == []


# ---------------------------------------------------------------------------
# _collapse_repetitions (word-doubling suppression)
# ---------------------------------------------------------------------------


def test_collapse_single_word_repeat():
    assert _collapse_repetitions(["niż", "niż", "pozostałe"]) == ["niż", "pozostałe"]


# ---------------------------------------------------------------------------
# _restrict_to_whitelist (detection probability re-argmax)
# ---------------------------------------------------------------------------


def test_whitelist_restriction_picks_best_allowed_language():
    probs = {"de": 0.6, "pl": 0.3, "en": 0.1, "ru": 0.001}
    code, prob, margin = _restrict_to_whitelist(probs, ["pl", "en"])
    assert (code, round(prob, 6), round(margin, 6)) == ("pl", 0.3, 0.2)


def test_whitelist_restriction_keeps_top_when_it_is_allowed():
    probs = {"en": 0.9, "pl": 0.05}
    assert _restrict_to_whitelist(probs, ["pl", "en"])[:2] == ("en", 0.9)


def test_whitelist_restriction_returns_none_when_nothing_matches():
    assert _restrict_to_whitelist({"de": 0.9, "ru": 0.1}, ["pl"]) is None


def test_collapse_multi_word_phrase_repeat():
    assert _collapse_repetitions(["z", "do", "nie", "z", "do", "nie"]) == ["z", "do", "nie"]


def test_collapse_longer_than_needed_phrase_runs_to_fixpoint():
    # "znacznie znacznie dłuższe" collapses, then the leftover "znacznie
    # znacznie" pair inside longer chains also collapses.
    assert _collapse_repetitions(["a", "a", "b", "b", "c"]) == ["a", "b", "c"]
    assert _collapse_repetitions(["x", "x", "x"]) == ["x"]


def test_no_collapse_for_non_adjacent_words():
    # "a b a b" IS an immediate phrase repeat and collapses; a repeat with a
    # word in between (or a non-immediate phrase echo) must survive.
    assert _collapse_repetitions(["a", "b", "a", "b", "c"]) == ["a", "b", "c"]
    assert _collapse_repetitions(["a", "b", "c", "a", "b"]) == ["a", "b", "c", "a", "b"]
    assert _collapse_repetitions(["a", "x", "a"]) == ["a", "x", "a"]


def test_collapse_runs_to_fixpoint():
    # "a b a b" -> "a b" (phrase), which itself is not a repeat -> stable.
    assert _collapse_repetitions(["a", "b", "a", "b"]) == ["a", "b"]
    # Nested: "z do nie z do nie z do nie" -> after one pass "z do nie z do nie" -> "z do nie"
    assert _collapse_repetitions(["z", "do", "nie"] * 3) == ["z", "do", "nie"]


def test_collapse_whole_sentence_loop():
    """large-v3's greedy streaming decode can emit the same 6-9 word sentence
    5-6 times inside one hypothesis (measured on the stress harness, mixed/
    numbers clips); whisper.cpp has no no_repeat_ngram_size and beam search is
    unusable in pywhispercpp 1.5.1, so sentence-length loops must collapse at
    the text layer."""
    sentence = "System działa na procesorze awaryjnie".split()
    assert _collapse_repetitions(sentence * 6) == sentence
    # Two different sentences interleaved as a loop also collapses.
    s1 = "Gdy okno dekodera rośnie".split()
    s2 = "System działa na procesorze".split()
    assert _collapse_repetitions(s1 + s2 + s1 + s2) == s1 + s2
    # A sentence repeated with a differing tail keeps the tail.
    assert _collapse_repetitions(s1 + ["koniec"]) == s1 + ["koniec"]


def test_decoder_hypothesis_collapsed_before_agreement():
    """A doubled word must not block the agreement prefix or reach a FINAL."""
    config = AsrConfig(step_ms=100, agreement_passes=2, end_silence_ms=300, prune_history_after_silence_seconds=5, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(
        ["lokalne modele modele", "lokalne modele", "lokalne modele"], config=config
    )
    feed_decode_step(decoder, clock, config)
    feed_decode_step(decoder, clock, config)
    finals = [e.text for e in events if e.type == CaptionEventType.FINAL]
    assert finals == ["lokalne modele"]


def test_force_finalize_output_collapsed():
    """The end-silence force-finalize path applies the same collapse."""
    config = AsrConfig(step_ms=100, agreement_passes=99, end_silence_ms=100, prune_history_after_silence_seconds=5, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(
        ["znacznie znacznie dłuższe", "znacznie znacznie dłuższe"], config=config
    )
    feed_decode_step(decoder, clock, config)
    feed_silence(decoder, 5)  # 5*32ms >= end_silence_ms=100 -> force-finalize
    finals = [e.text for e in events if e.type == CaptionEventType.FINAL]
    assert finals == ["znacznie dłuższe"]


def test_cross_commit_repeat_dropped():
    """Window-tail re-decode: word "15" committed, then the next window's
    hypothesis starts with "15" again -> the second must be dropped."""
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, prune_history_after_silence_seconds=5, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(
        ["liczyło 15", "liczyło 15 15 tysięcy", "liczyło 15 15 tysięcy mieszkańców"],
        config=config,
    )
    feed_decode_step(decoder, clock, config)  # commit "liczyło 15"
    feed_decode_step(decoder, clock, config)  # new commit starts with "15" -> dropped
    finals = [e.text for e in events if e.type == CaptionEventType.FINAL]
    assert finals == ["liczyło 15", "tysięcy"]


def test_cross_commit_guard_persists_across_utterances():
    """A stutter at an utterance boundary (next clip starting with the same
    word the previous FINAL ended with) is suppressed too."""
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=50, prune_history_after_silence_seconds=5, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(
        ["dobrze", "dobrze", "dobrze"],  # one word per decode
        config=config,
    )
    feed_decode_step(decoder, clock, config)  # FINAL "dobrze"
    feed_silence(decoder, 2)  # force-finalize (no remaining words)
    feed_decode_step(decoder, clock, config)  # new utterance, same word -> dropped
    finals = [e.text for e in events if e.type == CaptionEventType.FINAL]
    assert finals == ["dobrze"]


def test_cross_commit_guard_keeps_genuinely_new_word():
    """Only a first new word IDENTICAL to the previous FINAL's last word is
    dropped; a genuinely new word after the same prefix survives."""
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, prune_history_after_silence_seconds=5, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(
        ["liczyło 15", "liczyło 15 tysięcy", "liczyło 15 tysięcy mieszkańców"],
        config=config,
    )
    feed_decode_step(decoder, clock, config)  # FINAL "liczyło 15"
    feed_decode_step(decoder, clock, config)  # new word "tysięcy" ≠ "15" -> kept
    finals = [e.text for e in events if e.type == CaptionEventType.FINAL]
    assert finals == ["liczyło 15", "tysięcy"]


# ---------------------------------------------------------------------------
# long-utterance rollover (fresh whisper context mid-speech)
# ---------------------------------------------------------------------------


def test_utterance_rolls_over_at_max_utterance_ms():
    """Continuous speech longer than max_utterance_ms is force-finalized and
    the next speech frame starts a fresh utterance -- no silence required."""
    config = AsrConfig(
        step_ms=100, agreement_passes=1, end_silence_ms=5000,  # no silence path
        max_utterance_ms=1000,  # 1s of speech rolls the utterance over
        max_context_seconds=20,
        prune_history_after_silence_seconds=5,
        language_whitelist=["pl"],
    )
    decoder, clock, events = make_decoder(
        ["jeden", "dwa", "trzy", "cztery"], config=config
    )

    decoder._vad.level = 1.0
    finals_before = 0
    for _ in range(60):  # 60 * 30ms = 1.8s of continuous speech
        clock.advance(0.03)
        decoder.feed(FRAME, run_decode_step=False)
        finals_before = len([e for e in events if e.type == CaptionEventType.FINAL])

    # At least one rollover finalization happened mid-speech and the window
    # restarted with it (the utterance did not run on unbounded).
    assert finals_before >= 1
    assert decoder.rollover_count >= 1
    assert decoder._window.duration_seconds < 1.0


def test_no_rollover_below_max_utterance_ms():
    """Short speech stays one utterance: no finalization while speech is
    continuous and under the threshold."""
    config = AsrConfig(
        step_ms=100, agreement_passes=1, end_silence_ms=5000,
        max_utterance_ms=10000,
        max_context_seconds=20,
        prune_history_after_silence_seconds=5,
        language_whitelist=["pl"],
    )
    decoder, clock, events = make_decoder(["slow response"], config=config)

    decoder._vad.level = 1.0
    for _ in range(60):  # 1.8s of continuous speech, far under 10s
        clock.advance(0.03)
        decoder.feed(FRAME, run_decode_step=False)

    assert not any(e.type == CaptionEventType.FINAL for e in events)
    assert decoder.rollover_count == 0
    assert decoder._window.duration_seconds > 1.5


# ---------------------------------------------------------------------------
# per-utterance language selection (language_whitelist)
# ---------------------------------------------------------------------------


def test_single_entry_whitelist_is_permanent_forced():
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(["cześć"], config=config)
    backend = decoder._backend
    feed_decode_step(decoder, clock, config)
    assert backend.language_calls == ["pl"]
    assert backend.detect_calls == 0  # the language is known from config; detection never runs


def test_language_detected_once_and_locked_per_utterance():
    """Multi-entry whitelist: one detection per utterance, locked language
    passed to every decode step of that utterance. Detection defers until the
    window holds _LANGUAGE_LOCK_MIN_SECONDS of audio (short slices misdetect);
    decode steps before the lock are skipped entirely."""
    from narrator.asr.streaming_decoder import _LANGUAGE_LOCK_MIN_SECONDS

    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, language_whitelist=["pl", "en"])
    decoder, clock, events = make_decoder(["hello", "hello there", "hello there again"], config=config)
    backend = decoder._backend
    backend.detected_language = "en"

    # Before the window reaches the detection minimum: decode skipped, no
    # detection, no backend call.
    clock.advance(config.step_ms / 1000.0 + 0.001)
    decoder._vad.level = 1.0
    decoder.feed(FRAME)
    assert backend.detect_calls == 0
    assert backend.language_calls == []

    # Stuff the window past the detection minimum, then decode: exactly one
    # detection, and the locked language is used.
    n_frames = int(_LANGUAGE_LOCK_MIN_SECONDS * 1000 / 30) + 2
    for _ in range(n_frames):
        clock.advance(config.step_ms / 1000.0 + 0.001)
        decoder.feed(FRAME, run_decode_step=False)
    decoder.tick()
    assert backend.detect_calls == 1
    assert backend.language_calls == ["en"]

    # Further steps in the same utterance: no new detection, same language.
    feed_decode_step(decoder, clock, config)
    assert backend.detect_calls == 1
    assert backend.language_calls == ["en", "en"]


def test_language_redetected_for_new_utterance():
    """A new utterance re-detects: the speaker may have switched languages.
    Utterance 1 has enough audio to lock mid-speech; utterance 2 is a short
    burst that only reaches the detection minimum at force-finalize."""
    from narrator.asr.streaming_decoder import _LANGUAGE_LOCK_MIN_SECONDS

    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=50, language_whitelist=["pl", "en"])
    decoder, clock, events = make_decoder(["hello", "cześć", "ignored"], config=config)
    backend = decoder._backend
    backend.detected_language = "en"

    n_frames = int(_LANGUAGE_LOCK_MIN_SECONDS * 1000 / 30) + 2
    decoder._vad.level = 1.0
    for _ in range(n_frames):
        clock.advance(config.step_ms / 1000.0 + 0.001)
        decoder.feed(FRAME, run_decode_step=False)
    decoder.tick()  # utterance 1: detect -> en, decode once
    feed_silence(decoder, 2)  # force-finalize utterance 1 (decodes again, still "en")

    backend.detected_language = "pl"
    feed_decode_step(decoder, clock, config)  # utterance 2: too short to lock, decode skipped
    feed_silence(decoder, 2)  # force-finalize: force-lock -> pl, decode

    assert backend.detect_calls == 2
    assert backend.language_calls == ["en", "en", "pl"]


def test_ambiguous_margin_defers_lock_then_locks_when_decisive():
    """The margin gate: when the top two candidates are close, the lock is
    deferred and the growing window is re-detected on later steps -- the
    'build confidence up' behavior. Once the margin clears, the language
    locks and decoding proceeds."""
    from narrator.asr.streaming_decoder import _LANGUAGE_LOCK_MIN_SECONDS, _LANGUAGE_MIN_MARGIN

    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, language_whitelist=["pl", "en"])
    decoder, clock, events = make_decoder(["garbage"], config=config)
    backend = decoder._backend
    backend.detected_language = "pl"
    backend.detected_margin = _LANGUAGE_MIN_MARGIN / 2  # ambiguous

    n_frames = int(_LANGUAGE_LOCK_MIN_SECONDS * 1000 / 30) + 2
    decoder._vad.level = 1.0
    for _ in range(n_frames):
        clock.advance(config.step_ms / 1000.0 + 0.001)
        decoder.feed(FRAME, run_decode_step=False)
    decoder.tick()
    assert backend.detect_calls == 1
    assert decoder._locked_language is None  # not locked: margin too narrow
    assert backend.language_calls == []  # nothing decoded in an unverified language

    # More audio -> decisive margin -> next step locks and decodes.
    backend.detected_margin = 0.9
    feed_decode_step(decoder, clock, config)
    assert decoder._locked_language == "pl"
    assert backend.language_calls == ["pl"]


def test_detection_error_retries_without_crashing():
    """A detection failure must not crash the loop nor lock a language."""
    from narrator.asr.streaming_decoder import _LANGUAGE_LOCK_MIN_SECONDS

    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=50, language_whitelist=["pl", "en"])
    decoder, clock, events = make_decoder(["cześć", "ignored"], config=config)
    backend = decoder._backend
    backend.detect_fails = True

    n_frames = int(_LANGUAGE_LOCK_MIN_SECONDS * 1000 / 30) + 2
    decoder._vad.level = 1.0
    for _ in range(n_frames):
        clock.advance(config.step_ms / 1000.0 + 0.001)
        decoder.feed(FRAME, run_decode_step=False)
    decoder.tick()  # must not raise
    assert decoder._locked_language is None

    feed_silence(decoder, 2)  # force-finalize also retries, fails again -> no decode
    assert backend.language_calls == []


# ---------------------------------------------------------------------------
# per-utterance switching (multi-entry whitelist) and finalize hysteresis
# ---------------------------------------------------------------------------


def run_utterance(decoder, clock, config):
    """Decode one utterance: one decode step, then force-finalize via silence.
    The finalize path also decodes, hence two responses consumed per utterance."""
    feed_decode_step(decoder, clock, config)
    feed_silence(decoder, 2)


def test_switch_happens_immediately_on_decisive_detection():
    """With a multi-entry whitelist the best allowed language wins per
    utterance, immediately: the old threshold-based switch (N consecutive
    detections before flipping) is replaced by the margin gate -- a decisive
    detection IS the switch, and a language returns the moment it's detected
    again."""
    config = AsrConfig(
        step_ms=100,
        agreement_passes=1,
        end_silence_ms=50,
        language_whitelist=["pl", "en"],
        prune_history_after_silence_seconds=5,
    )
    decoder, clock, events = make_decoder(
        ["witajcie", "headsets", "witajcie z powrotem"], config=config
    )
    backend = decoder._backend
    backend.detected_language = "pl"
    run_utterance(decoder, clock, config)  # utterance 1: pl
    # In multi-entry mode the micro-utterances here (one 30ms frame) defer
    # the lock to finalize, so each utterance decodes exactly once.
    assert backend.language_calls == ["pl"]

    backend.detected_language = "en"
    run_utterance(decoder, clock, config)  # utterance 2: en immediately
    assert backend.language_calls[-1:] == ["en"]

    backend.detected_language = "pl"
    run_utterance(decoder, clock, config)  # utterance 3: back to pl immediately
    assert backend.language_calls[-1:] == ["pl"]


def test_ambiguous_finalize_keeps_previous_utterance_language():
    """Hysteresis at finalize: if the margin never cleared, the previous
    utterance's language is kept -- a wrong-but-consistent caption beats a
    coin flip that mixes languages mid-speech."""
    config = AsrConfig(
        step_ms=100, agreement_passes=1, end_silence_ms=50,
        language_whitelist=["pl", "en"], prune_history_after_silence_seconds=5,
    )
    decoder, clock, events = make_decoder(["a", "b", "c"], config=config)
    backend = decoder._backend
    backend.detected_language = "en"
    run_utterance(decoder, clock, config)  # decisive en
    assert decoder._last_utterance_language == "en"

    backend.detected_margin = 0.01  # ambiguous from here on
    run_utterance(decoder, clock, config)  # lock deferred; finalize keeps en
    assert backend.language_calls[-1:] == ["en"]
    run_utterance(decoder, clock, config)  # still ambiguous: en again
    assert backend.language_calls[-1:] == ["en"]


# ---------------------------------------------------------------------------
# long utterances roll over instead of trimming: no word is lost
# ---------------------------------------------------------------------------

CHUNK = 8000  # 0.5 s of audio per word in the word-encoded fake


class WordAudioBackend:
    """Transcribes audio whose samples encode word ids: every 0.5 s chunk of
    the source holds the constant (id + 1) / 1000. A word is heard when at
    least half its chunk is in the window -- so a word cut by a window
    boundary lands in exactly one of the two windows, like real speech."""

    resolved_device_description = "CPU (fake)"

    def __init__(self) -> None:
        self.window_samples: list[int] = []

    def detect_language(self, audio):
        return ("pl", 0.99, 0.9)

    def transcribe(self, audio, language=None, prompt=None):
        self.window_samples.append(len(audio))
        ids = np.rint(np.asarray(audio) * 1000).astype(int) - 1
        values, first, counts = np.unique(ids, return_index=True, return_counts=True)
        heard = [int(v) for f, v, c in sorted(zip(first, values, counts)) if v >= 0 and c >= CHUNK // 2]
        return [(0.0, 1.0, " ".join(f"w{i}" for i in heard))] if heard else []


def encoded_speech(seconds: float) -> np.ndarray:
    n = int(seconds * 16000)
    return ((np.arange(n) // CHUNK) + 1).astype(np.float32) / 1000.0


def run_audio(decoder, clock, audio, speech_levels=None):
    """Feed `audio` as 30 ms frames, drain-then-tick like the worker."""
    for k, i in enumerate(range(0, len(audio), 480)):
        if speech_levels is not None:
            decoder._vad.level = speech_levels(k)
        clock.advance(0.03)
        decoder.feed(audio[i : i + 480], run_decode_step=False)
        decoder.tick()


def finals_of(events) -> list[str]:
    return " ".join(e.text for e in events if e.type is CaptionEventType.FINAL).split()


def test_long_speech_loses_no_words_across_rollovers():
    """The regression the trimming design had: past max_context_seconds the
    window was trimmed under the committed-word count, and the finalize
    emitted only the tail -- seconds of speech silently vanished."""
    config = AsrConfig(
        step_ms=300, agreement_passes=2, end_silence_ms=600,
        max_context_seconds=10.0, max_utterance_ms=25000, language_whitelist=["pl"],
    )
    events = []
    clock = FakeClock()
    decoder = StreamingDecoder(WordAudioBackend(), FakeVad(), config, on_event=events.append, clock=clock)
    decoder._vad.level = 1.0
    run_audio(decoder, clock, encoded_speech(31.0))  # 62 words of unbroken speech
    decoder._vad.level = 0.0
    run_audio(decoder, clock, np.zeros(16000, dtype=np.float32))

    assert finals_of(events) == [f"w{i}" for i in range(62)]
    assert decoder.rollover_count == 3  # at 10, 20 and 30 s


def test_rollover_happens_at_the_smaller_cap():
    config = AsrConfig(
        step_ms=300, agreement_passes=1, end_silence_ms=600,
        max_context_seconds=20.0, max_utterance_ms=4000, language_whitelist=["pl"],
    )
    clock = FakeClock()
    backend = WordAudioBackend()
    decoder = StreamingDecoder(backend, FakeVad(), config, on_event=lambda e: None, clock=clock)
    decoder._vad.level = 1.0
    run_audio(decoder, clock, encoded_speech(9.0))
    assert decoder.rollover_count == 2
    assert max(backend.window_samples) <= 4.05 * 16000  # the decode window never exceeds the cap


def test_rollover_prefers_a_pause_once_the_window_is_mostly_full():
    """A pause too short to end the utterance still makes a clean cut once
    the window is 60% full; a pause before that changes nothing."""
    config = AsrConfig(
        step_ms=300, agreement_passes=1, end_silence_ms=2000,
        max_context_seconds=10.0, language_whitelist=["pl"],
    )
    events = []
    clock = FakeClock()
    decoder = StreamingDecoder(WordAudioBackend(), FakeVad(), config, on_event=events.append, clock=clock)
    rollover_at = []

    def levels(frame_index: int) -> float:
        t = frame_index * 0.03
        quiet = 4.0 <= t < 4.3 or 7.0 <= t < 7.3  # two 300 ms pauses
        if decoder.rollover_count and not rollover_at:
            rollover_at.append(t)
        return 0.0 if quiet else 1.0

    run_audio(decoder, clock, encoded_speech(9.0), speech_levels=levels)
    assert decoder.rollover_count == 1
    assert 7.1 <= rollover_at[0] <= 7.3  # at the second pause (window > 6 s), not the first


def test_language_lock_carries_across_a_rollover_but_not_a_pause():
    config = AsrConfig(
        step_ms=100, agreement_passes=1, end_silence_ms=300,
        max_context_seconds=1.0, language_whitelist=["pl", "en"],
    )
    decoder, clock, _ = make_decoder(["słowo"] * 200, config=config)
    backend = decoder._backend
    decoder._vad.level = 1.0
    for _ in range(84):  # 2.5 s of unbroken speech: two rollovers
        clock.advance(0.03)
        decoder.feed(FRAME, run_decode_step=False)
        decoder.tick()
    assert decoder.rollover_count == 2
    assert backend.detect_calls == 1  # decided once, carried through both rollovers
    feed_silence(decoder, 12)  # a real pause: the next speaker gets a fresh decision
    decoder._vad.level = 1.0
    for _ in range(20):
        clock.advance(0.03)
        decoder.feed(FRAME, run_decode_step=False)
    feed_silence(decoder, 12)
    assert backend.detect_calls == 2


# ---------------------------------------------------------------------------
# VAD pre-roll
# ---------------------------------------------------------------------------


def test_vad_preroll_prepends_audio_from_before_the_onset():
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, vad_preroll_ms=240, language_whitelist=["pl"])
    decoder, clock, _ = make_decoder(["x"], config=config)
    feed_silence(decoder, 20)
    decoder._vad.level = 1.0
    decoder.feed(FRAME, run_decode_step=False)
    # 240 ms of pre-roll (8 frames) + the onset frame
    assert decoder._window.duration_seconds == pytest.approx(0.27)


def test_no_preroll_by_default():
    decoder, clock, _ = make_decoder(["x"])
    feed_silence(decoder, 20)
    decoder._vad.level = 1.0
    decoder.feed(FRAME, run_decode_step=False)
    assert decoder._window.duration_seconds == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# prompt carry-over
# ---------------------------------------------------------------------------


class PromptBackend(FakeBackend):
    def __init__(self, responses):
        super().__init__(responses)
        self.prompts: list[str | None] = []

    def transcribe(self, audio, language=None, prompt=None):
        self.prompts.append(prompt)
        return super().transcribe(audio, language=language)


def prompt_decoder(config, responses):
    events = []
    clock = FakeClock()
    backend = PromptBackend(responses)
    decoder = StreamingDecoder(backend, FakeVad(), config, on_event=events.append, clock=clock, min_decode_seconds=0.01)
    return decoder, clock, backend, events


def speak(decoder, clock, config, steps=1):
    for _ in range(steps):
        feed_decode_step(decoder, clock, config)
    feed_silence(decoder, 12)  # end of utterance: the finalize decode


def test_prompt_carries_earlier_captions_and_resets_on_wipe():
    config = AsrConfig(
        step_ms=100, agreement_passes=1, end_silence_ms=300, prompt_carryover_chars=12,
        prune_history_after_silence_seconds=1.0, wipe_screen_after_silence_seconds=2.0,
        language_whitelist=["pl"],
    )
    decoder, clock, backend, _ = prompt_decoder(
        config, ["ala ma kota", "ala ma kota", "i psa też", "i psa też", "nowy", "nowy", "temat", "temat"]
    )
    speak(decoder, clock, config)  # utterance 1: no context yet
    assert backend.prompts == [None, None]
    speak(decoder, clock, config)  # utterance 2: utterance 1 as context
    assert backend.prompts[2:] == ["ala ma kota", "ala ma kota"]
    speak(decoder, clock, config)  # utterance 3: the last 12 chars, from a word start
    assert backend.prompts[4:] == ["i psa też", "i psa też"]
    clock.advance(3.0)  # a long lull: the screen wipes, the context goes with it
    feed_silence(decoder, 1)
    speak(decoder, clock, config)
    assert backend.prompts[6:] == [None, None]


def test_prompt_is_not_carried_into_another_language():
    config = AsrConfig(
        step_ms=100, agreement_passes=1, end_silence_ms=300, prompt_carryover_chars=100,
        language_whitelist=["pl", "en"],
    )
    decoder, clock, backend, _ = prompt_decoder(config, ["dzień dobry"] + ["good morning"] * 5)
    backend.detected_language = "pl"
    speak(decoder, clock, config)
    backend.detected_language = "en"
    speak(decoder, clock, config)
    assert all(p is None for p in backend.prompts)  # no Polish context for English speech
    speak(decoder, clock, config)
    assert backend.prompts[-1] == "good morning"


def test_no_prompt_when_carryover_is_off():
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, language_whitelist=["pl"])
    decoder, clock, backend, _ = prompt_decoder(config, ["raz"] * 4)
    speak(decoder, clock, config)
    speak(decoder, clock, config)
    assert backend.prompts == [None] * len(backend.prompts)



# ---------------------------------------------------------------------------
# repetition collapse ignores case and punctuation, reaches long sentences
# ---------------------------------------------------------------------------


def test_collapse_ignores_case_and_punctuation_keeping_the_first_occurrence():
    loop = "Zanim przebiegną tędy nowe jezdnie. zanim przebiegną tędy nowe jezdnie, zanim przebiegną tędy nowe jezdnie,".split()
    assert _collapse_repetitions(loop) == "Zanim przebiegną tędy nowe jezdnie.".split()


def test_collapse_catches_a_long_sentence_loop():
    sentence = (
        "Można umówić się z nią w kawiarni i nie wychodząc na lipcowy deszcz tuż obok kupić kwiaty."
    ).split()  # 17 words, past the old 16-word limit
    looped = ["kupić", "kwiaty."] + sentence * 3 + ["Można", "um"]
    assert _collapse_repetitions(looped) == ["kupić", "kwiaty."] + sentence + ["Można", "um"]


def test_cross_commit_guard_ignores_case_and_punctuation():
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, language_whitelist=["pl"])
    # utterance 1: decode step + finalize decode; utterance 2: one decode step
    decoder, clock, events = make_decoder(["nowe jezdnie.", "nowe jezdnie.", "Jezdnie, i chodniki"], config=config)
    feed_decode_step(decoder, clock, config)
    feed_silence(decoder, 12)  # utterance 1 ends on "jezdnie."
    feed_decode_step(decoder, clock, config)  # utterance 2 starts with "Jezdnie,"
    finals = [e.text for e in events if e.type is CaptionEventType.FINAL]
    assert finals[-1] == "i chodniki"


# ---------------------------------------------------------------------------
# Text filters and FINAL spoken-time stamps
# ---------------------------------------------------------------------------


def test_text_filters_rewrite_hypotheses_before_agreement():
    config = AsrConfig(
        step_ms=100, agreement_passes=2, end_silence_ms=300, prune_history_after_silence_seconds=5,
        language_whitelist=["pl"],
        text_filters=[
            {"pattern": r"\s*napisy stworzone przez.*", "replacement": ""},
            {"pattern": r"\bwarszawa\b", "replacement": "Warszawa"},
        ],
    )
    decoder, clock, events = make_decoder(
        [
            "jedziemy do warszawy napisy stworzone przez kogoś",
            "jedziemy do WARSZAWA napisy stworzone przez kogoś innego",
        ],
        config=config,
    )
    feed_decode_step(decoder, clock, config)
    feed_decode_step(decoder, clock, config)
    finals = [e.text for e in events if e.type == CaptionEventType.FINAL]
    # The credits tail is gone from both hypotheses, so they agree on
    # "jedziemy do" and differ only in the inflected word.
    assert finals == ["jedziemy do"]


def _feed_real_time(decoder, clock, level: float, frames: int) -> None:
    decoder._vad.level = level
    for _ in range(frames):
        clock.advance(len(FRAME) / 16000)  # real time: one frame per frame-length
        decoder.feed(FRAME)


def test_final_events_carry_spoken_time_not_emit_time():
    # agreement_passes=99: nothing commits early, so the one FINAL comes from
    # the finalize and spans the whole utterance.
    config = AsrConfig(
        step_ms=100, agreement_passes=99, end_silence_ms=300, prune_history_after_silence_seconds=5,
        language_whitelist=["pl"],
    )
    decoder, clock, events = make_decoder(["raz dwa trzy"] * 30, config=config)
    _feed_real_time(decoder, clock, 0.0, 10)  # 0.3s of silence
    speech_start = clock.now  # the first speech frame spans [now, now + 30ms]
    _feed_real_time(decoder, clock, 1.0, 30)  # 0.9s of speech
    speech_end = clock.now
    _feed_real_time(decoder, clock, 0.0, 15)  # long enough to finalize

    finals = [e for e in events if e.type == CaptionEventType.FINAL]
    assert [f.text for f in finals] == ["raz dwa trzy"]
    assert abs(finals[0].start - speech_start) < 0.05
    assert abs(finals[0].end - speech_end) < 0.05


def test_committed_finals_are_stamped_in_order_within_the_speech():
    config = AsrConfig(
        step_ms=100, agreement_passes=2, end_silence_ms=300, prune_history_after_silence_seconds=5,
        language_whitelist=["pl"],
    )
    words = ["raz", "dwa", "trzy", "cztery", "pięć", "sześć"]
    responses = [" ".join(words[: n + 1]) for n in range(6)]
    decoder, clock, events = make_decoder(responses + [responses[-1]] * 20, config=config)
    _feed_real_time(decoder, clock, 0.0, 10)
    speech_start = clock.now
    _feed_real_time(decoder, clock, 1.0, 30)
    speech_end = clock.now
    _feed_real_time(decoder, clock, 0.0, 15)

    finals = [e for e in events if e.type == CaptionEventType.FINAL]
    assert len(finals) > 1
    assert abs(finals[0].start - speech_start) < 0.05
    for earlier, later in zip(finals, finals[1:]):
        assert earlier.start <= earlier.end <= later.start + 1e-9 <= later.end + 1e-9
    assert finals[-1].end <= speech_end + 0.05


class _TimedBackend(FakeBackend):
    """FakeBackend that also reports word times, like WhisperBackend's
    last_word_times: each response is a list of (word, start, end)."""

    def __init__(self, timed_responses):
        super().__init__([" ".join(w for w, _, _ in r) for r in timed_responses])
        self._timed = iter(timed_responses)
        self.last_word_times = []

    def transcribe(self, audio, language=None):
        from narrator.asr.backend import WordTime

        self.last_word_times = [WordTime(w, s, e) for w, s, e in next(self._timed)]
        return super().transcribe(audio, language=language)


def test_final_times_come_from_backend_word_times_when_available():
    # One utterance, finalize-only (agreement_passes=99): the backend says
    # the words were spoken at 0.10-0.30 and 0.50-0.70 s into the window,
    # so the FINAL spans exactly that, not the whole speech stretch.
    config = AsrConfig(
        step_ms=100, agreement_passes=99, end_silence_ms=300, prune_history_after_silence_seconds=5,
        language_whitelist=["pl"],
    )
    events = []
    clock = FakeClock()
    backend = _TimedBackend([[("raz", 0.10, 0.30), ("dwa", 0.50, 0.70)]] * 30)
    decoder = StreamingDecoder(backend, FakeVad(), config, on_event=events.append, clock=clock, min_decode_seconds=0.01)
    _feed_real_time(decoder, clock, 0.0, 10)
    speech_start = clock.now
    _feed_real_time(decoder, clock, 1.0, 30)
    _feed_real_time(decoder, clock, 0.0, 15)
    finals = [e for e in events if e.type == CaptionEventType.FINAL]
    assert [f.text for f in finals] == ["raz dwa"]
    assert abs(finals[0].start - (speech_start + 0.10)) < 1e-6
    assert abs(finals[0].end - (speech_start + 0.70)) < 1e-6


def test_align_word_times_follows_text_processing():
    from narrator.asr.backend import WordTime
    from narrator.asr.decoder_common import align_word_times

    raw = [WordTime("raz", 0, 1), WordTime("dwa", 1, 2), WordTime("dwa", 2, 3), WordTime("Trzy.", 3, 4)]
    # repetition collapse dropped one "dwa"; case/punctuation differ
    assert align_word_times(["raz", "dwa", "trzy"], raw) == [(0, 1), (1, 2), (3, 4)]
    # a filter replaced two words with one: spread over the replaced run
    assert align_word_times(["raz", "X", "trzy"], raw[:1] + [WordTime("a", 1, 2), WordTime("b", 2, 3)] + raw[3:]) == [
        (0, 1), (1.0, 3.0), (3, 4)
    ]
    assert align_word_times(["raz"], []) is None
    assert align_word_times(["raz"], object()) is None
