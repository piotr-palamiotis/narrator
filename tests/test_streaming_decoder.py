import numpy as np

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
        clear_after_silence_seconds=5,
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
    config = AsrConfig(step_ms=100, agreement_passes=2, end_silence_ms=300, clear_after_silence_seconds=5, language_whitelist=["pl"])
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
    config = AsrConfig(step_ms=100, agreement_passes=99, end_silence_ms=100, clear_after_silence_seconds=5, language_whitelist=["pl"])
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
    config = AsrConfig(step_ms=100, agreement_passes=2, end_silence_ms=300, clear_after_silence_seconds=5, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(
        ["lokalne modele modele", "lokalne modele", "lokalne modele"], config=config
    )
    feed_decode_step(decoder, clock, config)
    feed_decode_step(decoder, clock, config)
    finals = [e.text for e in events if e.type == CaptionEventType.FINAL]
    assert finals == ["lokalne modele"]


def test_force_finalize_output_collapsed():
    """The end-silence force-finalize path applies the same collapse."""
    config = AsrConfig(step_ms=100, agreement_passes=99, end_silence_ms=100, clear_after_silence_seconds=5, language_whitelist=["pl"])
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
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, clear_after_silence_seconds=5, language_whitelist=["pl"])
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
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=50, clear_after_silence_seconds=5, language_whitelist=["pl"])
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
    config = AsrConfig(step_ms=100, agreement_passes=1, end_silence_ms=300, clear_after_silence_seconds=5, language_whitelist=["pl"])
    decoder, clock, events = make_decoder(
        ["liczyło 15", "liczyło 15 tysięcy", "liczyło 15 tysięcy mieszkańców"],
        config=config,
    )
    feed_decode_step(decoder, clock, config)  # FINAL "liczyło 15"
    feed_decode_step(decoder, clock, config)  # new word "tysięcy" ≠ "15" -> kept
    finals = [e.text for e in events if e.type == CaptionEventType.FINAL]
    assert finals == ["liczyło 15", "tysięcy"]


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
        clear_after_silence_seconds=5,
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
        language_whitelist=["pl", "en"], clear_after_silence_seconds=5,
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
