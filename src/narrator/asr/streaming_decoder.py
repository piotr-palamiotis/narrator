"""Sliding-window + local-agreement incremental decode loop.

Mirrors the well-known "whisper_streaming" pattern: whisper.cpp is re-run on
a growing audio window every `step_ms`; the new hypothesis is compared
word-by-word against the previous one, and a prefix is only committed
(emitted as a FINAL CaptionEvent) once it has stayed stable for
`agreement_passes` consecutive decode steps. This trades a little latency
(bounded by step_ms * agreement_passes) for captions that don't visibly
backtrack -- appropriate for a broadcast lower-third. On VAD-detected silence
past `end_silence_ms`, whatever remains is force-finalized regardless of
agreement, bounding worst-case latency for short utterances.

Simplification, called out explicitly: the audio window is bounded by
unconditionally trimming to `max_context_seconds` rather than trimming
precisely at committed-word audio boundaries (which would require per-token
timestamps and materially more bookkeeping for a marginal-latency edge case
that only matters for unusually long continuous speech without a pause).
End-of-utterance silence detection is the primary bound in the common case.

Decode hypotheses are passed through `_collapse_repetitions` before agreement
comparison and commit: streaming re-decode is prone to immediate phrase
repetitions ("niż niż") that survive into captions otherwise. See that
function for the tradeoff this makes.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from narrator.asr.backend import WhisperBackend, _LANGUAGE_PROB_THRESHOLD
from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.vad import SileroVad
from narrator.audio.ring_buffer import AudioWindow
from narrator.config.schema import AsrConfig

logger = logging.getLogger(__name__)

_VAD_WINDOW_MS = 32.0  # fixed by the Silero graph's 512-sample-at-16kHz requirement
_MIN_DECODE_SECONDS = 0.3  # skip decode steps on near-empty windows
_OVERRUN_WARN_FACTOR = 2.0  # warn once a decode step takes >2x the configured step
_OVERRUN_WARN_INTERVAL_SECONDS = 10.0
# asr.language == "auto": whisper's language detection is unreliable on very
# short slices (measured on large-v3: a 0.4s slice misdetects 1-in-7 clips;
# at 1.5s every clip detects correctly with >=0.97 confidence), and even at
# sufficient length a marginal clip can flip (measured: one clip locked
# Cyrillic at 1.5s). The lock therefore requires BOTH: the window holds
# _LANGUAGE_LOCK_MIN_SECONDS of audio AND the backend reports at least
# _LANGUAGE_PROB_THRESHOLD confidence (backend.py). Until then, decode steps
# are skipped -- output in an unverified language would be garbage we'd have
# to suppress anyway, and the post-lock decode covers the whole window.
# Shorter/ambiguous utterances lock at force-finalize.
_LANGUAGE_LOCK_MIN_SECONDS = 1.5


def _common_prefix_length(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _collapse_repetitions(words: list[str], max_phrase_len: int = 4) -> list[str]:
    """Collapse immediately repeated phrases: "niż niż" -> "niż",
    "modele modele rozpoznawania" -> "modele rozpoznawania",
    "z do nie z do nie" -> "z do nie".

    Whisper's streaming decode (a growing window re-decoded every step) is
    prone to repeating an n-gram across a window boundary; left in, the
    duplication reaches the committed caption. Measured on the stress
    harness: it accounted for roughly a third of the residual streaming WER
    on clips whose offline transcription was clean.

    Runs to a fixpoint for phrase lengths 1..max_phrase_len. Tradeoff, made
    deliberately: a speaker genuinely repeating a word for emphasis ("tak,
    tak") is merged into one word. For a live lower-third that is the right
    side of the trade -- a stuck repetition loop is far more visible than a
    lost emphasis.
    """
    collapsed = words
    changed = True
    while changed:
        changed = False
        for n in range(1, max_phrase_len + 1):
            out: list[str] = []
            i = 0
            while i < len(collapsed):
                if i + 2 * n <= len(collapsed) and collapsed[i : i + n] == collapsed[i + n : i + 2 * n]:
                    out.extend(collapsed[i : i + n])
                    i += 2 * n
                    changed = True
                else:
                    out.append(collapsed[i])
                    i += 1
            collapsed = out
    return collapsed


def _agreed_prefix_length(hypotheses: list[list[str]]) -> int:
    """Length of the prefix shared identically by every hypothesis in the list
    -- the standard "local agreement" measure: a word is confirmed once it has
    appeared at the same position in each of the last `agreement_passes`
    decode-step hypotheses.
    """
    common_len = len(hypotheses[0])
    for hypothesis in hypotheses[1:]:
        common_len = min(common_len, _common_prefix_length(hypothesis, hypotheses[0]))
    return common_len


class StreamingDecoder:
    def __init__(
        self,
        backend: WhisperBackend,
        vad: SileroVad,
        config: AsrConfig,
        on_event: Callable[[CaptionEvent], None],
        sample_rate: int = 16000,
        clock: Callable[[], float] = time.monotonic,
        min_decode_seconds: float = _MIN_DECODE_SECONDS,
    ) -> None:
        self._backend = backend
        self._vad = vad
        self._config = config
        self._on_event = on_event
        self._window = AudioWindow(sample_rate=sample_rate)
        self._clock = clock
        self._min_decode_seconds = min_decode_seconds

        self._in_speech = False
        self._silence_ms_accum = 0.0
        self._last_decode_monotonic = 0.0
        self._last_speech_monotonic: float | None = None
        self._cleared_since_silence = True

        self._recent_hypotheses: list[list[str]] = []  # last `agreement_passes` hypotheses
        self._committed_word_count = 0
        self._last_final_last_word: str | None = None  # cross-commit repetition guard (see _drop_repeated_boundary)
        # Per-utterance language lock; set in _start_utterance when config
        # language is "auto" (see there for why per-step detection is wrong).
        self._locked_language: str | None = None
        # Dynamic mode (forced language + language_switch_threshold > 0):
        # active decode language and consecutive different-language detection
        # streak (evaluated once per utterance finalize, after the finalize
        # decode).
        self._active_language: str = self._config.language
        self._language_streak: int = 0
        self._last_overrun_warning = float("-inf")

        self._on_event(CaptionEvent(CaptionEventType.STATUS, backend.resolved_device_description))

    def feed(self, frame: np.ndarray, *, run_decode_step: bool = True) -> None:
        """Feed one mono float32 frame at the configured sample rate.

        `run_decode_step=False` ingests the frame (VAD, window append,
        silence-triggered finalization) without ever starting a periodic
        decode. The ASR worker uses that to drain a backlog of queued frames
        cheaply and then call `tick()` exactly once -- see `tick` for why
        that separation matters. The default keeps the original
        ingest-and-maybe-decode behavior for direct callers.
        """
        probabilities = self._vad.process(frame)
        now = self._clock()

        for probability in probabilities:
            if probability >= self._config.vad_threshold:
                if not self._in_speech:
                    self._start_utterance()
                self._silence_ms_accum = 0.0
                self._last_speech_monotonic = now
                self._cleared_since_silence = False
            else:
                self._silence_ms_accum += _VAD_WINDOW_MS

        if self._in_speech:
            self._window.append(frame)
            self._window.trim_to_last_seconds(self._config.max_context_seconds)

            if self._silence_ms_accum >= self._config.end_silence_ms:
                self._finalize_utterance()
                return

            if run_decode_step:
                self._maybe_run_decode_step(now)
        else:
            self._maybe_emit_clear(now)

    def tick(self) -> None:
        """Run a periodic decode step if `step_ms` has elapsed.

        Exists so audio ingest can be decoupled from inference. A decode step
        takes far longer than one frame's worth of audio (measured: 0.3-2.1s
        per step for medium/large-v3 on an RX 6700 XT, against 30ms frames),
        so a loop that decodes inside every `feed()` consumes roughly one
        frame per decode while the microphone produces ~33 per second. The
        audio queue then saturates and starts dropping frames, and the decode
        window silently fills with discontiguous audio -- which is what
        produced garbled, repetitive captions. Draining all pending frames
        with `run_decode_step=False` and then calling `tick()` once bounds
        the work per decode to "whatever arrived since the last one".
        """
        if self._in_speech:
            self._maybe_run_decode_step(self._clock())

    def _maybe_run_decode_step(self, now: float) -> None:
        if (now - self._last_decode_monotonic) * 1000.0 >= self._config.step_ms:
            self._run_decode_step(now)

    def _start_utterance(self) -> None:
        self._in_speech = True
        self._window.clear()
        self._recent_hypotheses = []
        self._committed_word_count = 0
        self._last_decode_monotonic = 0.0
        # Language lock: reset here, then fixed for the whole utterance.
        # With a forced config language that is just the config value; with
        # "auto" it is detected ONCE on the first decode step
        # (_ensure_language_locked) and used verbatim -- whatever language the
        # model hears is what gets transcribed. Detection must happen on the
        # first decode step, not here: the window was just cleared and has no
        # audio yet. Per-step detection flaps between acoustically similar
        # languages (measured: Polish intermittently heard as Ukrainian),
        # producing mixed-language captions; one decision per utterance keeps
        # each caption in a single language.
        self._locked_language: str | None = (
            self._active_language if self._config.language != "auto" else None
        )
        # `self._last_final_last_word` deliberately persists across utterances:
        # a stutter at an utterance boundary would otherwise produce adjacent
        # FINALs ending/starting with the same word.

    def _ensure_language_locked(self, force: bool = False) -> None:
        """Resolve the decode language for the current utterance.

        Two modes:

        * language == "auto": detect and lock this utterance's language
          verbatim. Locks only when the window holds
          `_LANGUAGE_LOCK_MIN_SECONDS` of audio AND detection confidence
          clears `_LANGUAGE_PROB_THRESHOLD`; `force` (the finalize path) drops
          the length requirement but still honors the confidence gate. Until
          locked, decode steps are skipped -- output in an unverified language
          would be garbage we'd have to suppress anyway.

        * forced language + language_switch_threshold > 0 (dynamic mode): the
          configured language is the active decode language. At each utterance
          finalize -- after the finalize decode, so the detection's encoder
          pass never lands on the finalize latency path -- the window's
          language is detected; confident detections of a language DIFFERENT
          from the active one advance a streak, and at
          `language_switch_threshold` consecutive hits the active language
          switches. Decoding never waits in this mode: a foreign-language
          utterance is first captioned in the default language (imperfect
          captions beat delayed ones) and the switch applies from the next
          utterance onward. The streak requires confidence on every hit, so
          one ambiguous finalize does not flip the venue.
        """
        if self._config.language == "auto":
            if self._locked_language is not None:
                return
            if not force and self._window.duration_seconds < _LANGUAGE_LOCK_MIN_SECONDS:
                return
            result = self._backend.detect_language(self._window.samples)
            if result is None:
                return  # detection error; retry on a later step / at finalize
            language, probability = result
            if probability < _LANGUAGE_PROB_THRESHOLD:
                logger.debug(
                    "language detection inconclusive (%s @ %.2f); collecting more audio",
                    language,
                    probability,
                )
                return
            self._locked_language = language
            return

        # Dynamic mode over a forced base language.
        if self._config.language_switch_threshold <= 0:
            return  # switching disabled: the forced language is permanent
        result = self._backend.detect_language(self._window.samples)
        if result is None:
            return
        detected, probability = result
        if probability < _LANGUAGE_PROB_THRESHOLD:
            return  # inconclusive: neither confirms nor resets the streak
        if detected == self._active_language:
            self._language_streak = 0
            return
        self._language_streak += 1
        logger.info(
            "detected %s (%.2f) while decoding as %s -- streak %d/%d",
            detected,
            probability,
            self._active_language,
            self._language_streak,
            self._config.language_switch_threshold,
        )
        if self._language_streak >= self._config.language_switch_threshold:
            logger.info(
                "switching active caption language %s -> %s after %d consecutive detections",
                self._active_language,
                detected,
                self._language_streak,
            )
            self._active_language = detected
            self._locked_language = detected
            self._language_streak = 0

    def _run_decode_step(self, now: float) -> None:
        self._last_decode_monotonic = now
        if self._window.duration_seconds < self._min_decode_seconds:
            return

        # Auto language: before the lock exists we deliberately decode nothing
        # -- the window is shorter than the reliable-detection minimum, and a
        # decode in an unverified language produces text we would have to
        # suppress anyway. Skipping keeps the GPU free; the post-lock decode
        # transcribes the whole window (including this early audio) at once.
        if self._config.language == "auto" and self._locked_language is None:
            self._ensure_language_locked()
            if self._locked_language is None:
                return
        window_seconds = self._window.duration_seconds
        started = time.monotonic()
        try:
            segments = self._backend.transcribe(self._window.samples, language=self._locked_language)
        except Exception:
            logger.exception("ASR decode step failed; skipping this step")
            return
        self._note_decode_duration(time.monotonic() - started, window_seconds)

        text = " ".join(seg_text for _, _, seg_text in segments if seg_text).strip()
        words = _collapse_repetitions(text.split())

        self._recent_hypotheses.append(words)
        if len(self._recent_hypotheses) > self._config.agreement_passes:
            self._recent_hypotheses.pop(0)

        if len(self._recent_hypotheses) >= self._config.agreement_passes:
            agreed_len = _agreed_prefix_length(self._recent_hypotheses)
            if agreed_len > self._committed_word_count:
                new_words = self._drop_repeated_boundary(words, self._committed_word_count, agreed_len)
                if new_words:
                    self._emit_final(" ".join(new_words))
                self._committed_word_count = agreed_len

        if self._config.show_partial_captions:
            partial_words = words[self._committed_word_count :]
            if partial_words:
                self._on_event(CaptionEvent(CaptionEventType.PARTIAL, " ".join(partial_words)))

    def _finalize_utterance(self) -> None:
        # Dynamic mode: the streak check runs AFTER the finalize decode below
        # (via _ensure_language_locked at the end of this method): the
        # detection's encoder pass costs ~445ms on large-v3 (constant in slice
        # length -- the mel is padded to
        # a fixed 30s window), and its result never affects the utterance
        # being finalized (the switch applies from the NEXT utterance), so
        # running it first would add that cost to every utterance's finalize
        # latency for zero benefit. It runs on a copy of the window samples,
        # taken before the window is cleared.
        # Auto mode is different: its lock decides the finalize decode's
        # language, so its detection must run BEFORE the decode.
        is_dynamic = self._config.language != "auto" and self._config.language_switch_threshold > 0
        if self._window.duration_seconds >= self._min_decode_seconds:
            if self._config.language == "auto":
                self._ensure_language_locked(force=True)
                # If the lock never landed (detection failing or never
                # confident enough), transcribing here would run in the
                # backend's default language -- suppressing beats emitting
                # wrong-language text.
                if self._locked_language is None:
                    logger.warning(
                        "utterance finalized without a language lock; dropping %.1fs window",
                        self._window.duration_seconds,
                    )
                    self._in_speech = False
                    self._window.clear()
                    self._recent_hypotheses = []
                    self._committed_word_count = 0
                    self._silence_ms_accum = 0.0
                    return
            window_seconds = self._window.duration_seconds
            started = time.monotonic()
            try:
                segments = self._backend.transcribe(
                    self._window.samples, language=self._locked_language
                )
                self._note_decode_duration(time.monotonic() - started, window_seconds)
                text = " ".join(seg_text for _, _, seg_text in segments if seg_text).strip()
                words = _collapse_repetitions(text.split())
                remaining = self._drop_repeated_boundary(words, self._committed_word_count, len(words))
                if remaining:
                    self._emit_final(" ".join(remaining))
            except Exception:
                logger.exception("Final decode pass failed for utterance")

        self._in_speech = False
        # Dynamic mode: exactly ONE detection per finalize, on the still-
        # intact window (cleared just below) -- advances the switch streak
        # for the NEXT utterance. With threshold=2 the intended sequence
        # holds: the first foreign utterance is still decoded in the default
        # language, and the utterance after the threshold-crossing finalize
        # is decoded in the new language.
        if is_dynamic:
            self._ensure_language_locked()
        # Next utterance starts fresh: auto re-detects, dynamic re-locks on
        # the (possibly just-switched) active language.
        self._locked_language = self._active_language if self._config.language != "auto" else None
        self._window.clear()
        self._recent_hypotheses = []
        self._committed_word_count = 0
        self._silence_ms_accum = 0.0

    def _note_decode_duration(self, elapsed: float, window_seconds: float) -> None:
        """Record how long a decode took relative to the configured cadence.

        A decode step that overruns `step_ms` means the pipeline cannot hit
        the requested caption cadence on this hardware/model, and audio will
        pile up behind it -- past the audio queue's depth it gets dropped
        outright. That is a configuration problem the operator can act on
        (smaller model, longer step_ms, shorter max_context_seconds), so it
        is surfaced rather than silently absorbed. Warnings are rate-limited
        to avoid flooding the log once per step.
        """
        logger.debug(
            "decode step: %.2fs over a %.1fs window (step_ms=%d)",
            elapsed,
            window_seconds,
            self._config.step_ms,
        )
        if elapsed * 1000.0 <= self._config.step_ms * _OVERRUN_WARN_FACTOR:
            return
        now = self._clock()
        if now - self._last_overrun_warning < _OVERRUN_WARN_INTERVAL_SECONDS:
            return
        self._last_overrun_warning = now
        logger.warning(
            "ASR decode step took %.2fs for a %.1fs window, far over step_ms=%dms; "
            "captions will lag and audio may be dropped. Consider a smaller asr.model, "
            "a larger asr.step_ms, or a smaller asr.max_context_seconds.",
            elapsed,
            window_seconds,
            self._config.step_ms,
        )

    def _emit_final(self, text: str) -> None:
        self._on_event(CaptionEvent(CaptionEventType.FINAL, text))

    def _drop_repeated_boundary(self, words: list[str], start: int, end: int) -> list[str]:
        """Newly-committed words for one FINAL, minus a leading word that
        duplicates the utterance's previous FINAL's last word.

        `_collapse_repetitions` only sees one decode hypothesis at a time, so
        it cannot catch a repetition that straddles two commits -- the window
        tail re-decoded at the start of the next window emits the same word
        twice in adjacent FINALs ("... 15" then "15 tysięcy ..."). This
        compares against `self._last_final_last_word`, maintained here, and is
        deliberately case-sensitive like the rest of the word comparison.
        """
        new_words = words[start:end]
        if new_words and new_words[0] == self._last_final_last_word:
            new_words = new_words[1:]
        if new_words:
            self._last_final_last_word = new_words[-1]
        return new_words

    def _maybe_emit_clear(self, now: float) -> None:
        if self._cleared_since_silence or self._last_speech_monotonic is None:
            return
        idle_seconds = now - self._last_speech_monotonic
        if idle_seconds >= self._config.clear_after_silence_seconds:
            self._on_event(CaptionEvent(CaptionEventType.CLEAR))
            self._cleared_since_silence = True
