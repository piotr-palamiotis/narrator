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

The decode window is bounded by ROLLING OVER, never by trimming: continuous
speech is finalized at the smaller of `max_utterance_ms` and
`max_context_seconds` -- preferably at a short pause once the window is
60% full (a clean cut between words), at the cap otherwise -- and the next
speech frame starts a fresh utterance (fresh whisper context, fresh
hypotheses; the language lock carries over, it is the same speech). The
window used to be trimmed to `max_context_seconds` mid-utterance instead,
which shifted every hypothesis under the committed-word count: commits
stalled and the finalize emitted only the tail, dropping ~5s of speech per
long utterance (measured on the reference library: utterances past 20s
carry 40-65% of documentary narration). End-of-utterance silence detection
remains the primary bound in the common case.

Two optional aids, both off by default (`narrator-tune configure` measures
whether they help): `vad_preroll_ms` prepends the audio from just before the
VAD fired (Silero needs a window or two to cross its threshold, so first
syllables get clipped), and `prompt_carryover_chars` hands whisper the tail
of the already-captioned text as its prompt (context for names and
spelling across utterances; cleared by a screen wipe or a language change).

Decode hypotheses are passed through the operator's `text_filters`
(case-insensitive regex find-and-replace, from obs-localvocal) and then
`_collapse_repetitions` before agreement comparison and commit: streaming re-decode is prone to immediate phrase
repetitions ("niż niż") that survive into captions otherwise. See that
function for the tradeoff this makes.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from narrator.asr.asr_backend import AsrBackend, DevicePolicyError
from narrator.asr.decoder_common import (  # noqa: F401  (re-exported: older imports use this module)
    LANGUAGE_LOCK_MIN_SECONDS as _LANGUAGE_LOCK_MIN_SECONDS,
    LANGUAGE_MIN_MARGIN as _LANGUAGE_MIN_MARGIN,
    SOFT_ROLLOVER_FRACTION as _SOFT_ROLLOVER_FRACTION,
    SOFT_ROLLOVER_PAUSE_MS as _SOFT_ROLLOVER_PAUSE_MS,
    VAD_WINDOW_MS as _VAD_WINDOW_MS,
    BoundaryGuard,
    HypothesisText,
    LanguageLock,
    SilenceTimers,
    _collapse_indices,
    _collapse_repetitions,
    _repetition_key,
    align_word_times,
)
from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.text_settings import TextSettings
from narrator.asr.vad import SileroVad
from narrator.audio.ring_buffer import AudioWindow
from narrator.config.schema import AsrConfig

logger = logging.getLogger(__name__)

_MIN_DECODE_SECONDS = 0.3  # skip decode steps on near-empty windows
_OVERRUN_WARN_FACTOR = 2.0  # warn once a decode step takes >2x the configured step
_OVERRUN_WARN_INTERVAL_SECONDS = 10.0
# The language lock's gates (length, margin) and the rollover rule's
# constants live in decoder_common, shared with the segmenter.


def _tail_words(text: str, max_chars: int) -> str:
    """The last `max_chars` characters of `text`, starting at a word."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if cut >= 0 else tail


def _common_prefix_length(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


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
        backend: AsrBackend,
        vad: SileroVad,
        config: AsrConfig,
        on_event: Callable[[CaptionEvent], None],
        sample_rate: int = 16000,
        clock: Callable[[], float] = time.monotonic,
        min_decode_seconds: float = _MIN_DECODE_SECONDS,
        text_settings: TextSettings | None = None,
    ) -> None:
        self._backend = backend
        self._vad = vad
        self._config = config
        self._on_event = on_event
        # Text filters, vocabulary snapping, blocklist, repetition collapse
        # (decoder_common.HypothesisText; snapping/blocklist hot-reload via
        # set_text_settings).
        self._text = HypothesisText(config, text_settings)
        self._window = AudioWindow(sample_rate=sample_rate)
        self._sample_rate = sample_rate
        self._clock = clock
        self._min_decode_seconds = min_decode_seconds

        self._in_speech = False
        self._silence_ms_accum = 0.0
        self._last_decode_monotonic = 0.0
        self._silence_timers = SilenceTimers(config, on_event, on_wipe=self._on_wipe)

        self._recent_hypotheses: list[list[str]] = []  # last `agreement_passes` hypotheses
        self._committed_word_count = 0
        self._boundary = BoundaryGuard()  # cross-commit repetition guard (see _drop_repeated_boundary)
        # Per-utterance language lock (decoder_common.LanguageLock): None
        # until detection clears both the length and margin gates; the
        # previous utterance's language is the ambiguous-finalize fallback;
        # a rollover carries the lock, a real pause re-arms it.
        self._language = LanguageLock(backend, config)
        self._last_overrun_warning = float("-inf")
        # Rollovers so far (diagnostics: autoconfigure reports them).
        self.rollover_count = 0
        # Recent non-speech frames, prepended at the next utterance start.
        self._preroll: deque[np.ndarray] = deque()
        self._preroll_samples = 0
        # Prompt carry-over: tail of earlier utterances' captions and the
        # language they were decoded in.
        self._prompt_text = ""
        self._prompt_language: str | None = None
        self._utterance_text: list[str] = []  # FINAL texts of the current utterance
        # Audio position bookkeeping, so FINALs can say when their words were
        # spoken (CaptionEvent.start/end): samples fed so far, the clock at
        # the latest feed (the worker drains frames right as they arrive, so
        # the newest fed sample is "now"), and where the window starts.
        self._fed_samples = 0
        self._frame_start_sample = 0
        self._feed_clock = 0.0
        self._utterance_start_sample = 0
        # (start, end) seconds into the window for each word of the latest
        # hypothesis, from the backend's word timestamps; None = unknown
        # (FINAL times then fall back to the proportional estimate).
        self._word_times: list[tuple[float, float]] | None = None

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
        self._frame_start_sample = self._fed_samples
        self._fed_samples += len(frame)
        probabilities = self._vad.process(frame)
        now = self._clock()
        self._feed_clock = now

        for probability in probabilities:
            if probability >= self._config.vad_threshold:
                if not self._in_speech:
                    self._start_utterance()
                self._silence_ms_accum = 0.0
                self._silence_timers.speech(now)
            else:
                self._silence_ms_accum += _VAD_WINDOW_MS

        if self._in_speech:
            self._window.append(frame)

            if self._silence_ms_accum >= self._config.end_silence_ms:
                self._finalize_utterance()
                return

            # Long-utterance rollover (module docstring): finalize what is
            # held and let the next speech frame start a fresh utterance --
            # no pause is needed, the next frame re-enters speech at once. A
            # long single context also accumulates errors on noisy audio;
            # measured on a real 88-min recording, fresh contexts transcribe
            # markedly better than one running context.
            if self._should_roll_over():
                self.rollover_count += 1
                self._finalize_utterance(rollover=True)
                return

            if run_decode_step:
                self._maybe_run_decode_step(now)
        else:
            if self._silence_ms_accum >= self._config.end_silence_ms:
                # A real pause after a rollover: whoever speaks next is a new
                # utterance and gets a fresh language decision.
                self._language.on_pause()
            self._remember_preroll(frame)
            self._silence_timers.idle(now)

    def _should_roll_over(self) -> bool:
        cap_seconds = min(self._config.max_utterance_ms / 1000.0, self._config.max_context_seconds)
        held = self._window.duration_seconds
        if held >= cap_seconds:
            return True
        return held >= cap_seconds * _SOFT_ROLLOVER_FRACTION and self._silence_ms_accum >= _SOFT_ROLLOVER_PAUSE_MS

    def _remember_preroll(self, frame: np.ndarray) -> None:
        """Keep the most recent `vad_preroll_ms` of non-speech audio."""
        limit = int(self._config.vad_preroll_ms * self._sample_rate / 1000)
        if limit <= 0:
            return
        self._preroll.append(frame)
        self._preroll_samples += len(frame)
        while self._preroll and self._preroll_samples - len(self._preroll[0]) >= limit:
            self._preroll_samples -= len(self._preroll.popleft())

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
        self._utterance_start_sample = self._frame_start_sample - self._preroll_samples
        # VAD pre-roll: the audio from just before the VAD fired.
        for frame in self._preroll:
            self._window.append(frame)
        self._preroll.clear()
        self._preroll_samples = 0
        self._recent_hypotheses = []
        self._committed_word_count = 0
        self._last_decode_monotonic = 0.0
        self._utterance_text = []
        # Language lock: reset here, then fixed for the whole utterance via
        # _ensure_language_locked on decode steps (detection must happen on a
        # decode step, not here: the window was just cleared and has no audio
        # yet). One decision per utterance keeps each caption in a single
        # language. Speech continuing straight after a rollover keeps its
        # language: it is the same speech, cut only to bound decode cost.
        self._language.start()
        # The boundary guard deliberately persists across utterances:
        # a stutter at an utterance boundary would otherwise produce adjacent
        # FINALs ending/starting with the same word.

    def _ensure_language_locked(self, force: bool = False) -> None:
        """Resolve the decode language for the current utterance, once
        (decoder_common.LanguageLock.ensure). Until it locks, decode steps
        are skipped: output in an unverified language would be garbage we'd
        have to suppress anyway, and the post-lock decode covers the whole
        window at once -- so the deferred decision costs latency only on
        genuinely ambiguous audio."""
        self._language.ensure(self._window.samples, self._window.duration_seconds, force=force)

    # The language state, by the names the tests (and older code) know.
    @property
    def _locked_language(self) -> str | None:
        return self._language.locked

    @property
    def _last_utterance_language(self) -> str | None:
        return self._language.previous

    def _run_decode_step(self, now: float) -> None:
        self._last_decode_monotonic = now
        if self._window.duration_seconds < self._min_decode_seconds:
            return

        # Before the language lock exists we deliberately decode nothing --
        # the window is shorter than the reliable-detection minimum (or the
        # detection is still ambiguous), and a decode in an unverified
        # language produces text we would have to suppress anyway. Skipping
        # keeps the GPU free; the post-lock decode transcribes the whole
        # window (including this early audio) at once.
        if self._locked_language is None:
            self._ensure_language_locked()
            if self._locked_language is None:
                return
        window_seconds = self._window.duration_seconds
        started = time.monotonic()
        try:
            segments = self._backend.transcribe(self._window.samples, **self._decode_kwargs())
        except DevicePolicyError:
            # The configured device failed mid-stream. Re-raising ends the
            # worker process so the supervisor can apply device policy;
            # swallowing it (as any other decode failure) would leave a
            # worker that can never decode again spinning on every step.
            raise
        except Exception:
            logger.exception("ASR decode step failed; skipping this step")
            return
        self._note_decode_duration(time.monotonic() - started, window_seconds)

        words = self._hypothesis_words(segments)
        self._word_times = align_word_times(words, getattr(self._backend, "last_word_times", None))

        self._recent_hypotheses.append(words)
        if len(self._recent_hypotheses) > self._config.agreement_passes:
            self._recent_hypotheses.pop(0)

        if len(self._recent_hypotheses) >= self._config.agreement_passes:
            agreed_len = _agreed_prefix_length(self._recent_hypotheses)
            if agreed_len > self._committed_word_count:
                new_words = self._drop_repeated_boundary(words, self._committed_word_count, agreed_len)
                if new_words:
                    self._emit_final(" ".join(new_words), self._committed_word_count, agreed_len, len(words))
                self._committed_word_count = agreed_len

        if self._config.show_partial_captions:
            partial_words = words[self._committed_word_count :]
            if partial_words:
                self._on_event(CaptionEvent(CaptionEventType.PARTIAL, " ".join(partial_words)))

    def _decode_kwargs(self) -> dict:
        """Per-call decode arguments: the locked language, plus the prompt
        carry-over when enabled and in the same language."""
        kwargs: dict = {"language": self._locked_language}
        if (
            self._config.prompt_carryover_chars > 0
            and self._prompt_text
            and self._prompt_language == self._locked_language
        ):
            kwargs["prompt"] = self._prompt_text
        return kwargs

    def _remember_prompt(self) -> None:
        """Fold the finished utterance's captions into the prompt carry-over
        (a different language starts it afresh)."""
        if self._config.prompt_carryover_chars <= 0 or not self._utterance_text:
            return
        if self._prompt_language != self._locked_language:
            self._prompt_text = ""
        combined = " ".join([self._prompt_text, *self._utterance_text]).strip()
        self._prompt_text = _tail_words(combined, self._config.prompt_carryover_chars)
        self._prompt_language = self._locked_language

    def _finalize_utterance(self, rollover: bool = False) -> None:
        # The language lock decides the finalize decode's language, so the
        # force-lock runs BEFORE the decode. The detection's encoder pass
        # costs ~445ms on large-v3 (constant in slice length -- the mel is
        # padded to a fixed 30s window); it sits on the finalize latency path
        # only when the lock hasn't landed during the utterance, which the
        # margin gate makes rare.
        if self._window.duration_seconds >= self._min_decode_seconds:
            # An ambiguous finalize (margin never cleared) keeps the previous
            # utterance's language rather than coin-flip; with no history
            # (first utterance) the top candidate is used anyway -- dropping
            # the window would hide speech the operator can hear.
            if self._language.for_finalize(self._window.samples, self._window.duration_seconds) is None:
                logger.warning(
                    "language detection failed at finalize; dropping %.1fs window",
                    self._window.duration_seconds,
                )
                self._in_speech = False
                self._window.clear()
                self._recent_hypotheses = []
                self._committed_word_count = 0
                self._silence_ms_accum = 0.0
                self._utterance_text = []
                return
            window_seconds = self._window.duration_seconds
            started = time.monotonic()
            try:
                segments = self._backend.transcribe(self._window.samples, **self._decode_kwargs())
                self._note_decode_duration(time.monotonic() - started, window_seconds)
                words = self._hypothesis_words(segments)
                self._word_times = align_word_times(words, getattr(self._backend, "last_word_times", None))
                remaining = self._drop_repeated_boundary(words, self._committed_word_count, len(words))
                if remaining:
                    self._emit_final(" ".join(remaining), self._committed_word_count, len(words), len(words))
            except Exception:
                logger.exception("Final decode pass failed for utterance")

        self._remember_prompt()
        self._in_speech = False
        # Next utterance starts fresh; remember this one's language for the
        # ambiguity fallback above (and, after a rollover, to continue in).
        self._language.finish(rollover)
        self._window.clear()
        self._recent_hypotheses = []
        self._committed_word_count = 0
        self._utterance_text = []
        # Not reset after a rollover: speech has not paused, and a pause
        # already under way should keep counting toward end_silence_ms.
        if not rollover:
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

    def set_text_settings(self, settings: TextSettings) -> None:
        """Apply new vocabulary-snap/blocklist settings (live: the worker
        calls this on a hot-reload; the next decode step uses them). The
        prompt vocabulary is the backend's (AsrBackend.set_vocabulary)."""
        self._text.set_text_settings(settings)

    def _hypothesis_words(self, segments: list[tuple[float, float, str]]) -> list[str]:
        """One decode's segments as hypothesis words (decoder_common.
        HypothesisText: text filters, snapping, blocklist, repetition
        collapse -- all before agreement)."""
        return self._text.words(" ".join(seg_text for _, _, seg_text in segments if seg_text))

    def _emit_final(self, text: str, first_word: int, end_word: int, hypothesis_words: int) -> None:
        """Emit words [first_word, end_word) of a `hypothesis_words`-word
        hypothesis as a FINAL, stamped with when they were spoken: from
        whisper's word timestamps of that decode when the backend reports
        them (AsrBackend.last_word_times; token-timestamp heuristic, or
        DTW with asr.dtw_word_timestamps), else words are placed
        proportionally over the window's speech (the window minus its
        trailing silence). Measured on Piper clips with known pauses
        (word-boundary error, mean): proportional 0.4 s (small) / 2.0 s
        (large-v3, inflated by repetition loops), token timestamps
        0.14 / 0.25 s, DTW 0.15 / 0.07 s."""
        self._utterance_text.append(text)
        times = self._word_times
        if times is not None and len(times) == hypothesis_words and end_word > first_word:
            start_offset = times[first_word][0] * self._sample_rate
            end_offset = max(times[end_word - 1][1] * self._sample_rate, start_offset)
        else:
            speech_samples = max(
                0.0, len(self._window.samples) - self._silence_ms_accum * self._sample_rate / 1000.0
            )
            per_word = speech_samples / max(hypothesis_words, 1)
            start_offset = per_word * first_word
            end_offset = per_word * end_word
        start = self._sample_clock(self._utterance_start_sample + start_offset)
        end = self._sample_clock(self._utterance_start_sample + end_offset)
        self._on_event(CaptionEvent(CaptionEventType.FINAL, text, start=start, end=end))

    def _sample_clock(self, sample: float) -> float:
        """The decoder clock at which audio sample `sample` was fed."""
        return self._feed_clock - (self._fed_samples - sample) / self._sample_rate

    def _drop_repeated_boundary(self, words: list[str], start: int, end: int) -> list[str]:
        """Newly-committed words [start, end) for one FINAL, minus a leading
        word that duplicates the previous FINAL's last word
        (decoder_common.BoundaryGuard)."""
        return self._boundary.apply(words[start:end])

    def _on_wipe(self) -> None:
        """A silence WIPE cleared the screen: the prompt context goes too."""
        self._prompt_text = ""
        self._prompt_language = None
