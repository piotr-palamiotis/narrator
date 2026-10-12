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

Two optional aids, both off by default (`narrator --autoconfigure` measures
whether they help): `vad_preroll_ms` prepends the audio from just before the
VAD fired (Silero needs a window or two to cross its threshold, so first
syllables get clipped), and `prompt_carryover_chars` hands whisper the tail
of the already-captioned text as its prompt (context for names and
spelling across utterances; cleared by a screen wipe or a language change).

Decode hypotheses are passed through `_collapse_repetitions` before agreement
comparison and commit: streaming re-decode is prone to immediate phrase
repetitions ("niż niż") that survive into captions otherwise. See that
function for the tradeoff this makes.
"""
from __future__ import annotations

import logging
import time
import unicodedata
from collections import deque
from collections.abc import Callable

import numpy as np

from narrator.asr.backend import DevicePolicyError, WhisperBackend
from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.vad import SileroVad
from narrator.audio.ring_buffer import AudioWindow
from narrator.config.schema import AsrConfig

logger = logging.getLogger(__name__)

_VAD_WINDOW_MS = 32.0  # fixed by the Silero graph's 512-sample-at-16kHz requirement
_MIN_DECODE_SECONDS = 0.3  # skip decode steps on near-empty windows
_OVERRUN_WARN_FACTOR = 2.0  # warn once a decode step takes >2x the configured step
_OVERRUN_WARN_INTERVAL_SECONDS = 10.0
# Language selection is a per-utterance decision over the configured
# whitelist (backend.detect_language returns the best allowed language plus
# the probability MARGIN to the runner-up). Two gates keep a marginal
# decision from being locked as if it were sure:
#   length: short slices misdetect (large-v3: 0.4s slices get 1-in-7 wrong;
#   >=1.5s detects correctly), so nothing locks before the window holds
#   this much audio -- decode steps are skipped until then, and the
#   post-lock decode covers the whole window at once.
#   margin: when the top two candidates are close (a 3-language split can
#   sit near 33/33/33), the decision is deferred and the growing window is
#   re-detected every step -- which is the "build confidence up" effect:
#   each step sees more speech until the gap is decisive. At force-finalize
#   an ambiguous result keeps the PREVIOUS utterance's language (a wrong-but
#   -consistent caption beats a coin flip that mixes languages mid-speech);
#   with no history, the top candidate is used anyway (suppressing the
#   finalize decode would drop audible speech the operator can hear).
_LANGUAGE_LOCK_MIN_SECONDS = 1.5
_LANGUAGE_MIN_MARGIN = 0.15
# Long-utterance rollover (see the module docstring): once the window holds
# this fraction of the cap, the first pause at least this long ends the
# utterance -- between-word gaps are ~0-100ms, between phrases 150ms+.
_SOFT_ROLLOVER_FRACTION = 0.6
_SOFT_ROLLOVER_PAUSE_MS = 160.0


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


def _repetition_key(word: str) -> str:
    """A word as repetition detection compares it: case and punctuation
    ignored, because whisper's loops vary both -- measured on the reference
    library, "Zanim przebiegną tędy nowe jezdnie, zanim przebiegną tędy nowe
    jezdnie," survived an exact-token comparison."""
    return "".join(ch for ch in word.lower() if not unicodedata.category(ch).startswith("P"))


def _collapse_indices(keys: list[str], max_phrase_len: int) -> list[int]:
    """Indices into `keys` that survive collapsing immediately repeated
    phrases (the first occurrence is kept), run to a fixpoint for phrase
    lengths 1..max_phrase_len."""
    kept = list(range(len(keys)))
    changed = True
    while changed:
        changed = False
        for n in range(1, max_phrase_len + 1):
            if 2 * n > len(kept):
                break
            current = [keys[i] for i in kept]
            out: list[int] = []
            i = 0
            while i < len(kept):
                if i + 2 * n <= len(kept) and current[i : i + n] == current[i + n : i + 2 * n]:
                    out.extend(kept[i : i + n])
                    i += 2 * n
                    changed = True
                else:
                    out.append(kept[i])
                    i += 1
            kept = out
    return kept


def _collapse_repetitions(words: list[str], max_phrase_len: int = 32) -> list[str]:
    """Collapse immediately repeated phrases: "niż niż" -> "niż",
    "modele modele rozpoznawania" -> "modele rozpoznawania",
    "z do nie z do nie" -> "z do nie", and whole-sentence loops.

    Whisper's streaming decode (a growing window re-decoded every step) is
    prone to repeating an n-gram across a window boundary; left in, the
    duplication reaches the committed caption. Measured on the stress
    harness: word/short-phrase repeats accounted for roughly a third of the
    residual streaming WER on clips whose offline transcription was clean,
    and large-v3's greedy decoding additionally produces whole-sentence
    repetition loops (the same sentence emitted several times inside a
    single hypothesis, until the token cap) -- whisper.cpp has no
    no_repeat_ngram_size knob and beam search is unusable in pywhispercpp
    1.5.1, so the text layer is the only place this can be broken.

    Repeats are detected on `_repetition_key` (case and punctuation
    ignored) and the first occurrence's original words are kept. Phrases up
    to 32 words: a 17-word sentence loop was measured on the reference
    library, past the old 16-word limit.

    Tradeoffs, made deliberately: a speaker genuinely repeating a word for
    emphasis ("tak, tak") or a short phrase is merged into one instance, and
    a genuinely repeated long sentence (chorus lines, read-backs) likewise
    shows once. For a live lower-third that is the right side of the trade
    -- a stuck repetition loop is far more visible than a lost emphasis.
    """
    keys = [_repetition_key(w) for w in words]
    return [words[i] for i in _collapse_indices(keys, max_phrase_len)]


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
        self._sample_rate = sample_rate
        self._clock = clock
        self._min_decode_seconds = min_decode_seconds

        self._in_speech = False
        self._silence_ms_accum = 0.0
        self._last_decode_monotonic = 0.0
        self._last_speech_monotonic: float | None = None
        self._pruned_since_silence = True
        self._wiped_since_silence = True

        self._recent_hypotheses: list[list[str]] = []  # last `agreement_passes` hypotheses
        self._committed_word_count = 0
        self._last_final_last_key: str | None = None  # cross-commit repetition guard (see _drop_repeated_boundary)
        # Per-utterance language lock: None until detection clears both the
        # length and margin gates (see the constants' comment). One decision
        # per utterance -- per-step detection flaps between acoustically
        # similar languages (measured: Polish intermittently heard as
        # Ukrainian), producing mixed-language captions.
        self._locked_language: str | None = None
        # Language of the previous utterance: an ambiguous finalize keeps
        # this instead of coin-flipping (see _finalize_utterance).
        self._last_utterance_language: str | None = None
        # Language carried across a long-utterance rollover (the same
        # continuous speech); expires if a real pause follows.
        self._rollover_language: str | None = None
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
                self._pruned_since_silence = False
                self._wiped_since_silence = False
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
                self._rollover_language = None
            self._remember_preroll(frame)
            self._maybe_emit_prune(now)
            self._maybe_emit_wipe(now)

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
        self._locked_language = self._rollover_language
        self._rollover_language = None
        # `self._last_final_last_key` deliberately persists across utterances:
        # a stutter at an utterance boundary would otherwise produce adjacent
        # FINALs ending/starting with the same word.

    def _ensure_language_locked(self, force: bool = False) -> None:
        """Resolve the decode language for the current utterance, once.

        Locks when the window holds `_LANGUAGE_LOCK_MIN_SECONDS` of audio AND
        detection clears the margin gate (`_LANGUAGE_MIN_MARGIN` between the
        top two allowed candidates). Until then decode steps are skipped:
        output in an unverified language would be garbage we'd have to
        suppress anyway, and the post-lock decode covers the whole window at
        once -- so the deferred decision costs latency only on genuinely
        ambiguous audio. `force` (the finalize path) drops the length
        requirement; the margin gate degrades to 'keep the previous
        utterance's language on ambiguity' there (see _finalize_utterance).
        """
        if self._locked_language is not None:
            return
        # Single-entry whitelist: the language is known from config, no
        # detection, no waiting -- this is the permanent-forced mode.
        if len(self._config.language_whitelist) == 1:
            self._locked_language = self._config.language_whitelist[0]
            return
        if not force and self._window.duration_seconds < _LANGUAGE_LOCK_MIN_SECONDS:
            return
        result = self._backend.detect_language(self._window.samples)
        if result is None:
            return  # detection error; retry on a later step / at finalize
        language, probability, margin = result
        # The margin gate applies at force too: finalize handles ambiguity by
        # keeping the previous utterance's language (see _finalize_utterance).
        if margin < _LANGUAGE_MIN_MARGIN:
            logger.debug(
                "language detection ambiguous (%s @ %.2f, margin %.2f); collecting more audio",
                language,
                probability,
                margin,
            )
            return
        self._locked_language = language

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
            self._ensure_language_locked(force=True)
            # An ambiguous finalize (margin never cleared): keep the previous
            # utterance's language rather than coin-flip -- a wrong-but-
            # consistent caption beats one that mixes languages mid-speech.
            # With no history (first utterance), use the top candidate anyway:
            # dropping the window would hide speech the operator can hear.
            if self._locked_language is None:
                if self._last_utterance_language is not None:
                    self._locked_language = self._last_utterance_language
                    logger.info(
                        "language still ambiguous at finalize; keeping previous utterance's language %s",
                        self._locked_language,
                    )
                else:
                    result = self._backend.detect_language(self._window.samples)
                    if result is None:
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
                    self._locked_language = result[0]
            window_seconds = self._window.duration_seconds
            started = time.monotonic()
            try:
                segments = self._backend.transcribe(self._window.samples, **self._decode_kwargs())
                self._note_decode_duration(time.monotonic() - started, window_seconds)
                text = " ".join(seg_text for _, _, seg_text in segments if seg_text).strip()
                words = _collapse_repetitions(text.split())
                remaining = self._drop_repeated_boundary(words, self._committed_word_count, len(words))
                if remaining:
                    self._emit_final(" ".join(remaining))
            except Exception:
                logger.exception("Final decode pass failed for utterance")

        self._remember_prompt()
        self._in_speech = False
        # Next utterance starts fresh; remember this one's language for the
        # ambiguity fallback above (and, after a rollover, to continue in).
        self._last_utterance_language = self._locked_language
        self._rollover_language = self._locked_language if rollover else None
        self._locked_language = None
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

    def _emit_final(self, text: str) -> None:
        self._utterance_text.append(text)
        self._on_event(CaptionEvent(CaptionEventType.FINAL, text))

    def _drop_repeated_boundary(self, words: list[str], start: int, end: int) -> list[str]:
        """Newly-committed words for one FINAL, minus a leading word that
        duplicates the utterance's previous FINAL's last word.

        `_collapse_repetitions` only sees one decode hypothesis at a time, so
        it cannot catch a repetition that straddles two commits -- the window
        tail re-decoded at the start of the next window emits the same word
        twice in adjacent FINALs ("... 15" then "15 tysięcy ..."). This
        compares against `self._last_final_last_key`, maintained here, with
        the same case/punctuation-blind key as `_collapse_repetitions`.
        """
        new_words = words[start:end]
        if new_words and _repetition_key(new_words[0]) == self._last_final_last_key:
            new_words = new_words[1:]
        if new_words:
            self._last_final_last_key = _repetition_key(new_words[-1])
        return new_words

    def _maybe_emit_prune(self, now: float) -> None:
        """Idle past `prune_history_after_silence_seconds`: CLEAR, which drops
        only caption lines that are already invisible under layout.max_lines
        (memory hygiene -- memory is also bounded by the buffer's history cap,
        so this is not load-bearing for memory). The display is untouched.
        """
        if self._pruned_since_silence or self._last_speech_monotonic is None:
            return
        idle_seconds = now - self._last_speech_monotonic
        if idle_seconds >= self._config.prune_history_after_silence_seconds:
            self._on_event(CaptionEvent(CaptionEventType.CLEAR))
            self._pruned_since_silence = True

    def _maybe_emit_wipe(self, now: float) -> None:
        """Idle past `wipe_screen_after_silence_seconds`: WIPE, a true screen
        clear, same as the operator's Ctrl+N. 0 disables it -- the default,
        because stale captions are usually better than blank ones and the
        operator knows their venue. Nonzero exists for broadcasts where a
        very old sentence would misattribute to unrelated new speech.
        """
        if (
            self._config.wipe_screen_after_silence_seconds <= 0
            or self._wiped_since_silence
            or self._last_speech_monotonic is None
        ):
            return
        idle_seconds = now - self._last_speech_monotonic
        if idle_seconds >= self._config.wipe_screen_after_silence_seconds:
            self._on_event(CaptionEvent(CaptionEventType.WIPE))
            self._wiped_since_silence = True
            # The screen starts over, and so does the context.
            self._prompt_text = ""
            self._prompt_language = None
