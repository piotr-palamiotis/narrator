"""What every caption decoder shares, whatever its streaming strategy.

The ASR worker and stream_sim drive a decoder through one small surface
(`CaptionDecoder`): drain-then-tick ingest (`feed(frame, run_decode_step=
False)` for every queued frame, then one `tick()`), events through the
`on_event` callback, and live text settings. Two implementations exist,
picked by `asr.decoder` (`create_decoder`):

  * `local_agreement` -- narrator.asr.streaming_decoder.StreamingDecoder:
    re-decode a growing window every step_ms, commit the prefix the last
    `agreement_passes` hypotheses agree on.
  * `segmenter` -- narrator.asr.segmenter.Segmenter (obs-localvocal
    architecture): VAD-cut segments decoded once when they close (FINAL),
    optional whole-segment peeks while open (PARTIAL).

The guards both must keep live here, so neither carries its own copy: the
hypothesis text pipeline (operator text filters -> vocabulary snapping ->
blocklist -> repetition collapse), the cross-FINAL boundary dedupe, the
per-utterance language lock over the whitelist, and the silence timers
(prune/wipe, which must never blank what the audience is reading).
"""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from collections.abc import Callable
from typing import Protocol

import numpy as np

from narrator.asr.asr_backend import AsrBackend
from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.text_settings import TextSettings
from narrator.asr.vocabulary import VocabularySnapper

logger = logging.getLogger(__name__)

# Language selection is a per-utterance decision over the configured
# whitelist (backend.detect_language returns the best allowed language plus
# the probability MARGIN to the runner-up). Two gates keep a marginal
# decision from being locked as if it were sure:
#   length: short slices misdetect (large-v3: 0.4s slices get 1-in-7 wrong;
#   >=1.5s detects correctly), so nothing locks before the window holds
#   this much audio -- decodes are skipped until then, and the post-lock
#   decode covers the whole window at once.
#   margin: when the top two candidates are close (a 3-language split can
#   sit near 33/33/33), the decision is deferred and the growing window is
#   re-detected on later steps -- which is the "build confidence up" effect.
#   At finalize an ambiguous result keeps the PREVIOUS utterance's language
#   (a wrong-but-consistent caption beats a coin flip that mixes languages
#   mid-speech); with no history, the top candidate is used anyway
#   (suppressing the finalize decode would drop audible speech).
LANGUAGE_LOCK_MIN_SECONDS = 1.5
LANGUAGE_MIN_MARGIN = 0.15
# One Silero window: fixed by the graph's 512-samples-at-16kHz requirement.
VAD_WINDOW_MS = 32.0
# Long-speech cut ("rollover"): once the held audio reaches this fraction of
# the cap, the first pause at least this long ends it -- between-word gaps
# are ~0-100ms, between phrases 150ms+; otherwise the cap itself does.
SOFT_ROLLOVER_FRACTION = 0.6
SOFT_ROLLOVER_PAUSE_MS = 160.0


class CaptionDecoder(Protocol):
    """The surface the worker and stream_sim use (see the module docstring)."""

    rollover_count: int

    def feed(self, frame: np.ndarray, *, run_decode_step: bool = True) -> None: ...

    def tick(self) -> None: ...

    def set_text_settings(self, settings: TextSettings) -> None: ...


def create_decoder(backend: AsrBackend, vad, config, on_event: Callable[[CaptionEvent], None], **kwargs) -> CaptionDecoder:
    """The decoder `config.decoder` selects, with the shared constructor
    arguments (sample_rate, clock, min_decode_seconds, text_settings)."""
    decoder = getattr(config.decoder, "value", config.decoder)
    if decoder == "segmenter":
        from narrator.asr.segmenter import Segmenter

        return Segmenter(backend, vad, config, on_event, **kwargs)
    from narrator.asr.streaming_decoder import StreamingDecoder

    return StreamingDecoder(backend, vad, config, on_event, **kwargs)


# ---------------------------------------------------------------------------
# Repetition collapse
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hypothesis text pipeline and the cross-FINAL boundary guard
# ---------------------------------------------------------------------------


class HypothesisText:
    """One decode's text as caption words: the operator's text filters
    (case-insensitive regex find-and-replace), vocabulary snapping, the
    caption blocklist, then repetition collapse -- in that order, and
    before any agreement/commit decision, so a filtered word can't block
    commits.

    Snapping and the blocklist are hot-reloadable (set_text_settings).
    Without explicit settings: the inline vocabulary and no blocklist --
    the raw text an offline evaluation wants."""

    def __init__(self, config, text_settings: TextSettings | None = None) -> None:
        self._filters = [(re.compile(rule.pattern, re.IGNORECASE), rule.replacement) for rule in config.text_filters]
        self._snapper: VocabularySnapper | None = None
        self._blocklist = None
        self.set_text_settings(
            text_settings
            or TextSettings(vocabulary=tuple(config.vocabulary), vocabulary_snap=config.vocabulary_snap)
        )

    def set_text_settings(self, settings: TextSettings) -> None:
        snapper = VocabularySnapper(list(settings.vocabulary)) if settings.vocabulary_snap else None
        self._snapper = snapper or None
        self._blocklist = settings.blocklist or None

    def words(self, text: str) -> list[str]:
        text = text.strip()
        for pattern, replacement in self._filters:
            text = pattern.sub(replacement, text)
        if self._snapper is not None:
            text = self._snapper.apply(text)
        if self._blocklist is not None:
            text = self._blocklist.apply(text)
        return _collapse_repetitions(text.split())


def align_word_times(words: list[str], word_times) -> list[tuple[float, float]] | None:
    """Per-word (start, end) seconds for a hypothesis' caption `words`, from
    the backend's word times of the same decode (AsrBackend.
    last_word_times: the raw words, before text filters, snapping, the
    blocklist and repetition collapse). Words are matched on
    `_repetition_key` (difflib): a kept word takes its own raw word's time;
    a replaced run is spread over the raw run it replaced; an inserted word
    sits at the boundary of its neighbours. None when there are no word
    times (a backend without them, e.g. a test fake), so the caller falls
    back to its own estimate."""
    if not isinstance(word_times, (list, tuple)) or not word_times or not words:
        return None
    raw_keys = [_repetition_key(w.text) for w in word_times]
    keys = [_repetition_key(w) for w in words]
    times: list[tuple[float, float] | None] = [None] * len(words)
    matcher = difflib.SequenceMatcher(a=keys, b=raw_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                raw = word_times[j1 + k]
                times[i1 + k] = (raw.start, raw.end)
        elif tag == "replace":
            start, end = word_times[j1].start, word_times[j2 - 1].end
            n = i2 - i1
            for k in range(n):
                times[i1 + k] = (start + (end - start) * k / n, start + (end - start) * (k + 1) / n)
        elif tag == "insert":
            at = word_times[j1 - 1].end if j1 > 0 else word_times[0].start
            for k in range(i1, i2):
                times[k] = (at, at)
    return [t if t is not None else (0.0, 0.0) for t in times]


class BoundaryGuard:
    """Drops a new FINAL's first word when it duplicates the previous
    FINAL's last word.

    `_collapse_repetitions` only sees one decode at a time, so it cannot
    catch a repetition that straddles two FINALs -- a window tail re-decoded
    at the start of the next window emits the same word twice in adjacent
    FINALs ("... 15" then "15 tysięcy ..."). Compared with the same
    case/punctuation-blind key. Persists across utterances/segments: a
    stutter at a boundary would otherwise produce the same doubled word.
    Tradeoff: an emphatic "tak, tak" across two FINALs becomes "tak".
    """

    def __init__(self) -> None:
        self.last_key: str | None = None

    def apply(self, words: list[str]) -> list[str]:
        words = self.peek(words)
        if words:
            self.last_key = _repetition_key(words[-1])
        return words

    def peek(self, words: list[str]) -> list[str]:
        """`apply` without remembering: for display-only text (a segmenter
        partial must not show the doubled word its FINAL will drop)."""
        if words and _repetition_key(words[0]) == self.last_key:
            return words[1:]
        return words


# ---------------------------------------------------------------------------
# Language lock
# ---------------------------------------------------------------------------


class LanguageLock:
    """The per-utterance (per-segment) language decision over
    `asr.language_whitelist`; see the constants above for the gates.

    One decision per utterance -- per-step detection flaps between
    acoustically similar languages (measured: Polish intermittently heard as
    Ukrainian), producing mixed-language captions. A long-utterance rollover
    carries the lock into the next utterance (it is the same speech); a real
    pause (`on_pause`) re-arms it.
    """

    def __init__(self, backend: AsrBackend, config) -> None:
        self._backend = backend
        self._config = config
        self.locked: str | None = None
        self.previous: str | None = None  # the previous utterance's language
        self.carried: str | None = None  # carried across a rollover

    def start(self) -> None:
        """A new utterance begins (detection waits for audio)."""
        self.locked = self.carried
        self.carried = None

    def on_pause(self) -> None:
        """A real pause: whoever speaks next gets a fresh decision."""
        self.carried = None

    def ensure(self, samples: np.ndarray, seconds: float, force: bool = False) -> None:
        """Lock once the audio holds LANGUAGE_LOCK_MIN_SECONDS AND detection
        clears the margin gate. `force` (the finalize path) drops the length
        requirement only; ambiguity is handled by `for_finalize`."""
        if self.locked is not None:
            return
        # Single-entry whitelist: the language is known from config, no
        # detection, no waiting -- the permanent-forced mode.
        if len(self._config.language_whitelist) == 1:
            self.locked = self._config.language_whitelist[0]
            return
        if not force and seconds < LANGUAGE_LOCK_MIN_SECONDS:
            return
        result = self._backend.detect_language(samples)
        if result is None:
            return  # detection error; retry on a later step / at finalize
        language, probability, margin = result
        if margin < LANGUAGE_MIN_MARGIN:
            logger.debug(
                "language detection ambiguous (%s @ %.2f, margin %.2f); collecting more audio",
                language,
                probability,
                margin,
            )
            return
        self.locked = language

    def for_finalize(self, samples: np.ndarray, seconds: float) -> str | None:
        """The language a finalize decode must use: the lock (forced), else
        the previous utterance's language, else the top candidate. None only
        when detection itself fails with no history to fall back on."""
        self.ensure(samples, seconds, force=True)
        if self.locked is None:
            if self.previous is not None:
                self.locked = self.previous
                logger.info(
                    "language still ambiguous at finalize; keeping previous utterance's language %s",
                    self.locked,
                )
            else:
                result = self._backend.detect_language(samples)
                if result is None:
                    return None
                self.locked = result[0]
        return self.locked

    def finish(self, rollover: bool) -> None:
        """The utterance ended: remember its language for the ambiguity
        fallback and, after a rollover, to continue in."""
        self.previous = self.locked
        self.carried = self.locked if rollover else None
        self.locked = None


# ---------------------------------------------------------------------------
# Silence timers
# ---------------------------------------------------------------------------


class SilenceTimers:
    """CLEAR after `prune_history_after_silence_seconds` and (when enabled)
    WIPE after `wipe_screen_after_silence_seconds` of no speech, each once
    per lull.

    CLEAR drops only caption lines already invisible under layout.max_lines
    -- the display is untouched (memory hygiene). WIPE is a true screen
    clear, same as the operator's Clear; 0 disables it (the default:
    stale captions are usually better than blank ones). `on_wipe` lets the
    decoder drop its prompt context with the screen.
    """

    def __init__(self, config, emit: Callable[[CaptionEvent], None], on_wipe: Callable[[], None] | None = None) -> None:
        self._config = config
        self._emit = emit
        self._on_wipe = on_wipe
        self._last_speech: float | None = None
        self._pruned = True
        self._wiped = True

    def speech(self, now: float) -> None:
        self._last_speech = now
        self._pruned = False
        self._wiped = False

    def idle(self, now: float) -> None:
        if self._last_speech is None:
            return
        idle_seconds = now - self._last_speech
        if not self._pruned and idle_seconds >= self._config.prune_history_after_silence_seconds:
            self._emit(CaptionEvent(CaptionEventType.CLEAR))
            self._pruned = True
        wipe_after = self._config.wipe_screen_after_silence_seconds
        if wipe_after > 0 and not self._wiped and idle_seconds >= wipe_after:
            self._emit(CaptionEvent(CaptionEventType.WIPE))
            self._wiped = True
            if self._on_wipe is not None:
                self._on_wipe()
