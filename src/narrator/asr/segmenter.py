"""VAD-segmented, one-shot decoding: the obs-localvocal architecture
(`asr.decoder: segmenter`), next to the local-agreement StreamingDecoder.

Silero VAD cuts the audio into speech segments; when a segment ends, the
WHOLE segment is decoded once and that result is the FINAL -- no agreement,
no diffing. While a segment is still open, and `show_partial_captions` is
on, the segment so far is re-decoded (a "peek") after every
`partial_latency_ms` of new audio and shown as a PARTIAL that replaces the
previous one; a partial is never committed, the FINAL decode of the closed
segment replaces it. Ported from LocalVocal's vad-processing.cpp
(vad_based_segmentation / hybrid_vad_segmentation /
vad_disabled_segmentation) as one class with `asr.segmentation_mode`:

  vad     a segment is one stretch of speech: it opens at the first speech
          window (pulling in 100 ms of pre-roll) and closes after
          `end_silence_ms` of silence.
  hybrid  everything is accumulated and a segment closes every
          `segment_duration_ms`; peeks run only once Silero has heard speech
          in it, and a segment without speech is kept to its last 250 ms.
  fixed   LocalVocal's "VAD disabled": `segment_duration_ms` segments, no
          VAD at all (so the prune/wipe silence timers never fire).

Narrator's additions, all shared with the local-agreement decoder through
decoder_common:

  * a segment cap (`max_segment_ms`, vad mode; LocalVocal's is infinite):
    the first >=160 ms pause once the segment is 60% full cuts it, the cap
    otherwise; a cut at the cap starts the next segment AT ONCE, so no audio
    is ever trimmed or lost (the guarantee of the StreamingDecoder's
    no-loss rollover). The language lock carries across the cut.
  * the per-segment language lock over `asr.language_whitelist` (>=1.5 s
    and the margin gate; peeks wait for it; a FINAL falls back to the
    previous segment's language, then the top candidate; a real pause
    re-arms it). hybrid/fixed segments are not speech-aligned, so each is
    decided afresh.
  * the hypothesis text pipeline (text filters -> vocabulary snapping ->
    blocklist -> repetition collapse) on PARTIAL and FINAL alike, the
    boundary dedupe against the previous FINAL, and no event for empty or
    punctuation-only text.
  * the silence timers (prune/wipe), driven by VAD silence as before.

Context: the last `context_sentences` FINAL texts are whisper's prompt
(LocalVocal's n_context_sentences), after the event vocabulary that the
backend puts first; reset by a WIPE and when the language changes.
Silero's state is reset on every transition into non-speech (LocalVocal's
rule), on top of the vad.py quiet-reset backstop.

Peek cost (todo B §5): re-decoding the whole open segment gets expensive on
long segments, so the peek interval grows with the segment's length, and a
peek is skipped while the previous one took more than half the worker's
time since it started -- partials give way, FINALs never wait for them.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from narrator.asr.asr_backend import AsrBackend, DevicePolicyError
from narrator.asr.decoder_common import (
    LANGUAGE_LOCK_MIN_SECONDS,
    SOFT_ROLLOVER_FRACTION,
    SOFT_ROLLOVER_PAUSE_MS,
    VAD_WINDOW_MS,
    BoundaryGuard,
    HypothesisText,
    LanguageLock,
    SilenceTimers,
)
from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.text_settings import TextSettings
from narrator.audio.ring_buffer import AudioWindow

logger = logging.getLogger(__name__)

_MIN_DECODE_SECONDS = 0.3  # a segment shorter than this is not decoded
# Pre-roll pulled in at a speech start (vad mode): Silero needs a window or
# two to cross its threshold, which would clip first syllables.
_PREROLL_MS = 100.0
# hybrid mode: a segment in which Silero has heard no speech keeps only this.
_HYBRID_IDLE_KEEP_SECONDS = 0.25
# Peek interval = partial_latency_ms + this many seconds per second of open
# segment (a 15 s segment peeks every ~2.6 s at the 1.1 s default).
_PEEK_INTERVAL_GROWTH = 0.1
# A peek waits until at least this multiple of the previous peek's cost has
# passed since that peek started: peeks take at most half the worker.
_PEEK_LOAD_FACTOR = 2.0
_SLOW_WARN_INTERVAL_SECONDS = 10.0


class Segmenter:
    """Same driving surface as StreamingDecoder (decoder_common.
    CaptionDecoder): feed(frame, run_decode_step=False) for every queued
    frame, then tick() once; events go to `on_event`."""

    def __init__(
        self,
        backend: AsrBackend,
        vad,
        config,
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
        self._sample_rate = sample_rate
        self._clock = clock
        self._min_decode_seconds = min_decode_seconds
        self._mode = getattr(config.segmentation_mode, "value", config.segmentation_mode)

        self._text = HypothesisText(config, text_settings)
        self._boundary = BoundaryGuard()
        self._language = LanguageLock(backend, config)
        self._silence_timers = SilenceTimers(config, on_event, on_wipe=self._reset_context)

        self._window = AudioWindow(sample_rate=sample_rate)
        self._in_segment = False
        self._speech_seen = False  # hybrid: Silero heard speech in this segment
        self._vad_in_speech = False  # for the reset-on-transition rule
        self._silence_ms_accum = 0.0
        self._preroll: deque[np.ndarray] = deque()
        self._preroll_samples = 0

        # Peek scheduling (see the module docstring).
        self._held_at_last_peek = 0.0
        self._peek_started: float | None = None
        self._peek_cost: float | None = None
        self._partial_shown = False

        # Prompt context: the last `context_sentences` FINAL texts and the
        # language they were decoded in.
        self._context: deque[str] = deque(maxlen=max(config.context_sentences, 1))
        self._context_language: str | None = None

        # Audio position bookkeeping for FINAL start/end (spoken time).
        self._fed_samples = 0
        self._frame_start_sample = 0
        self._feed_clock = 0.0
        self._segment_start_sample = 0
        self._last_slow_warning = float("-inf")

        # Diagnostics (stream_sim / narrator-tune report them).
        self.rollover_count = 0
        self.peek_count = 0
        self.peeks_skipped_busy = 0

        self._on_event(CaptionEvent(CaptionEventType.STATUS, backend.resolved_device_description))

    # -- ingest -------------------------------------------------------------

    def feed(self, frame: np.ndarray, *, run_decode_step: bool = True) -> None:
        """Ingest one mono float32 frame: VAD, segment bookkeeping, and the
        FINAL decode of a segment that closes. `run_decode_step=False` never
        starts a peek (the worker drains a backlog this way, then ticks)."""
        self._frame_start_sample = self._fed_samples
        self._fed_samples += len(frame)
        now = self._clock()
        self._feed_clock = now
        if self._mode == "fixed":
            self._feed_fixed(frame)
        else:
            heard_speech = self._run_vad(frame, now)
            if self._mode == "hybrid":
                self._feed_hybrid(frame, heard_speech, now)
            else:
                if not self._feed_vad(frame, now):
                    return
        if run_decode_step:
            self._maybe_peek(now)

    def _run_vad(self, frame: np.ndarray, now: float) -> bool:
        """Silero over the frame's windows; tracks silence, the silence
        timers and the reset-on-transition rule. True when a window held
        speech."""
        heard = False
        for probability in self._vad.process(frame):
            if probability >= self._config.vad_threshold:
                heard = True
                self._vad_in_speech = True
                self._silence_ms_accum = 0.0
                self._silence_timers.speech(now)
                if self._mode == "vad" and not self._in_segment:
                    self._open_segment(self._frame_start_sample, with_preroll=True)
            else:
                self._silence_ms_accum += VAD_WINDOW_MS
        if self._vad_in_speech and self._silence_ms_accum >= self._config.end_silence_ms:
            # The transition into non-speech: fresh Silero state for the next
            # speech stretch (obs-localvocal's rule).
            self._vad_in_speech = False
            reset = getattr(self._vad, "reset_state", None)
            if reset is not None:
                reset()
        return heard

    def _feed_vad(self, frame: np.ndarray, now: float) -> bool:
        """vad mode. Returns False when the frame closed a segment."""
        if not self._in_segment:
            if self._silence_ms_accum >= self._config.end_silence_ms:
                self._language.on_pause()  # a real pause re-arms the language decision
            self._remember_preroll(frame)
            self._silence_timers.idle(now)
            return True
        self._window.append(frame)
        if self._silence_ms_accum >= self._config.end_silence_ms:
            self._close_segment(rollover=False, at_pause=True)
            return False
        cap = self._config.max_segment_ms / 1000.0
        held = self._window.duration_seconds
        if held >= cap:
            # Mid-speech: the next segment starts right here, so nothing
            # falls between the two.
            self.rollover_count += 1
            self._close_segment(rollover=True, at_pause=False)
            self._open_segment(self._fed_samples)
            return False
        if held >= cap * SOFT_ROLLOVER_FRACTION and self._silence_ms_accum >= SOFT_ROLLOVER_PAUSE_MS:
            # A clean cut at a pause; the next speech window opens the next
            # segment (with pre-roll). Only silence is skipped.
            self.rollover_count += 1
            self._close_segment(rollover=True, at_pause=True)
            return False
        return True

    def _feed_hybrid(self, frame: np.ndarray, heard_speech: bool, now: float) -> None:
        if not self._in_segment:
            self._open_segment(self._frame_start_sample)
        self._window.append(frame)
        if heard_speech:
            self._speech_seen = True
        else:
            self._silence_timers.idle(now)
        if not self._speech_seen:
            self._window.trim_to_last_seconds(_HYBRID_IDLE_KEEP_SECONDS)
            self._segment_start_sample = self._fed_samples - len(self._window)
        elif self._window.duration_seconds >= self._config.segment_duration_ms / 1000.0:
            self._close_segment(rollover=False, at_pause=False)

    def _feed_fixed(self, frame: np.ndarray) -> None:
        if not self._in_segment:
            self._open_segment(self._frame_start_sample)
            self._speech_seen = True
        self._window.append(frame)
        if self._window.duration_seconds >= self._config.segment_duration_ms / 1000.0:
            self._close_segment(rollover=False, at_pause=False)

    def _remember_preroll(self, frame: np.ndarray) -> None:
        limit = int(_PREROLL_MS * self._sample_rate / 1000)
        self._preroll.append(frame)
        self._preroll_samples += len(frame)
        while self._preroll and self._preroll_samples - len(self._preroll[0]) >= limit:
            self._preroll_samples -= len(self._preroll.popleft())

    def _open_segment(self, start_sample: int, *, with_preroll: bool = False) -> None:
        """Open a segment whose first new audio is sample `start_sample`
        (plus, at a speech start, the pre-roll before it)."""
        self._in_segment = True
        self._window.clear()
        self._segment_start_sample = start_sample
        if with_preroll:
            self._segment_start_sample -= self._preroll_samples
            for frame in self._preroll:
                self._window.append(frame)
        self._preroll.clear()
        self._preroll_samples = 0
        self._speech_seen = False
        self._held_at_last_peek = 0.0
        self._partial_shown = False
        self._language.start()

    # -- decoding -----------------------------------------------------------

    def tick(self) -> None:
        """Run a peek if one is due (see the module docstring); FINALs are
        decoded in feed(), the moment a segment closes."""
        now = self._clock()
        if self._peek_started is not None and self._peek_cost is None:
            # The previous peek's cost, as seen from the first tick after it
            # (live: the peek ran inside the previous tick; stream_sim: ticks
            # run only once the worker is free again). Measured here, not in
            # _maybe_peek, so a segment closing in between cannot leave it to
            # be read long after -- as a huge cost that would starve peeks.
            self._peek_cost = max(now - self._peek_started, 0.0)
        if self._in_segment:
            self._maybe_peek(now)

    def flush(self) -> None:
        """Finalize an open segment now (end of a replayed stream: hybrid
        and fixed segments otherwise wait for their full duration)."""
        if self._in_segment:
            self._close_segment(rollover=False, at_pause=True)

    def _maybe_peek(self, now: float) -> None:
        if not self._config.show_partial_captions or not self._in_segment:
            return
        if self._mode == "hybrid" and not self._speech_seen:
            return
        held = self._window.duration_seconds
        if held < self._min_decode_seconds:
            return
        interval = self._config.partial_latency_ms / 1000.0 + _PEEK_INTERVAL_GROWTH * held
        if held - self._held_at_last_peek < interval:
            return
        if (
            self._peek_started is not None
            and self._peek_cost is not None
            and now - self._peek_started < _PEEK_LOAD_FACTOR * self._peek_cost
        ):
            self.peeks_skipped_busy += 1
            return
        if self._language.locked is None:
            detects = held >= LANGUAGE_LOCK_MIN_SECONDS and len(self._config.language_whitelist) != 1
            self._language.ensure(self._window.samples, held)
            if self._language.locked is None:
                # No partials in an unverified language. An ambiguous
                # detection is retried at the peek cadence, not every tick:
                # each detection costs a fixed encoder pass.
                if detects:
                    self._held_at_last_peek = held
                return
        self._held_at_last_peek = held
        self._peek_started = now
        self._peek_cost = None
        self.peek_count += 1
        result = self._decode(peek=True, drop_trailing_period=True)
        if result is None:
            return
        words = self._text.words(result.text)
        # Display-only boundary check: a peek must not show the doubled word
        # the FINAL will drop (the guard's state is not advanced here).
        words = self._boundary.peek(words)
        text = " ".join(words)
        if _has_text(text):
            self._partial_shown = True
            self._on_event(CaptionEvent(CaptionEventType.PARTIAL, text))
        elif self._partial_shown:
            self._partial_shown = False
            self._on_event(CaptionEvent(CaptionEventType.PARTIAL, ""))

    def _decode(self, *, peek: bool, drop_trailing_period: bool):
        """One whole-segment decode in the locked language, with the context
        prompt; None on failure (DevicePolicyError always propagates: the
        configured device is final)."""
        language = self._language.locked
        prompt = None
        if self._config.context_sentences > 0 and self._context and self._context_language == language:
            prompt = " ".join(self._context)
        started = time.monotonic()
        try:
            result = self._backend.decode_segment(
                self._window.samples,
                prompt=prompt,
                language=language,
                drop_trailing_period=drop_trailing_period,
                peek=peek,
            )
        except DevicePolicyError:
            raise
        except Exception:
            logger.exception("segment %s failed; skipping it", "peek" if peek else "decode")
            return None
        if not peek:
            self._note_decode_duration(time.monotonic() - started, self._window.duration_seconds)
        return result

    def _close_segment(self, *, rollover: bool, at_pause: bool) -> None:
        held = self._window.duration_seconds
        decodable = held >= self._min_decode_seconds and (self._mode != "hybrid" or self._speech_seen)
        emitted = False
        if decodable:
            language = self._language.for_finalize(self._window.samples, held)
            if language is None:
                logger.warning("language detection failed at segment end; dropping %.1fs of audio", held)
            else:
                # A segment cut mid-speech is not a sentence end: drop the
                # "." whisper puts there; one that ended at a pause keeps it.
                result = self._decode(peek=False, drop_trailing_period=not at_pause)
                if result is not None:
                    emitted = self._emit_final(result.text, language)
        if self._partial_shown and not emitted:
            self._on_event(CaptionEvent(CaptionEventType.PARTIAL, ""))  # no stale partial left on screen
        # hybrid/fixed cuts are not speech-aligned: each segment decides its
        # language afresh (only a vad-mode cut is "the same speech").
        self._language.finish(rollover and self._mode == "vad")
        self._in_segment = False
        self._window.clear()
        self._speech_seen = False
        self._partial_shown = False
        self._held_at_last_peek = 0.0

    def _emit_final(self, raw_text: str, language: str) -> bool:
        words = self._boundary.apply(self._text.words(raw_text))
        text = " ".join(words)
        if not _has_text(text):
            return False
        trailing_silence = self._silence_ms_accum * self._sample_rate / 1000.0
        end_sample = max(self._fed_samples - trailing_silence, self._segment_start_sample)
        start = self._sample_clock(self._segment_start_sample)
        end = self._sample_clock(end_sample)
        self._remember_context(text, language)
        self._on_event(CaptionEvent(CaptionEventType.FINAL, text, start=start, end=end))
        return True

    def _remember_context(self, text: str, language: str) -> None:
        if self._config.context_sentences <= 0:
            return
        if self._context_language != language:
            self._context.clear()  # no Polish context for English speech
            self._context_language = language
        self._context.append(text)

    def _reset_context(self) -> None:
        """A silence WIPE cleared the screen: the context goes with it."""
        self._context.clear()
        self._context_language = None

    def _sample_clock(self, sample: float) -> float:
        """The decoder clock at which audio sample `sample` was fed."""
        return self._feed_clock - (self._fed_samples - sample) / self._sample_rate

    def _note_decode_duration(self, elapsed: float, segment_seconds: float) -> None:
        """A FINAL decode slower than the segment it decodes means the worker
        cannot keep up; say so (rate-limited) rather than lag silently."""
        if elapsed <= max(segment_seconds, 1.0):
            return
        now = self._clock()
        if now - self._last_slow_warning < _SLOW_WARN_INTERVAL_SECONDS:
            return
        self._last_slow_warning = now
        logger.warning(
            "segment decode took %.2fs for %.1fs of audio; captions will lag. Consider a smaller "
            "asr.model, a smaller asr.max_segment_ms, or turning partial captions off.",
            elapsed,
            segment_seconds,
        )

    def set_text_settings(self, settings: TextSettings) -> None:
        """Hot-reload vocabulary snapping/blocklist (the worker's control
        queue); the prompt vocabulary is the backend's."""
        self._text.set_text_settings(settings)


def _has_text(text: str) -> bool:
    """Empty or punctuation-only text ("", ".") produces no caption."""
    return any(ch.isalnum() for ch in text)
