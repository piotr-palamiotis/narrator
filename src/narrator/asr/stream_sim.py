"""Live-faithful, faster-than-real-time replay of the streaming decoder.

Autoconfigure needs to know what the LIVE captioner would have shown for
hours of audio, for many configurations -- and the live path is defined by
timing: the ASR worker drains every queued 30 ms frame, then ticks once, so
a decode step that takes D seconds lets D seconds of audio pile up before
the next tick (worker.py, StreamingDecoder.tick). Replaying audio in real
time reproduces that, at one hour per hour of audio per configuration; the
old quality probe did exactly that and timed out on every candidate.

This module replays the same sequence on a VIRTUAL clock:
  * the clock is audio time -- frame k is fully captured at (k+1) * 30 ms;
  * every backend call (decode or language detection) is charged a cost by
    a `DecodeCostModel` and occupies the worker: it starts at
    max(now, busy_until) and keeps the worker busy until start + cost;
  * frames are fed as they arrive -- feeding during a busy period is the
    same as the live drain that follows it (same order, same decoder
    state) -- and `tick()` runs only once the worker is free again;
  * FINAL captions are stamped with the completion time of the call that
    produced them: when the audience would have seen them.

With a fitted cost model the replay is deterministic -- identical inputs
give identical captions, so two configurations are compared without GPU
timing jitter deciding the outcome. `measured=True` charges real wall time
instead (the fidelity check). Inference itself still runs for real; only
the waiting is skipped, so a run costs the decode time alone (~0.1-0.25x
real time for the large models on the reference RX 6700 XT).
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.streaming_decoder import StreamingDecoder
from narrator.asr.vad import SileroVad
from narrator.config.schema import AsrConfig

SAMPLE_RATE = 16000
FRAME_MS = 30


@dataclass(frozen=True)
class DecodeCostModel:
    """Seconds one backend call keeps the ASR worker busy.

    decode = base_s + per_audio_s * window_seconds + per_word_s * words:
    whisper's encoder cost scales with audio_ctx (the window), its decoder
    cost with the tokens it emits -- which is what makes a speech-dense
    window markedly slower than a sparse one of the same length on the
    large models (32 decoder layers). The emitted word count is known once
    the call returns, so charging it keeps the replay deterministic.
    Language detection is the constant `detect_s` (it always runs on a
    fixed-size slice, backend.py).
    """

    base_s: float
    per_audio_s: float
    detect_s: float = 0.0
    per_word_s: float = 0.0

    def decode_seconds(self, window_seconds: float, words: int = 0) -> float:
        return self.base_s + self.per_audio_s * window_seconds + self.per_word_s * words

    def typical_decode_seconds(self) -> float:
        """One decode of 10s of continuous speech (~25 words): the figure
        models are compared and reported by."""
        return self.decode_seconds(10.0, 25)

    @classmethod
    def fit_calls(cls, calls: list[tuple[float, int, float]], detect_s: float) -> "DecodeCostModel":
        """Least-squares fit of (window seconds, words emitted, seconds)
        triples measured on real streaming decodes, coefficients kept
        non-negative (a term that fits negative is dropped and the rest
        refit). Deliberately no outlier rejection: the slowest calls are
        repetition loops running to the token cap (62 words out of a 1.6s
        window, 3.4s), which happen live too -- dropping them made the
        model underestimate total decode time by 5-8%, i.e. the worker load
        the search constrains. One-time GPU setup is excluded at the source
        (the probe warms up before timed decodes)."""
        if len(calls) < 3:
            raise ValueError("need at least 3 measured decode calls to fit a cost model")
        x = np.array([[1.0, w, n] for w, n, _ in calls], dtype=np.float64)
        y = np.array([t for _, _, t in calls], dtype=np.float64)
        coef = _nonnegative_lstsq(x, y)
        return cls(
            base_s=round(float(coef[0]), 3),
            per_audio_s=round(float(coef[1]), 4),
            detect_s=round(max(detect_s, 0.0), 3),
            per_word_s=round(float(coef[2]), 4),
        )

    @classmethod
    def fit(
        cls, decode_seconds_by_window: dict[str, float], detect_s: float = 0.0
    ) -> "DecodeCostModel":
        """Least-squares line through bench timings ({"10.0s": 0.94, ...}).

        Clamped to non-negative coefficients (a negative intercept from
        noisy short-window timings is refit through the origin) and rounded
        to the millisecond, so near-identical benches give the same model.
        """
        points = [
            (float(label.rstrip("s")), float(seconds))
            for label, seconds in decode_seconds_by_window.items()
        ]
        if not points:
            raise ValueError("no bench timings to fit a decode cost model")
        if len(points) == 1:
            window, seconds = points[0]
            base, slope = 0.0, seconds / window
        else:
            mean_x = sum(x for x, _ in points) / len(points)
            mean_y = sum(y for _, y in points) / len(points)
            var = sum((x - mean_x) ** 2 for x, _ in points)
            slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / var if var else 0.0
            base = mean_y - slope * mean_x
            if base < 0.0:
                slope = sum(x * y for x, y in points) / sum(x * x for x, _ in points)
                base = 0.0
        return cls(
            base_s=round(max(base, 0.0), 3),
            per_audio_s=round(max(slope, 0.0), 4),
            detect_s=round(max(detect_s, 0.0), 3),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "base_s": self.base_s,
            "per_audio_s": self.per_audio_s,
            "detect_s": self.detect_s,
            "per_word_s": self.per_word_s,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecodeCostModel":
        return cls(
            base_s=float(data["base_s"]),
            per_audio_s=float(data["per_audio_s"]),
            detect_s=float(data.get("detect_s", 0.0)),
            per_word_s=float(data.get("per_word_s", 0.0)),
        )


def _nonnegative_lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with every coefficient >= 0 (drop the most negative
    term and refit until none is negative)."""
    active = list(range(x.shape[1]))
    coef = np.zeros(x.shape[1])
    while active:
        solution, *_ = np.linalg.lstsq(x[:, active], y, rcond=None)
        if (solution >= 0).all():
            coef[active] = solution
            break
        active.pop(int(np.argmin(solution)))
    return coef


class VirtualClock:
    """The decoder's clock: audio time of the last captured frame."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _TimedBackend:
    """Backend proxy that charges every call to the virtual worker.

    A call starts when the worker is free (max(now, busy_until)) and holds
    it for the modeled -- or, with `measured`, the real -- duration. The
    completion time of the latest call is what a caption it produced gets
    stamped with.
    """

    def __init__(self, backend, clock: VirtualClock, cost: DecodeCostModel | None, measured: bool) -> None:
        if cost is None and not measured:
            raise ValueError("a cost model is required unless measured=True")
        self._backend = backend
        self._clock = clock
        self._cost = cost
        self._measured = measured
        self.busy_until = 0.0
        self.completed_at = 0.0
        self.busy_seconds = 0.0
        self.decode_calls = 0
        self.detect_calls = 0
        self.languages: Counter[str] = Counter()
        # (window seconds, words emitted, wall seconds) per decode, measured
        # mode only: the data a cost model is calibrated from
        self.measured_calls: list[tuple[float, int, float]] = []
        self.measured_detect: list[float] = []

    @property
    def resolved_device_description(self) -> str:
        return self._backend.resolved_device_description

    def _charge(self, modeled_seconds: float, wall_seconds: float) -> None:
        cost = wall_seconds if self._measured else modeled_seconds
        start = max(self._clock.now, self.busy_until)
        self.busy_until = start + cost
        self.completed_at = self.busy_until
        self.busy_seconds += cost

    def transcribe(self, audio: np.ndarray, *args, **kwargs):
        started = time.monotonic()
        result = self._backend.transcribe(audio, *args, **kwargs)
        wall = time.monotonic() - started
        window = len(audio) / SAMPLE_RATE
        words = sum(len(text.split()) for _, _, text in result)
        modeled = self._cost.decode_seconds(window, words) if self._cost else 0.0
        self._charge(modeled, wall)
        if self._measured:
            self.measured_calls.append((round(window, 3), words, round(wall, 4)))
        self.decode_calls += 1
        language = kwargs.get("language")
        if language:
            self.languages[str(language)] += 1
        return result

    def detect_language(self, audio: np.ndarray):
        started = time.monotonic()
        result = self._backend.detect_language(audio)
        wall = time.monotonic() - started
        modeled = self._cost.detect_s if self._cost else 0.0
        self._charge(modeled, wall)
        if self._measured:
            self.measured_detect.append(round(wall, 4))
        self.detect_calls += 1
        return result


@dataclass
class SimResult:
    """What the live captioner would have emitted for one stretch of audio."""

    finals: list[tuple[float, str]]  # (emit time in audio seconds, caption text)
    audio_seconds: float
    decode_calls: int
    detect_calls: int
    busy_seconds: float  # total time the worker spent in backend calls
    # decode calls per language the decoder locked (utterance language mix)
    languages: dict[str, int] = field(default_factory=dict)
    rollovers: int = 0  # utterances force-finalized at the length cap
    # measured mode only: per-call timings (see _TimedBackend)
    calls: list[tuple[float, int, float]] = field(default_factory=list)
    detects: list[float] = field(default_factory=list)

    @property
    def busy_fraction(self) -> float:
        """Worker occupancy; at 1.0 the hardware only just keeps up and
        every extra second of decode becomes caption delay."""
        return self.busy_seconds / self.audio_seconds if self.audio_seconds > 0 else 0.0

    @property
    def text(self) -> str:
        return " ".join(text for _, text in self.finals)

    def to_dict(self) -> dict:
        return {
            "finals": [[round(t, 3), text] for t, text in self.finals],
            "audio_seconds": round(self.audio_seconds, 3),
            "decode_calls": self.decode_calls,
            "detect_calls": self.detect_calls,
            "busy_seconds": round(self.busy_seconds, 3),
            "languages": dict(self.languages),
            "rollovers": self.rollovers,
            **({"calls": [list(c) for c in self.calls], "detects": self.detects} if self.calls else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimResult":
        return cls(
            finals=[(float(t), str(text)) for t, text in data["finals"]],
            audio_seconds=float(data["audio_seconds"]),
            decode_calls=int(data["decode_calls"]),
            detect_calls=int(data["detect_calls"]),
            busy_seconds=float(data["busy_seconds"]),
            languages={str(k): int(v) for k, v in data.get("languages", {}).items()},
            rollovers=int(data.get("rollovers", 0)),
            calls=[(float(w), int(n), float(t)) for w, n, t in data.get("calls", [])],
            detects=[float(t) for t in data.get("detects", [])],
        )


def simulate_stream(
    backend,
    config: AsrConfig,
    audio: np.ndarray,
    cost: DecodeCostModel | None = None,
    *,
    measured: bool = False,
    vad=None,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = FRAME_MS,
) -> SimResult:
    """Replay `audio` (mono float32) through a fresh StreamingDecoder as the
    live worker would have processed it, on the virtual clock.

    After the audio, silence is fed long enough to cross end_silence_ms so
    the trailing utterance finalizes exactly as when a speaker stops.
    """
    clock = VirtualClock()
    timed = _TimedBackend(backend, clock, cost, measured)
    finals: list[tuple[float, str]] = []

    def on_event(event: CaptionEvent) -> None:
        if event.type is CaptionEventType.FINAL and event.text.strip():
            finals.append((timed.completed_at, event.text.strip()))

    decoder = StreamingDecoder(
        timed,
        vad if vad is not None else SileroVad(threshold=config.vad_threshold),
        config,
        on_event,
        sample_rate=sample_rate,
        clock=clock,
    )
    frame_samples = sample_rate * frame_ms // 1000
    frame_seconds = frame_samples / sample_rate

    def step(frame: np.ndarray) -> None:
        clock.now += len(frame) / sample_rate
        decoder.feed(frame, run_decode_step=False)
        if clock.now >= timed.busy_until:
            decoder.tick()

    for offset in range(0, len(audio), frame_samples):
        step(audio[offset : offset + frame_samples])
    silence = np.zeros(frame_samples, dtype=np.float32)
    for _ in range(math.ceil(config.end_silence_ms / frame_ms) + 4):
        step(silence)
    # Let a decode still in flight at the end complete before reading out.
    clock.now = max(clock.now, timed.busy_until) + frame_seconds
    decoder.tick()

    return SimResult(
        finals=finals,
        audio_seconds=len(audio) / sample_rate,
        decode_calls=timed.decode_calls,
        detect_calls=timed.detect_calls,
        busy_seconds=timed.busy_seconds,
        languages=dict(timed.languages),
        rollovers=int(getattr(decoder, "rollover_count", 0)),
        calls=timed.measured_calls,
        detects=timed.measured_detect,
    )
