"""Silero VAD wrapper, driven directly through ONNX Runtime.

Deliberately does NOT depend on the `silero-vad` PyPI package: that package
unconditionally imports torch+torchaudio even when asked to use its ONNX
model (see silero_vad/model.py upstream), which drags a multi-hundred-MB
deep learning framework into a project chosen specifically to stay on
ONNX Runtime for a lightweight, easy-to-package VAD. The model file itself
(assets/vad/silero_vad.onnx, MIT-licensed, see assets/vad/LICENSE-silero-vad.txt)
is redistributed directly and driven with a small hand-written wrapper instead.

Model contract (Silero VAD v5 ONNX graph, confirmed via `get_inputs()` /
`get_outputs()` on the bundled graph):
  inputs:  input (1, N) float32 audio chunk, state (2, 1, 128) float32 recurrent
           state, sr () int64 sample rate
  outputs: output (1, 1) float32 speech probability, stateN (2, 1, 128) float32
           updated recurrent state
Chunk size must be exactly 512 samples at 16kHz (32ms) -- this wrapper buffers
arbitrary-length input frames internally and only runs inference once a full
512-sample window has accumulated, so callers (audio/capture.py's frame_ms)
don't need to match this exactly.

The v5 graph's causal convolutions also require 64 samples of left context
from the *previous* window prepended to each 512-sample window fed to
`input` (verified against the official `silero-vad` package's own
`OnnxWrapper.__call__`, which does the same) -- without it the graph is
structurally starved of temporal continuity between windows and its output
saturates near a fixed low-probability constant regardless of input,
including on real, clearly-audible speech. This was confirmed empirically:
feeding bare 512-sample windows with no context produced <0.04 probability
across an entire 60s reference speech file; prepending the trailing 64
samples of the prior window raised that to a 0.78 mean with 77% of windows
correctly crossing the default 0.5 threshold on the same file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_MODEL_PATH = Path(__file__).resolve().parents[3] / "assets" / "vad" / "silero_vad.onnx"
_WINDOW_SAMPLES = 512  # required by the Silero v5 graph at 16kHz
_CONTEXT_SAMPLES = 64  # required left-context for the same graph's causal convs
_SAMPLE_RATE = 16000

# The v5 LSTM state drifts: after minutes of continuously fed audio (music,
# crowd noise, anything that is not clean speech) the probabilities collapse
# toward zero and STAY there even when clear speech resumes -- measured on a
# real 88-min recording: after ~4 min of concert, speech that a fresh-state
# VAD scores at 0.58 mean (59% of windows over threshold) scored 0.004 mean
# (0% over threshold) on the continuous stream, i.e. captions would silently
# stop working. The state is only meaningful over utterance-scale horizons.
# Antidote: reset the state after this many seconds of continuously
# below-threshold windows (music counts as below-threshold, so the reset
# keeps firing through songs). Measured on the same recording: 10s -> 56%
# of speech windows over threshold (fresh-state reference: 59%); 30s -> 48%;
# no reset -> 0%.
_RESET_AFTER_QUIET_SECONDS = 10.0
_WINDOW_SECONDS = _WINDOW_SAMPLES / _SAMPLE_RATE
_QUIET_WINDOWS_FOR_RESET = round(_RESET_AFTER_QUIET_SECONDS / _WINDOW_SECONDS)


class SileroVad:
    def __init__(
        self,
        threshold: float = 0.5,
        model_path: Path = _MODEL_PATH,
        reset_after_quiet_seconds: float = _RESET_AFTER_QUIET_SECONDS,
    ) -> None:
        self._threshold = threshold
        # 0 or negative disables the auto-reset (useful in tests; not
        # recommended for live use -- see the drift measurement above).
        self._quiet_windows_for_reset = (
            round(reset_after_quiet_seconds / _WINDOW_SECONDS)
            if reset_after_quiet_seconds > 0
            else 0
        )
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(_SAMPLE_RATE, dtype=np.int64)
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        self._quiet_windows = 0

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        self._quiet_windows = 0

    def process(self, frame: np.ndarray) -> list[float]:
        """Feed an arbitrary-length mono float32 frame at 16kHz. Returns the
        list of speech probabilities for each full 512-sample window that
        became available (usually 0 or 1 entries per call at typical frame
        sizes, more if `frame` is long).
        """
        self._pending = np.concatenate([self._pending, frame.astype(np.float32)])
        probabilities: list[float] = []
        while len(self._pending) >= _WINDOW_SAMPLES:
            window, self._pending = (
                self._pending[:_WINDOW_SAMPLES],
                self._pending[_WINDOW_SAMPLES:],
            )
            model_input = np.concatenate([self._context, window])
            outputs = self._session.run(
                None,
                {
                    "input": model_input.reshape(1, -1),
                    "state": self._state,
                    "sr": self._sr,
                },
            )
            prob, self._state = outputs[0], outputs[1]
            probabilities.append(float(prob[0][0]))
            self._context = window[-_CONTEXT_SAMPLES:]
            # State-drift antidote: see _RESET_AFTER_QUIET_SECONDS above.
            # Fires only after long quiet, which the streaming decoder treats
            # as no active utterance anyway (end_silence_ms is ~0.65s), so a
            # reset can never land mid-utterance.
            if float(prob[0][0]) >= self._threshold:
                self._quiet_windows = 0
            elif self._quiet_windows_for_reset:
                self._quiet_windows += 1
                if self._quiet_windows >= self._quiet_windows_for_reset:
                    self._state = np.zeros((2, 1, 128), dtype=np.float32)
                    self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
                    self._quiet_windows = 0
        return probabilities

    def is_speech(self, probabilities: list[float]) -> bool:
        return any(p >= self._threshold for p in probabilities)
