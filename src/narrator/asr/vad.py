"""Silero VAD wrapper, driven directly through ONNX Runtime.

Deliberately does NOT depend on the `silero-vad` PyPI package: that package
unconditionally imports torch+torchaudio even when asked to use its ONNX
model (see silero_vad/model.py upstream), which drags a multi-hundred-MB
deep learning framework into a project chosen specifically to stay on
ONNX Runtime for a lightweight, easy-to-package VAD. The model file itself
(assets/vad/silero_vad.onnx, MIT-licensed, see assets/vad/LICENSE-silero-vad.txt)
is redistributed directly and driven with a small hand-written wrapper instead.

Model contract (Silero VAD v4/v5 ONNX graph, confirmed via `get_inputs()` /
`get_outputs()` on the bundled graph):
  inputs:  input (1, N) float32 audio chunk, state (2, 1, 128) float32 recurrent
           state, sr () int64 sample rate
  outputs: output (1, 1) float32 speech probability, stateN (2, 1, 128) float32
           updated recurrent state
Chunk size must be exactly 512 samples at 16kHz (32ms) -- this wrapper buffers
arbitrary-length input frames internally and only runs inference once a full
512-sample window has accumulated, so callers (audio/capture.py's frame_ms)
don't need to match this exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_MODEL_PATH = Path(__file__).resolve().parents[3] / "assets" / "vad" / "silero_vad.onnx"
_WINDOW_SAMPLES = 512  # required by the Silero v4/v5 graph at 16kHz
_SAMPLE_RATE = 16000


class SileroVad:
    def __init__(self, threshold: float = 0.5, model_path: Path = _MODEL_PATH) -> None:
        self._threshold = threshold
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(_SAMPLE_RATE, dtype=np.int64)
        self._pending = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

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
            outputs = self._session.run(
                None,
                {
                    "input": window.reshape(1, -1),
                    "state": self._state,
                    "sr": self._sr,
                },
            )
            prob, self._state = outputs[0], outputs[1]
            probabilities.append(float(prob[0][0]))
        return probabilities

    def is_speech(self, probabilities: list[float]) -> bool:
        return any(p >= self._threshold for p in probabilities)
