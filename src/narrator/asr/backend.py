"""whisper.cpp binding + GPU backend detection/fallback.

Uses `pywhispercpp` (a pybind11 wrapper around whisper.cpp's C API) as the
prototyping-tier backend called out in the plan: `Model(name, context_params=
{"use_gpu": ...})` loads/keeps the model resident, and `Model.transcribe()`
runs inference on an in-memory numpy array -- exactly the long-lived,
repeatedly-called shape this streaming pipeline needs (see
streaming_decoder.py), unlike a CLI-subprocess-per-chunk or HTTP server
approach which would reload the model every call.

IMPORTANT CAVEAT (confirmed empirically during development, not theoretical):
prebuilt `pywhispercpp` wheels from PyPI are plain CPU builds with no Vulkan
support compiled in, so `use_gpu=True` here will not actually engage a GPU
unless pywhispercpp is rebuilt from source with its `GGML_VULKAN=1` env-var
build flag (confirmed against the pywhispercpp project's own documented
build options) -- see packaging/whisper_cpp/README.md for the exact
commands. That build step must happen on a machine with the Vulkan SDK
installed; it is a packaging-time concern, not something this module can
detect or fix at runtime. This module still implements the full
auto/vulkan/cpu selection and fallback contract against whatever
whisper.cpp build is actually loaded, so behavior is identical once a
GPU-enabled build is dropped in.
"""
from __future__ import annotations

import logging
import shutil
import subprocess

import numpy as np

from narrator.config.schema import AsrConfig, AsrDevice

logger = logging.getLogger(__name__)


def _vulkan_device_names() -> list[str]:
    """Best-effort enumeration of Vulkan-capable GPUs via the `vulkaninfo` CLI
    (part of the Vulkan SDK / most GPU driver packages). Returns an empty
    list if the tool isn't present or reports no devices -- callers treat
    that as "can't confirm a GPU, may still be worth trying" rather than a
    hard "no GPU" signal, since whisper.cpp's own Vulkan init is the real
    source of truth.
    """
    if shutil.which("vulkaninfo") is None:
        return []
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=5
        )
    except (subprocess.SubprocessError, OSError):
        return []
    names = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("deviceName"):
            names.append(line.split("=", 1)[1].strip())
    return names


class WhisperBackend:
    def __init__(self, config: AsrConfig) -> None:
        self._config = config
        self._model = None
        self._resolved_device_description = "unresolved"
        self._init_model()

    def _init_model(self) -> None:
        want_gpu = self._config.device in (AsrDevice.AUTO, AsrDevice.VULKAN)
        if want_gpu:
            gpu_names = _vulkan_device_names()
            if self._config.device == AsrDevice.VULKAN and not gpu_names:
                logger.warning(
                    "asr.device=vulkan but no Vulkan device was detected via vulkaninfo; "
                    "attempting GPU init anyway since whisper.cpp's own detection is authoritative"
                )
            try:
                self._model = self._load_model(use_gpu=True)
                gpu_label = gpu_names[0] if gpu_names else "GPU"
                self._resolved_device_description = f"Vulkan ({gpu_label})"
                return
            except Exception as exc:  # native init can fail in many ways; any failure -> CPU fallback
                logger.warning("GPU backend init failed (%s); falling back to CPU", exc)

        self._model = self._load_model(use_gpu=False)
        self._resolved_device_description = "CPU"

    def _load_model(self, use_gpu: bool):
        from pywhispercpp.model import Model  # imported lazily: heavy native extension

        # flash_attn is forced off here: whisper.cpp's Vulkan flash-attention
        # kernel is known to crash with an illegal instruction (SIGILL, i.e.
        # multiprocessing exitcode -4) on a number of GPU/driver combinations,
        # which kills the ASR worker process outright -- no Python exception
        # is raised for us to catch and fall back from, so the only fix is to
        # never select that code path in the first place.
        return Model(
            self._config.model,
            context_params={"use_gpu": use_gpu, "flash_attn": False},
            print_progress=False,
            print_realtime=False,
            no_context=True,
        )

    @property
    def resolved_device_description(self) -> str:
        return self._resolved_device_description

    def transcribe(self, audio: np.ndarray) -> list[tuple[float, float, str]]:
        """Run one decode pass over `audio` (mono float32 @16kHz). Returns a
        list of (start_seconds, end_seconds, text) segments. If the current
        backend is GPU-based and inference raises (a real, observed failure
        mode with immature Vulkan driver paths), reinitializes on CPU and
        retries once rather than crashing the whole ASR worker process.
        """
        language = None if self._config.language == "auto" else self._config.language
        kwargs = {"single_segment": False}
        if language is not None:
            kwargs["language"] = language
        else:
            kwargs["detect_language"] = True

        try:
            segments = self._model.transcribe(audio, **kwargs)
        except Exception as exc:
            if self._resolved_device_description.startswith("Vulkan"):
                logger.warning("GPU inference failed (%s); reinitializing on CPU", exc)
                self._model = self._load_model(use_gpu=False)
                self._resolved_device_description = "CPU (fallback after GPU failure)"
                segments = self._model.transcribe(audio, **kwargs)
            else:
                raise
        return [(seg.t0 / 100.0, seg.t1 / 100.0, seg.text.strip()) for seg in segments]
