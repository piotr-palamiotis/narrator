"""whisper.cpp binding + GPU backend detection/selection/fallback.

Uses `pywhispercpp` (a pybind11 wrapper around whisper.cpp's C API) as the
backend: `Model(name, context_params={"use_gpu": ...})` loads/keeps the model
resident, and `Model.transcribe()` runs inference on an in-memory numpy array
-- exactly the long-lived, repeatedly-called shape this streaming pipeline
needs (see streaming_decoder.py), unlike a CLI-subprocess-per-chunk or HTTP
server approach which would reload the model every call.

GPU support, verified empirically on this project's target hardware
(Radeon RX 6700 XT / RADV, Mesa 25.2):

  * The prebuilt `pywhispercpp` wheel on PyPI (1.5.1) **already ships a
    Vulkan backend** -- it installs `libggml-vulkan.so` alongside
    `libggml-cpu.so`, and `use_gpu=True` really does run inference on the
    discrete GPU with no rebuild. An earlier revision of this file claimed a
    from-source rebuild with `GGML_VULKAN=1` was required; that is not true
    for current wheels and the claim has been removed rather than left to
    mislead deployment.

  * Those same wheels are built for **baseline x86-64**: disassembly of
    `libggml-cpu.so`, `libggml-base.so`, `libwhisper.so` and
    `libggml-vulkan.so` finds zero AVX/AVX2/FMA/VEX-encoded instructions.
    They therefore load and run on pre-AVX CPUs (verified on a Pentium
    G4560, which has only up to SSE4.2), so an explicitly configured CPU
    device (asr.device: cpu) is safe everywhere and no CPU-specific build is
    needed for deployment.

Device detection is done by calling into `libvulkan.so.1` directly with
ctypes rather than shelling out to the `vulkaninfo` CLI. `vulkaninfo` ships
in the *vulkan-tools* package, which is a developer/diagnostic package and is
routinely absent on machines that have a perfectly working Vulkan runtime
(the ICD + `libvulkan.so.1` come from the driver package instead). The old
CLI-based probe therefore reported "no Vulkan devices" on a machine with a
working RX 6700 XT, producing a spurious warning and an unhelpful "Vulkan
(GPU)" status label with no adapter name.
"""
from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

import numpy as np

from narrator.config.schema import AsrConfig, AsrDevice

logger = logging.getLogger(__name__)

# Vulkan constants (vulkan_core.h). Only the handful we need.
_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
_VK_SUCCESS = 0
_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256
_VK_PHYSICAL_DEVICE_TYPE_CPU = 4
_VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2

_DEVICE_TYPE_NAMES = {
    0: "other",
    1: "integrated",
    2: "discrete",
    3: "virtual",
    4: "cpu",
}

# VkPhysicalDeviceProperties is ~824 bytes and ends with two large nested
# structs (limits, sparseProperties) we never read. Rather than model those
# field-for-field, hand Vulkan an over-sized buffer and slice the leading
# scalar fields out by offset: apiVersion/driverVersion/vendorID/deviceID are
# 4 bytes each (16 total), then deviceType (4), then deviceName.
_PROPERTIES_BUFFER_SIZE = 2048
_DEVICE_TYPE_OFFSET = 16
_DEVICE_NAME_OFFSET = 20


@dataclass(frozen=True)
class VulkanDevice:
    """One GPU as whisper.cpp/ggml will index it (see `_vulkan_devices`)."""

    index: int
    name: str
    device_type: int

    @property
    def is_discrete(self) -> bool:
        return self.device_type == _VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU

    def __str__(self) -> str:
        kind = _DEVICE_TYPE_NAMES.get(self.device_type, str(self.device_type))
        return f"{self.name} [{kind}]"


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def _vulkan_devices() -> list[VulkanDevice]:
    """Enumerate Vulkan GPUs through `libvulkan.so.1` via ctypes.

    Returns devices in ggml's own indexing order, which is what
    `context_params["gpu_device"]` selects against. ggml's Vulkan backend
    skips `VK_PHYSICAL_DEVICE_TYPE_CPU` adapters (software rasterizers such
    as llvmpipe) when it builds its device list, so this filters them out
    too -- otherwise a machine with llvmpipe installed would produce indices
    here that don't line up with the ones whisper.cpp actually uses.

    Any failure (no loader, no ICD, driver error) returns an empty list:
    callers treat that as "couldn't confirm a GPU", not a hard "no GPU",
    since whisper.cpp's own init remains the source of truth.
    """
    try:
        lib = ctypes.CDLL("libvulkan.so.1")
    except OSError:
        try:
            lib = ctypes.CDLL("vulkan-1.dll")  # Windows loader name
        except OSError:
            logger.debug("Vulkan loader not found; cannot enumerate GPUs")
            return []

    instance = ctypes.c_void_p()
    create_info = _VkInstanceCreateInfo(sType=_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
    try:
        if lib.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance)) != _VK_SUCCESS:
            logger.debug("vkCreateInstance failed; treating as no Vulkan devices")
            return []
    except (AttributeError, OSError):
        logger.debug("Vulkan loader present but vkCreateInstance unusable")
        return []

    try:
        count = ctypes.c_uint32(0)
        if lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None) != _VK_SUCCESS:
            return []
        if count.value == 0:
            return []
        handles = (ctypes.c_void_p * count.value)()
        if lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles) != _VK_SUCCESS:
            return []

        devices: list[VulkanDevice] = []
        for handle in handles:
            buffer = ctypes.create_string_buffer(_PROPERTIES_BUFFER_SIZE)
            lib.vkGetPhysicalDeviceProperties(handle, buffer)
            raw = buffer.raw
            device_type = int.from_bytes(
                raw[_DEVICE_TYPE_OFFSET : _DEVICE_TYPE_OFFSET + 4], "little"
            )
            name = (
                raw[_DEVICE_NAME_OFFSET : _DEVICE_NAME_OFFSET + _VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]
                .split(b"\0")[0]
                .decode("utf-8", "replace")
            )
            if device_type == _VK_PHYSICAL_DEVICE_TYPE_CPU:
                continue  # ggml skips software rasterizers; keep indices aligned
            devices.append(VulkanDevice(index=len(devices), name=name, device_type=device_type))
        return devices
    finally:
        try:
            lib.vkDestroyInstance(instance, None)
        except (AttributeError, OSError):
            pass


def _vulkan_device_names() -> list[str]:
    """Backwards-compatible helper: just the adapter names, in ggml order."""
    return [device.name for device in _vulkan_devices()]


def select_gpu_index(devices: list[VulkanDevice], configured: int | str) -> int:
    """Resolve `asr.gpu_device` against the enumerated devices.

    "auto" prefers the first discrete GPU, because the common failure mode is
    a workstation whose integrated adapter enumerates first: whisper.cpp
    would then default to `gpu_device=0` and run the model on the iGPU, which
    is dramatically slower but fails silently (it still "uses the GPU").
    """
    if configured != "auto":
        index = int(configured)
        if devices and not any(device.index == index for device in devices):
            logger.warning(
                "asr.gpu_device=%d is not among the %d enumerated Vulkan device(s); "
                "passing it to whisper.cpp anyway",
                index,
                len(devices),
            )
        return index

    for device in devices:
        if device.is_discrete:
            return device.index
    return 0


# Roughly 200 tokens is far more than a max_context_seconds-sized window of
# real speech produces; it only bites on runaway repetition loops.
_MAX_TOKENS_PER_STEP = 200

# asr.language="auto": pywhispercpp 1.5.1's `detect_language=True` transcribe
# param is broken -- it returns zero segments on any input (verified on a
# fresh model with clean audio). Detection is done through the dedicated
# `auto_detect_language()` API instead, and whatever code it reports is used
# verbatim: the decoder transcribes each utterance in whatever language it
# hears, with no per-language allow-list. The StreamingDecoder calls this once
# per utterance and locks the result (per-step detection measurably flaps
# between acoustically similar languages, mixing languages mid-caption).
# Detection on raw noise/breath can return an arbitrary language at low
# confidence; a garbage transcription of noise is preferable to silently
# dropping or substituting speech the operator can hear.
# Detection only needs a few seconds of audio; capping the slice keeps the
# detection cost roughly constant on long windows. 2.0s sits comfortably above
# the measured reliability floor (large-v3: >=1.5s detects pl/en 7/7 at >=0.97
# confidence; 0.4s slices misdetect 1-in-7) while costing ~180ms per call on
# large-v3/GPU versus ~450ms at 5s -- detection runs once per utterance
# finalize and once per recheck, so its cost lands directly on finalize
# latency.
_LANGUAGE_DETECT_MAX_SECONDS = 2.0
# Lock readiness: below this detection confidence the StreamingDecoder keeps
# collecting audio instead of locking (re-detects on later steps / finalize).
# Time alone is not enough -- marginal clips can still flip at the minimum
# length; a low-probability lock is exactly how Cyrillic Polish captions
# happen. Returned probability is whisper's softmax over languages.
_LANGUAGE_PROB_THRESHOLD = 0.5

_FULL_AUDIO_CTX = 1500  # whisper's native 30s encoder context
_MIN_AUDIO_CTX = 256
_AUDIO_CTX_MARGIN = 64


def audio_ctx_for_seconds(window_seconds: float) -> int:
    """Encoder context length (in mel frames) sufficient for `window_seconds`.

    whisper's encoder is fixed at a 30s / 1500-frame context and pads every
    input up to it, so a 2s decode window costs the same encode as a 30s one.
    whisper.cpp exposes `audio_ctx` to shorten that context, which is the
    single biggest win available to a streaming decoder: measured on an
    RX 6700 XT, large-v3 on a 2s window drops from 0.88s to 0.31s.

    A margin is added over the exact requirement and the result is clamped to
    a floor, since cutting the context too close to the audio length degrades
    accuracy at the tail of the window.
    """
    exact = int(1500.0 * window_seconds / 30.0)
    return max(_MIN_AUDIO_CTX, min(_FULL_AUDIO_CTX, exact + _AUDIO_CTX_MARGIN))


class DevicePolicyError(RuntimeError):
    """The configured ASR device could not be used or stopped working.

    Raised instead of silently falling back to CPU. The configured device is
    final by policy: a transient failure (the previous worker generation not
    yet having released the GPU, a driver hiccup) must surface as a worker
    crash the supervisor can see -- not as an invisible downgrade that leaves
    the operator's "GPU" config running 10x slower on CPU with a GUI
    indicator claiming all is well.
    """


class WhisperBackend:
    def __init__(self, config: AsrConfig) -> None:
        self._config = config
        self._model = None
        self._resolved_device_description = "unresolved"
        self._gpu_index = 0
        self._init_model()

    def _init_model(self) -> None:
        want_gpu = self._config.device in (AsrDevice.AUTO, AsrDevice.VULKAN)
        if want_gpu:
            devices = _vulkan_devices()
            if devices:
                logger.info(
                    "Vulkan devices: %s",
                    ", ".join(f"{d.index}={d}" for d in devices),
                )
            elif self._config.device == AsrDevice.VULKAN:
                logger.warning(
                    "asr.device=vulkan but no Vulkan device could be enumerated; "
                    "attempting GPU init anyway since whisper.cpp's own detection is authoritative"
                )
            self._gpu_index = select_gpu_index(devices, self._config.gpu_device)
            try:
                self._model = self._load_model(use_gpu=True)
                selected = next((d for d in devices if d.index == self._gpu_index), None)
                label = str(selected) if selected is not None else f"GPU {self._gpu_index}"
                self._resolved_device_description = f"Vulkan ({label})"
                logger.info("ASR backend: %s", self._resolved_device_description)
                return
            except Exception as exc:
                # The configured device is final. Silently falling back to CPU
                # here is what turned an operator's GPU config into a CPU
                # worker with zero feedback (and left a GUI indicator saying
                # everything was fine). Surface it; the supervisor decides
                # what a failed device means (config policy, or app exit).
                raise DevicePolicyError(
                    f"GPU backend init failed on {self._config.device.value}: {exc}"
                ) from exc

        self._model = self._load_model(use_gpu=False)
        self._resolved_device_description = "CPU"
        logger.info("ASR backend: %s", self._resolved_device_description)

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
            context_params={
                "use_gpu": use_gpu,
                "flash_attn": False,
                "gpu_device": self._gpu_index if use_gpu else 0,
            },
            print_progress=False,
            print_realtime=False,
            no_context=True,
        )

    @property
    def resolved_device_description(self) -> str:
        return self._resolved_device_description

    def close(self) -> None:
        """Release the whisper model and its native memory deterministically.

        pywhispercpp's Model frees the C context in `__del__`, which only runs
        when CPython's GC collects the object -- and collection timing inside
        reference cycles is unspecified. On a VRAM-constrained GPU the
        successor generation can then race this generation's cleanup: its
        `Model(use_gpu=True)` init fails for lack of memory and, before this
        change, that failure silently became a CPU downgrade. Dropping the
        reference here (and breaking any reference cycle by clearing our own
        handle) makes the teardown deterministic: by the time the worker
        process exits, the model is either freed or in the hands of the GC of
        a process that is about to die anyway -- never still resident while
        the next generation initializes.
        """
        self._model = None

    def _detect_language(self, audio: np.ndarray) -> str:
        """Language code for this decode pass (asr.language == "auto").

        Uses the dedicated detection API on a short leading slice of the
        window (see _LANGUAGE_DETECT_MAX_SECONDS) and returns the detected
        code verbatim -- no confidence gate, no allow-list: the caption shows
        whatever the model hears, in the language it hears. The
        StreamingDecoder calls this once per utterance and locks the result;
        per-step detection flaps between acoustically similar languages
        (measured: Polish clips intermittently decoded as Ukrainian), which
        mixes languages mid-caption.
        """
        max_samples = int(_LANGUAGE_DETECT_MAX_SECONDS * 16000)
        slice_audio = audio[:max_samples] if len(audio) > max_samples else audio
        try:
            (detected, probability), _ = self._model.auto_detect_language(slice_audio)
        except Exception:
            logger.exception("language detection failed; falling back to config default")
            return self._config.language if self._config.language != "auto" else "en"
        detected = str(detected)
        logger.debug("detected language %s (%.2f)", detected, float(probability))
        return detected, float(probability)

    def detect_language(self, audio: np.ndarray) -> tuple[str, float] | None:
        """Public hook so the StreamingDecoder can lock language per utterance.
        Returns (language_code, confidence), or None on failure so the caller
        can retry on a later step. See _detect_language for details."""
        try:
            return self._detect_language(audio)
        except Exception:
            logger.exception("language detection failed")
            return None

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> list[tuple[float, float, str]]:
        """Run one decode pass over `audio` (mono float32 @16kHz). Returns a
        list of (start_seconds, end_seconds, text) segments. If the current
        backend is GPU-based and inference raises (a real, observed failure
        mode with immature Vulkan driver paths), reinitializes on CPU and
        retries once rather than crashing the whole ASR worker process.
        """
        # Explicit per-call override (used by the StreamingDecoder's per-
        # utterance language lock) wins over the config; otherwise config
        # decides between a forced language and per-call detection.
        if language is None:
            language = None if self._config.language == "auto" else self._config.language
        kwargs = {
            "single_segment": False,
            # Shrink the encoder context to the actual window length; see
            # audio_ctx_for_seconds. Without this every decode step pays for a
            # full 30s encode no matter how short the live window is.
            "audio_ctx": audio_ctx_for_seconds(len(audio) / sample_rate),
            # Disable whisper's temperature-fallback ladder. On a low-confidence
            # window (a clipped word at the window edge, breath noise, music)
            # whisper re-runs the *entire* decode at successively higher
            # temperatures, up to six times, before giving up. Measured here,
            # that turned ordinary 0.3-1.4s decode steps into 5.4s, 6.8s and
            # 17.0s stalls -- on windows of only 1.2-3.5s of audio. Those
            # stalls, not average throughput, were what overflowed the audio
            # queue and got live speech dropped. whisper.cpp's own streaming
            # example disables fallback for the same reason: for a live caption
            # a slightly worse hypothesis now beats a better one 17s late.
            "temperature_inc": 0.0,
            # Belt-and-braces cap on a single pass: even without the fallback
            # ladder, a repetition loop can emit tokens up to the text context
            # limit. A live caption window never legitimately needs this many.
            "max_tokens": _MAX_TOKENS_PER_STEP,
        }
        if language is not None:
            kwargs["language"] = language
        else:
            kwargs["language"] = self._detect_language(audio)

        try:
            segments = self._model.transcribe(audio, **kwargs)
        except Exception as exc:
            # Same policy as init: the configured device is final. The old
            # behavior -- silently reinitializing on CPU mid-stream -- made a
            # transient GPU hiccup a permanent, invisible downgrade. The
            # exception propagates; the supervisor restarts the worker and, if
            # the device is genuinely gone, exits the app rather than whisper.
            raise DevicePolicyError(
                f"GPU inference failed on {self._config.device.value}: {exc}"
            ) from exc
        return [(seg.t0 / 100.0, seg.t1 / 100.0, seg.text.strip()) for seg in segments]
