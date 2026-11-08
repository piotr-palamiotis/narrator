"""whisper.cpp binding + GPU backend detection/selection.

Uses `pywhispercpp` (a pybind11 wrapper around whisper.cpp's C API) as the
backend: `Model(name, context_params={"use_gpu": ...})` loads/keeps the model
resident, and `Model.transcribe()` runs inference on an in-memory numpy array
-- exactly the long-lived, repeatedly-called shape this streaming pipeline
needs (see streaming_decoder.py), unlike a CLI-subprocess-per-chunk or HTTP
server approach which would reload the model every call.

The binding comes from a source build (packaging/whisper_cpp/
build_vulkan_linux.sh: pywhispercpp v1.5.1's bindings over whisper.cpp
v1.9.5, GGML_VULKAN=1, CPU backend compiled -march=native). The PyPI wheel
is not usable here: it ships no Vulkan backend, and its CPU path SIGILLs at
whisper_init on pre-AVX CPUs such as the Pentium G4560 test box. See
packaging/whisper_cpp/README.md.

GPU enumeration lives in vulkan_devices.py and the processing-unit list in
units.py (asr.unit, todo C6); this module resolves the configured unit when
the worker loads the model. The interface the decoders use is
asr_backend.AsrBackend; WhisperBackend is its whisper.cpp implementation.
"""
from __future__ import annotations

import logging
import os
import re
import time

import numpy as np

from narrator.asr.asr_backend import (  # noqa: F401  (re-exported: older imports)
    AsrBackend,
    DevicePolicyError,
    ModelLoadError,
    SegmentResult,
    SegmentToken,
    WordTime,
)
from narrator.asr.vocabulary import build_initial_prompt
from narrator.asr.vulkan_devices import (  # noqa: F401  (re-exported: older imports)
    GpuSelectionError,
    VulkanDevice,
    select_gpu_index,
    vulkan_devices as _vulkan_devices,
)
from narrator.config.schema import AsrConfig

logger = logging.getLogger(__name__)


def _asr_build():
    """The installed pywhispercpp build (package metadata only; a module
    function so tests can fake it)."""
    from narrator.asr import units as unit_mod

    return unit_mod.LocalProbes().asr_build()


def _cuda_devices():
    """NVIDIA cards from nvidia-smi, PCI order ([] / None: unknown)."""
    from narrator.asr import units as unit_mod

    return unit_mod.LocalProbes().cuda_devices()


# CUDA numbers devices fastest-first by default; nvidia-smi (and so the
# cuda:<n> unit ids) uses PCI bus order. Set before the CUDA runtime starts.
_CUDA_DEVICE_ORDER_ENV = "CUDA_DEVICE_ORDER"


def _vulkan_device_names() -> list[str]:
    """Backwards-compatible helper: just the adapter names, in ggml order."""
    return [device.name for device in _vulkan_devices()]


# Roughly 200 tokens is far more than a max_context_seconds-sized window of
# real speech produces; it only bites on runaway repetition loops.
_MAX_TOKENS_PER_STEP = 200

# pywhispercpp 1.5.1's `detect_language=True` transcribe
# param is broken -- it returns zero segments on any input (verified on a
# fresh model with clean audio). Detection is done through the dedicated
# `auto_detect_language()` API instead, and its probability table is
# restricted to `asr.language_whitelist` when one is set (see
# _restrict_to_whitelist). The StreamingDecoder calls this once per
# utterance and locks the result (per-step detection measurably flaps
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

# ggml-vulkan's graph optimizer (node reordering for GPU concurrency) runs
# on the CPU for every graph, i.e. for every decoded token. On the slow
# Pentium G4560 host that costs more than it saves: measured on the
# RTX 3070, large-v3 decode 12.6 ms/token with it vs 8.5 ms/token without
# (whisper.cpp v1.9.5; v1.8.5: 9.2 vs 8.5), identical text. Disabled for
# the worker process unless the environment already sets the variable;
# NARRATOR_VK_GRAPH_OPTIMIZE=1 keeps ggml's default (on).
_VK_GRAPH_OPTIMIZE_ENV = "GGML_VK_DISABLE_GRAPH_OPTIMIZE"

# DTW alignment-head presets by model family (quantization suffixes such as
# "-q5_0" are ignored); whisper.cpp's enum names.
_DTW_PRESETS = {
    "tiny": "WHISPER_AHEADS_TINY",
    "tiny.en": "WHISPER_AHEADS_TINY_EN",
    "base": "WHISPER_AHEADS_BASE",
    "base.en": "WHISPER_AHEADS_BASE_EN",
    "small": "WHISPER_AHEADS_SMALL",
    "small.en": "WHISPER_AHEADS_SMALL_EN",
    "medium": "WHISPER_AHEADS_MEDIUM",
    "medium.en": "WHISPER_AHEADS_MEDIUM_EN",
    "large-v1": "WHISPER_AHEADS_LARGE_V1",
    "large-v2": "WHISPER_AHEADS_LARGE_V2",
    "large-v3": "WHISPER_AHEADS_LARGE_V3",
    "large-v3-turbo": "WHISPER_AHEADS_LARGE_V3_TURBO",
}
_QUANT_SUFFIX_RE = re.compile(r"-q\d(_\d|_k)?$")


def dtw_preset_name(model: str) -> str | None:
    """whisper.cpp's alignment-head preset for `model` (a name or a path to
    ggml-<name>.bin), or None when the model family is unknown."""
    name = model.rsplit("/", 1)[-1]
    if name.startswith("ggml-") and name.endswith(".bin"):
        name = name[len("ggml-") : -len(".bin")]
    return _DTW_PRESETS.get(_QUANT_SUFFIX_RE.sub("", name))


def words_from_tokens(
    tokens: list[tuple[bytes, float, float, float]],
    *,
    window_seconds: float,
    strip_brackets: bool,
) -> list[WordTime]:
    """Group a decode's text tokens `(utf-8 bytes, t0, t1, t_dtw)` -- seconds,
    negative = unknown -- into whitespace-separated words with times.

    A word starts at a token with a leading space, so the words are those of
    the decode's text split on whitespace, as the caption's words are
    ("spotkaniu." is one word in both). The start is the DTW time of its
    first token when known, else whisper's token-timestamp estimate; the end
    is its last token's t1. Times are clamped to the window (whisper's last
    timestamp can run past the audio). With `strip_brackets`, words inside a
    [bracketed annotation] are dropped, as the caption text drops them.
    """
    words: list[list] = []  # [bytes, start, end]
    for raw, t0, t1, t_dtw in tokens:
        start = t_dtw if t_dtw >= 0 else t0
        if raw.startswith(b" ") or not words:
            words.append([raw.lstrip(b" "), start, t1])
        else:
            words[-1][0] += raw
            words[-1][2] = t1
    result: list[WordTime] = []
    in_bracket = False
    for raw, start, end in words:
        text = raw.decode("utf-8", "replace").strip()
        if not text:
            continue
        if strip_brackets:
            if in_bracket or text.startswith("["):
                in_bracket = "]" not in text
                continue
        start = min(max(start, 0.0), window_seconds)
        end = min(max(end, start), window_seconds)
        result.append(WordTime(text, start, end))
    return result


_FULL_AUDIO_CTX = 1500  # whisper's native 30s encoder context
_MIN_AUDIO_CTX = 256
_AUDIO_CTX_MARGIN = 64


def _restrict_to_whitelist(
    all_probs: dict[str, float], whitelist: list[str]
) -> tuple[str, float, float] | None:
    """Re-argmax the detector's full probability table over `whitelist`.

    auto_detect_language returns softmax probabilities for every language it
    knows; restricting the decision to the operator's set is therefore exact
    (the same computation whisper does, minus the disallowed entries) -- not
    a guess. Returns (code, probability, runner_up_margin): margin is the gap
    to the best DISALLOWED... no -- to the second-best ALLOWED language, the
    quantity the StreamingDecoder gates its per-utterance lock on (a wide
    margin means the decision is safe; a narrow one means the utterance is
    genuinely ambiguous and more audio is needed). Returns None only when
    nothing in the whitelist has a usable probability.
    """
    allowed = set(whitelist)
    scored = sorted(
        ((float(prob), code) for code, prob in all_probs.items() if code in allowed),
        reverse=True,
    )
    if not scored:
        return None
    best_prob, best_code = scored[0]
    runner_up_prob = scored[1][0] if len(scored) > 1 else 0.0
    return best_code, best_prob, best_prob - runner_up_prob


# whisper's timestamp tokens are 20ms apart (<|0.00|>, <|0.02|>, ...).
_TIMESTAMP_TOKEN_SECONDS = 0.02
# Floor for the duration a timestamp ratio is measured against. The encoder
# context never shrinks below _MIN_AUDIO_CTX (~5s, see audio_ctx_for_seconds),
# so on a sub-second window a legitimate end timestamp can land seconds past
# the audio; measuring against at least 1s keeps short windows from being
# rejected for that alone.
_TIMESTAMP_RATIO_MIN_SECONDS = 1.0

_BRACKETED_ANNOTATION_RE = re.compile(r"\[[^\]]*\]")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_bracketed_annotations(text: str) -> str:
    """Remove whisper's [bracketed] sound annotations ("[oklaski]",
    "[muzyka]", "[wypowiedź w języku ukraińskim]"), tidying the spaces left
    behind."""
    if "[" not in text:
        return text
    return _WHITESPACE_RE.sub(" ", _BRACKETED_ANNOTATION_RE.sub(" ", text)).strip()


def decode_reject_reason(
    tokens: list[tuple[int, float]],
    *,
    eot: int,
    timestamp_begin: int,
    audio_seconds: float,
    min_mean_token_p: float,
    max_timestamp_ratio: float,
) -> str | None:
    """Why a decode's hypothesis should be discarded, or None to keep it.

    The two quality gates obs-localvocal applies to every inference
    (whisper-processing.cpp), over the decode's `(token_id, probability)`
    list: text tokens are ids below `eot`; ids from `timestamp_begin` up
    are timestamps, `_TIMESTAMP_TOKEN_SECONDS` apart; the specials in
    between are ignored.

      * timestamp ratio: a timestamp later than `max_timestamp_ratio` times
        the real audio length means whisper is "transcribing" audio that
        does not exist -- a hallucination/repetition loop.
      * mean token probability below `min_mean_token_p`: whisper was
        guessing (noise, music, applause decoded as words).

    A gate at 0 is off. An empty hypothesis is never rejected (nothing to
    drop).
    """
    if max_timestamp_ratio > 0:
        duration = max(audio_seconds, _TIMESTAMP_RATIO_MIN_SECONDS)
        last = max((tid for tid, _ in tokens if tid >= timestamp_begin), default=None)
        if last is not None:
            ratio = (last - timestamp_begin) * _TIMESTAMP_TOKEN_SECONDS / duration
            if ratio > max_timestamp_ratio:
                return f"timestamp ratio {ratio:.2f} > {max_timestamp_ratio:.2f}"
    if min_mean_token_p > 0:
        probabilities = [p for tid, p in tokens if tid < eot]
        if probabilities:
            mean = sum(probabilities) / len(probabilities)
            if mean < min_mean_token_p:
                return f"mean token p {mean:.2f} < {min_mean_token_p:.2f}"
    return None


# Segment decoding (decode_segment, the segmenter's decode; obs-localvocal's
# run_whisper_inference). Whisper misbehaves on very short inputs: a segment
# under 1 s is centred in 1.01 s of faint white noise, every segment gets
# 10 ms of silence on each side, and one under 50 ms is not decoded at all.
_SEGMENT_MIN_SECONDS = 0.05
_SEGMENT_SHORT_SECONDS = 1.0
_SEGMENT_PAD_TO_SECONDS = 1.01
_SEGMENT_EDGE_SECONDS = 0.01
_SEGMENT_PAD_NOISE = 0.01


def pad_segment(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
    """The decode input for a segment (see the constants above), or None
    when the segment is too short to decode. The noise is seeded by the
    segment length, so a replay pads identically."""
    n = len(audio)
    if n < _SEGMENT_MIN_SECONDS * sample_rate:
        return None
    audio = np.asarray(audio, dtype=np.float32)
    if n < _SEGMENT_SHORT_SECONDS * sample_rate:
        target = int(round(_SEGMENT_PAD_TO_SECONDS * sample_rate))
        rng = np.random.default_rng(n)
        padded = rng.uniform(-_SEGMENT_PAD_NOISE, _SEGMENT_PAD_NOISE, target).astype(np.float32)
        offset = (target - n) // 2
        padded[offset : offset + n] = audio
        audio = padded
    edge = np.zeros(int(round(_SEGMENT_EDGE_SECONDS * sample_rate)), dtype=np.float32)
    return np.concatenate([edge, audio, edge])


def filter_segment_tokens(
    tokens: list[SegmentToken], *, eot: int, drop_trailing_period: bool
) -> list[SegmentToken]:
    """The caption tokens of a decode: special and timestamp tokens (ids from
    `eot` up -- read from the model, never a hard-coded 50256, since
    large-v3's vocabulary differs) are dropped, and with
    `drop_trailing_period` so is a lone "." token right before the end (a
    segment cut mid-speech is not a sentence end) -- or an ellipsis ("...",
    "…"), which whisper ends such cuts with as often (measured on test.mp3
    with medium: "Użyjemy..." at a 7 s cut). [bracketed] annotations span
    several tokens, so they are stripped from the joined text."""
    kept = [t for t in tokens if t.id < eot]
    if drop_trailing_period and kept and _is_period_token(kept[-1].text):
        kept = kept[:-1]
    return kept


def _is_period_token(text: str) -> bool:
    text = text.strip()
    return bool(text) and all(ch in ".…" for ch in text)


def segment_text(tokens: list[SegmentToken], *, strip_brackets: bool) -> str:
    """Caption text of the kept tokens: their bytes joined and decoded once,
    [bracketed] annotations optionally stripped, whitespace tidied."""
    text = b"".join(t.raw or t.text.encode("utf-8") for t in tokens).decode("utf-8", "replace")
    if strip_brackets:
        text = strip_bracketed_annotations(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


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


class WhisperBackend(AsrBackend):
    """whisper.cpp (pywhispercpp) on the configured processing unit
    (asr.unit: cpu, cuda:<n>, vulkan:<n>, or auto). The AsrBackend
    implementation."""

    def __init__(self, config: AsrConfig) -> None:
        self._config = config
        self._model = None
        self._resolved_device_description = "unresolved"
        self._gpu_index = 0
        # The unit it runs on ("vulkan:0", "cpu"), once resolved.
        self.resolved_unit: str | None = None
        # Decodes discarded by the quality gates, by gate (diagnostics).
        self.rejected_decodes: dict[str, int] = {}
        # Event vocabulary, prepended to every decode's prompt. Starts as the
        # inline asr.vocabulary; the worker sets the resolved list (inline +
        # named list) and updates it live (set_vocabulary).
        self._vocabulary: list[str] = list(config.vocabulary)
        # Words of the latest transcribe() with their times in seconds of
        # its input (see words_from_tokens); the StreamingDecoder stamps
        # FINALs with them. Empty after a rejected or empty decode.
        self.last_word_times: list[WordTime] = []
        self._dtw = False
        self._init_model()

    def _init_model(self) -> None:
        from narrator.asr import units as unit_mod
        from narrator.asr.catalog import entry_for

        kind, index = unit_mod.parse_unit(self._config.unit)
        if kind == unit_mod.AUTO:
            # The recommended unit, among the kinds this model runs on
            # (narrator.asr.units: CUDA > discrete Vulkan GPU > CPU > iGPU).
            # Probed here, in the worker, which loads the GPU driver anyway.
            found = unit_mod.discover_units(unit_mod.LocalProbes())
            chosen = unit_mod.recommended_unit(found, entry_for(self._config.model).units)
            if chosen is None:
                reasons = "; ".join(f"{u.id}: {u.reason}" for u in found if u.reason)
                raise DevicePolicyError(f"asr.unit auto: no usable processing unit ({reasons or 'none found'})")
            logger.info("asr.unit auto resolved to %s (%s)", chosen.id, chosen.name)
            kind, index = chosen.kind, chosen.index
        if kind in (unit_mod.CUDA, unit_mod.VULKAN):
            # One GPU backend per build (build_vulkan_linux.sh, ASR_GPU), so
            # whisper.cpp's gpu_device counts only that backend's devices. A
            # unit of the other kind would land on whatever device has that
            # index in this build (or on the CPU): refuse it instead.
            build = _asr_build()
            has = build.has_cuda if kind == unit_mod.CUDA else build.has_vulkan
            if build.installed and not has:
                label = "CUDA" if kind == unit_mod.CUDA else "Vulkan"
                raise DevicePolicyError(
                    f"asr.unit {kind}:{index} needs a {label} build of whisper.cpp, and the installed "
                    f"pywhispercpp {build.version} has none (rebuild: make asr-backend ASR_GPU={kind})"
                )
        if kind == unit_mod.CUDA:
            self._init_cuda(index)
            return
        if kind == unit_mod.VULKAN:
            devices = _vulkan_devices()
            if devices:
                logger.info(
                    "Vulkan devices: %s",
                    ", ".join(f"vulkan:{d.index}={d}" for d in devices),
                )
            else:
                logger.warning(
                    "asr.unit=vulkan:%s but no Vulkan device could be enumerated; "
                    "attempting GPU init anyway since whisper.cpp's own detection is authoritative",
                    index,
                )
            self._gpu_index = select_gpu_index(devices, index)
            selected = next((d for d in devices if d.index == self._gpu_index), None)
            recorded = self._config.unit_identity
            if recorded and selected is not None and self._config.unit_kind == unit_mod.VULKAN:
                unit = unit_mod.ProcessingUnit(
                    f"vulkan:{selected.index}", unit_mod.VULKAN, selected.index, selected.name, "", selected.pci_bus_id
                )
                if not unit.matches_identity(recorded):
                    # The indices shifted (a card added or removed): running
                    # on whatever is at this index now would be a silent
                    # switch to another card. The operator decides (C8).
                    raise GpuSelectionError(
                        f"asr.unit vulkan:{selected.index} is recorded as {recorded!r}, but that index is now "
                        f"{unit.identity!r}"
                    )
            try:
                self._model = self._load_model(use_gpu=True)
                label = str(selected) if selected is not None else f"GPU {self._gpu_index}"
                self._resolved_device_description = f"Vulkan ({label})"
                self.resolved_unit = f"vulkan:{self._gpu_index}"
                logger.info("ASR backend: %s", self._resolved_device_description)
                return
            except ModelLoadError:
                raise  # not a device problem: say so, not "GPU init failed"
            except Exception as exc:
                # The configured unit is final. Silently falling back to CPU
                # here is what turned an operator's GPU config into a CPU
                # worker with zero feedback (and left a GUI indicator saying
                # everything was fine). Surface it; the supervisor decides
                # what a failed unit means (crash loop -> ASR halted, problem shown).
                raise DevicePolicyError(
                    f"GPU backend init failed on {self._config.unit}: {exc}"
                ) from exc

        self._model = self._load_model(use_gpu=False)
        self._resolved_device_description = "CPU"
        self.resolved_unit = unit_mod.CPU
        logger.info("ASR backend: %s", self._resolved_device_description)

    def _init_cuda(self, index: int) -> None:
        """Load on CUDA card `index` (nvidia-smi's index = PCI order)."""
        from narrator.asr import units as unit_mod

        os.environ.setdefault(_CUDA_DEVICE_ORDER_ENV, "PCI_BUS_ID")
        devices = _cuda_devices()
        selected = None
        if devices:
            logger.info("CUDA devices: %s", ", ".join(f"cuda:{d.index}={d.name}" for d in devices))
            selected = next((d for d in devices if d.index == index), None)
            if selected is None:
                # whisper.cpp would load on the CPU without an error.
                raise GpuSelectionError(
                    f"asr.unit cuda:{index} is not present (CUDA cards: "
                    f"{', '.join(f'cuda:{d.index}' for d in devices)})"
                )
            recorded = self._config.unit_identity
            if recorded and self._config.unit_kind == unit_mod.CUDA:
                unit = unit_mod.ProcessingUnit(
                    f"cuda:{index}", unit_mod.CUDA, index, selected.name, "discrete", selected.pci_bus_id
                )
                if not unit.matches_identity(recorded):
                    raise GpuSelectionError(
                        f"asr.unit cuda:{index} is recorded as {recorded!r}, but that index is now {unit.identity!r}"
                    )
        else:
            logger.warning(
                "asr.unit=cuda:%s but nvidia-smi listed no card; attempting CUDA init anyway", index
            )
        self._gpu_index = index
        try:
            self._model = self._load_model(use_gpu=True)
        except ModelLoadError:
            raise
        except Exception as exc:
            # The configured unit is final (no CPU fallback), as for Vulkan.
            raise DevicePolicyError(f"GPU backend init failed on cuda:{index}: {exc}") from exc
        label = selected.name if selected is not None else f"GPU {index}"
        self._resolved_device_description = f"CUDA ({label})"
        self.resolved_unit = f"cuda:{index}"
        logger.info("ASR backend: %s", self._resolved_device_description)

    def _load_model(self, use_gpu: bool):
        if use_gpu and os.environ.get("NARRATOR_VK_GRAPH_OPTIMIZE") != "1":
            # Read by ggml-vulkan at device init (see _VK_GRAPH_OPTIMIZE_ENV).
            os.environ.setdefault(_VK_GRAPH_OPTIMIZE_ENV, "1")
        import _pywhispercpp as pw
        from pywhispercpp.model import Model  # imported lazily: heavy native extension

        # flash_attn is opt-in (asr.flash_attn): whisper.cpp's Vulkan
        # flash-attention kernel is known to crash with an illegal
        # instruction (SIGILL, i.e. multiprocessing exitcode -4) on a number
        # of GPU/driver combinations, which kills the ASR worker process
        # outright -- no Python exception is raised for us to catch, so the
        # only protection is not selecting that code path by default. A crash
        # loop then halts ASR and shows a problem (app_controller, narrator.health).
        # Models live in <repo>/models (narrator.paths); one already in the
        # legacy user cache is reused rather than downloaded again.
        from narrator.paths import model_load_args

        model_name, models_dir = model_load_args(self._config.model)
        # flash_attn is always passed: whisper.cpp >= 1.8 defaults it to ON.
        flash_attn = bool(self._config.flash_attn and use_gpu)
        context_params = {
            "use_gpu": use_gpu,
            "flash_attn": flash_attn,
            "gpu_device": self._gpu_index if use_gpu else 0,
        }
        self._dtw = False
        if self._config.dtw_word_timestamps:
            preset = dtw_preset_name(self._config.model)
            if flash_attn:
                logger.warning("asr.dtw_word_timestamps is ignored while asr.flash_attn is on (whisper.cpp limitation)")
            elif preset is None:
                logger.warning(
                    "asr.dtw_word_timestamps: no alignment-head preset for model %r; using token timestamps",
                    self._config.model,
                )
            else:
                context_params["dtw_token_timestamps"] = True
                context_params["dtw_aheads_preset"] = getattr(pw.whisper_alignment_heads_preset, preset)
                self._dtw = True
        model = Model(
            model_name,
            models_dir=models_dir,
            context_params=context_params,
            print_progress=False,
            print_realtime=False,
            no_context=True,
        )
        # pywhispercpp does not raise for a model it cannot get: it logs
        # "Invalid model name", leaves model_path None and builds a context
        # around whisper.cpp's NULL -- the first decode then crashes natively.
        # Fail here, with a message the problems panel can show.
        if getattr(model, "model_path", "") is None:
            raise ModelLoadError(
                f"model {self._config.model!r} could not be loaded: it is not a local model file "
                "and not a downloadable whisper.cpp model name"
            )
        return model

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

    def _detect_language(self, audio: np.ndarray) -> tuple[str, float, float]:
        """Language decision over the configured whitelist.

        Runs the dedicated detection API on a short leading slice of the
        window (see _LANGUAGE_DETECT_MAX_SECONDS). With no whitelist the
        unrestricted top is returned with its margin over the runner-up
        (computationally identical shape to the whitelist case). The caller
        (StreamingDecoder) gates its per-utterance lock on the margin: a
        narrow gap means the utterance is ambiguous and needs more audio.
        Returns (code, probability, margin).
        """
        max_samples = int(_LANGUAGE_DETECT_MAX_SECONDS * 16000)
        slice_audio = audio[:max_samples] if len(audio) > max_samples else audio
        try:
            (detected, probability), all_probs = self._model.auto_detect_language(slice_audio)
        except Exception:
            logger.exception("language detection failed")
            raise
        whitelist = self._config.language_whitelist
        if whitelist:
            restricted = _restrict_to_whitelist(all_probs, whitelist)
            if restricted is None:
                # Validated configs always overlap the detector's table; treat
                # as a failed detection rather than inventing a language.
                raise ValueError(
                    f"none of the whitelisted languages {whitelist} appear in detection probabilities"
                )
            code, probability, margin = restricted
            if code != str(detected):
                logger.info(
                    "language detection %s (%.2f) restricted by whitelist to %s (%.2f)",
                    detected,
                    float(probability),
                    code,
                    probability,
                )
            return code, probability, margin
        # Unrestricted: margin over the best non-chosen language.
        ranked = sorted(((float(p), str(c)) for c, p in all_probs.items()), reverse=True)
        if not ranked:
            raise ValueError("detector returned no language probabilities")
        probability = float(probability)
        runner_up = ranked[1][0] if len(ranked) > 1 and ranked[1][1] != str(detected) else 0.0
        return str(detected), probability, probability - runner_up

    def detect_language(self, audio: np.ndarray) -> tuple[str, float, float] | None:
        """Public hook so the StreamingDecoder can lock language per utterance.
        Returns (language_code, confidence, margin_to_runner_up), or None on
        failure so the caller can retry on a later step."""
        try:
            return self._detect_language(audio)
        except Exception:
            logger.exception("language detection failed")
            return None

    def set_vocabulary(self, vocabulary: list[str]) -> None:
        """Replace the event vocabulary used in the prompt (hot-reload)."""
        self._vocabulary = list(vocabulary)

    def _decode_kwargs(self, audio: np.ndarray, sample_rate: int, language: str | None, prompt: str | None) -> dict:
        """whisper decode parameters for one call over `audio` (shared by
        transcribe and decode_segment)."""
        # Explicit per-call override (a language the StreamingDecoder locked
        # for this utterance) wins; otherwise the whitelist decides per call:
        # empty = let whisper detect (language=None), single/multiple = the
        # best allowed language for THIS audio.
        if language is None:
            if not self._config.language_whitelist:
                language = None
            elif len(self._config.language_whitelist) == 1:
                language = self._config.language_whitelist[0]
            else:
                language = self._detect_language(audio)[0]
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
            # Always set explicitly: pywhispercpp keeps decode params between
            # calls, so a prompt from an earlier call would otherwise leak
            # into this one. ("" clears it; the binding rejects None.)
            # The vocabulary hint goes first, then the carry-over `prompt`,
            # sized to whisper's prompt budget (build_initial_prompt).
            "initial_prompt": build_initial_prompt(self._vocabulary, prompt),
            # Per-token t0/t1 (whisper's timestamp heuristic, computed after
            # the decode -- the text is unchanged and the cost is within
            # noise): the word times of last_word_times / decode_segment's
            # tokens. Without it every token's t0/t1 is -1.
            "token_timestamps": True,
        }
        if language is not None:
            kwargs["language"] = language
        else:
            kwargs["language"] = self._detect_language(audio)[0]

        return kwargs

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
        prompt: str | None = None,
    ) -> list[tuple[float, float, str]]:
        """Run one decode pass over `audio` (mono float32 @16kHz). Returns a
        list of (start_seconds, end_seconds, text) segments. `prompt` is text
        whisper treats as what was said just before this audio (context for
        names, spelling, style); None decodes from a blank context. A failure
        raises DevicePolicyError -- the configured device is final, there is
        no silent CPU fallback.
        """
        kwargs = self._decode_kwargs(audio, sample_rate, language, prompt)
        self.last_word_times = []
        try:
            segments = self._model.transcribe(audio, **kwargs)
        except Exception as exc:
            # Same policy as init: the configured device is final. The old
            # behavior -- silently reinitializing on CPU mid-stream -- made a
            # transient GPU hiccup a permanent, invisible downgrade. The
            # exception propagates; the supervisor restarts the worker and, if
            # the device is genuinely gone, exits the app rather than whisper.
            raise DevicePolicyError(
                f"GPU inference failed on {self._config.unit}: {exc}"
            ) from exc
        reason = self._reject_reason(len(audio) / sample_rate)
        if reason is not None:
            gate = reason.split(" ", 1)[0]
            self.rejected_decodes[gate] = self.rejected_decodes.get(gate, 0) + 1
            logger.debug("decode rejected (%s): %r", reason, " ".join(seg.text for seg in segments))
            return []
        strip = self._config.strip_bracketed_annotations
        result = []
        for seg in segments:
            text = strip_bracketed_annotations(seg.text) if strip else seg.text.strip()
            result.append((seg.t0 / 100.0, seg.t1 / 100.0, text))
        self.last_word_times = self._read_word_times(len(audio) / sample_rate)
        return result

    def _read_word_times(self, window_seconds: float) -> list[WordTime]:
        """The latest decode's words with times (words_from_tokens), read
        from the whisper context's token data. Never fails a decode: on any
        error the caller just has no word times."""
        try:
            import _pywhispercpp as pw

            ctx = self._model._ctx
            eot = pw.whisper_token_eot(ctx)
            tokens = []
            for i in range(pw.whisper_full_n_segments(ctx)):
                for j in range(pw.whisper_full_n_tokens(ctx, i)):
                    data = pw.whisper_full_get_token_data(ctx, i, j)
                    if data.id >= eot:
                        continue
                    t_dtw = data.t_dtw / 100.0 if self._dtw else -1.0
                    tokens.append((pw.whisper_token_to_bytes(ctx, data.id), data.t0 / 100.0, data.t1 / 100.0, t_dtw))
            return words_from_tokens(
                tokens, window_seconds=window_seconds, strip_brackets=self._config.strip_bracketed_annotations
            )
        except Exception:
            logger.debug("could not read word timestamps", exc_info=True)
            return []

    def decode_segment(
        self,
        audio: np.ndarray,
        *,
        prompt: str | None = None,
        language: str | None = None,
        drop_trailing_period: bool = False,
        peek: bool = False,
        sample_rate: int = 16000,
    ) -> SegmentResult:
        """Decode one whole segment once (the segmenter's decode; a port of
        obs-localvocal's run_whisper_inference): pad it (pad_segment), run
        whisper with the same latency-bounding parameters as transcribe,
        read the tokens from the whisper context, drop special tokens (and
        optionally a trailing lone "."), strip [bracketed] annotations, and
        apply the quality gates. `peek` marks a partial (an open segment
        re-decoded for display); it changes nothing here but the log line,
        and lets stream_sim charge peeks and finals separately.
        A failure raises DevicePolicyError, as transcribe does."""
        started = time.monotonic()
        padded = pad_segment(audio, sample_rate)
        if padded is None:
            return SegmentResult(text="", language=language)
        kwargs = self._decode_kwargs(padded, sample_rate, language, prompt)
        try:
            self._model.transcribe(padded, **kwargs)
        except Exception as exc:
            raise DevicePolicyError(f"GPU inference failed on {self._config.unit}: {exc}") from exc
        import _pywhispercpp as pw  # the native module behind pywhispercpp.model

        ctx = self._model._ctx
        eot = pw.whisper_token_eot(ctx)
        raw: list[SegmentToken] = []
        for i in range(pw.whisper_full_n_segments(ctx)):
            for j in range(pw.whisper_full_n_tokens(ctx, i)):
                data = pw.whisper_full_get_token_data(ctx, i, j)
                piece = pw.whisper_token_to_bytes(ctx, data.id) if data.id < eot else b""
                raw.append(
                    SegmentToken(
                        id=data.id,
                        text=piece.decode("utf-8", "replace"),
                        p=float(data.p),
                        t0=max(data.t0, 0) / 100.0,
                        t1=max(data.t1, 0) / 100.0,
                        raw=piece,
                    )
                )
        decode_language = kwargs.get("language")
        reason = decode_reject_reason(
            [(t.id, t.p) for t in raw],
            eot=eot,
            timestamp_begin=pw.whisper_token_beg(ctx),
            audio_seconds=len(audio) / sample_rate,
            min_mean_token_p=self._config.min_mean_token_p,
            max_timestamp_ratio=self._config.max_timestamp_ratio,
        )
        elapsed = time.monotonic() - started
        if reason is not None:
            gate = reason.split(" ", 1)[0]
            self.rejected_decodes[gate] = self.rejected_decodes.get(gate, 0) + 1
            logger.debug("segment %s rejected (%s)", "peek" if peek else "decode", reason)
            return SegmentResult(text="", language=decode_language, decode_seconds=elapsed, rejected=reason)
        kept = filter_segment_tokens(raw, eot=eot, drop_trailing_period=drop_trailing_period)
        text = segment_text(kept, strip_brackets=self._config.strip_bracketed_annotations)
        return SegmentResult(text=text, tokens=tuple(kept), language=decode_language, decode_seconds=elapsed)

    def _reject_reason(self, audio_seconds: float) -> str | None:
        """Apply the quality gates (see decode_reject_reason) to the decode
        that just ran. Token data is read from the whisper context directly:
        pywhispercpp's Segment carries only text and timings. (`_ctx` is
        pywhispercpp's private handle; the binding is pinned to 1.5.1 by
        build_vulkan_linux.sh.)"""
        config = self._config
        if config.min_mean_token_p <= 0 and config.max_timestamp_ratio <= 0:
            return None
        import _pywhispercpp as pw  # the native module behind pywhispercpp.model

        ctx = self._model._ctx
        tokens = []
        for i in range(pw.whisper_full_n_segments(ctx)):
            for j in range(pw.whisper_full_n_tokens(ctx, i)):
                data = pw.whisper_full_get_token_data(ctx, i, j)
                tokens.append((data.id, data.p))
        return decode_reject_reason(
            tokens,
            eot=pw.whisper_token_eot(ctx),
            timestamp_begin=pw.whisper_token_beg(ctx),
            audio_seconds=audio_seconds,
            min_mean_token_p=config.min_mean_token_p,
            max_timestamp_ratio=config.max_timestamp_ratio,
        )
