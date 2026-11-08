"""The model catalog (todo C6): each entry is a model, the ASR backend that
runs it, its file(s), and the kinds of processing unit that backend can use
(like obs-localvocal's model list).

Today every entry is a whisper.cpp ggml model (CPU, CUDA, Vulkan). The
other backends are listed so the unit rules exist before their models do:
ONNX models (sherpa-onnx/onnxruntime) run on CPU and CUDA but not Vulkan,
NeMo needs CUDA. A model name that is not in the catalog (a custom ggml
file, a path) is taken as a whisper.cpp model, which is what the worker
loads it as.

Installed or not comes from narrator.paths (models/, then pywhispercpp's
legacy cache). This module never imports pywhispercpp's native code, so the
config schema can use it for validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AsrBackendKind(str, Enum):
    WHISPER_CPP = "whisper.cpp"
    ONNX = "onnx"  # sherpa-onnx / onnxruntime (todo C6, not built yet)
    NEMO = "nemo"  # NVIDIA NeMo, PyTorch (todo C6, not built yet)


# Unit kinds (narrator.asr.units: "cpu", "cuda", "vulkan") each backend can
# run on. A model's own entry may narrow this further.
BACKEND_UNITS: dict[AsrBackendKind, tuple[str, ...]] = {
    AsrBackendKind.WHISPER_CPP: ("cpu", "cuda", "vulkan"),
    AsrBackendKind.ONNX: ("cpu", "cuda"),
    AsrBackendKind.NEMO: ("cuda",),
}

# Approximate ggml download sizes (MB): the catalog's size column and the
# "enough disk to download?" setup check. A quantized variant ("-q5_0") is
# about half its base model.
WHISPER_MODEL_SIZES_MB = {
    "tiny": 78, "base": 148, "small": 488, "medium": 1530,
    "large-v1": 3100, "large-v2": 3100, "large-v3": 3100, "large": 3100, "large-v3-turbo": 1650,
}

# The whisper.cpp models offered when pywhispercpp's list is not importable
# (it is plain Python, but a broken install should not empty the catalog).
_FALLBACK_WHISPER_MODELS = (
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo",
)


@dataclass(frozen=True)
class CatalogEntry:
    model: str
    backend: AsrBackendKind
    files: tuple[str, ...]
    size_mb: float | None = None
    # None = every unit kind the backend supports (BACKEND_UNITS).
    unit_kinds: tuple[str, ...] | None = None
    note: str = ""

    @property
    def units(self) -> tuple[str, ...]:
        return self.unit_kinds if self.unit_kinds is not None else BACKEND_UNITS[self.backend]

    def supports(self, unit_kind: str) -> bool:
        return unit_kind in self.units

    def to_dict(self, installed: bool | None = None) -> dict:
        row = {
            "model": self.model,
            "backend": self.backend.value,
            "files": list(self.files),
            "units": list(self.units),
            "size_mb": self.size_mb,
        }
        if installed is not None:
            row["installed"] = installed
        return row


# Entries of the other backends, once they exist (todo C6: Canary/Parakeet).
# Empty today; tests add entries here to check the unit rules.
EXTRA_ENTRIES: list[CatalogEntry] = []


def whisper_model_size_mb(model: str) -> float | None:
    base = model.split("-q")[0]
    size = WHISPER_MODEL_SIZES_MB.get(base.removesuffix(".en"))
    if size is None:
        return None
    return size / 2 if "-q" in model else size


def whisper_entry(model: str) -> CatalogEntry:
    return CatalogEntry(
        model=model,
        backend=AsrBackendKind.WHISPER_CPP,
        files=(f"ggml-{model}.bin",),
        size_mb=whisper_model_size_mb(model),
    )


def _whisper_model_names() -> list[str]:
    try:
        from pywhispercpp.constants import AVAILABLE_MODELS  # plain Python, no native code

        return list(AVAILABLE_MODELS)
    except Exception:
        return list(_FALLBACK_WHISPER_MODELS)


def entry_for(model: str) -> CatalogEntry:
    """The catalog entry of `model`; a name the catalog does not know (a
    custom ggml file or a path) is a whisper.cpp model."""
    for entry in EXTRA_ENTRIES:
        if entry.model == model:
            return entry
    return whisper_entry(model)


def catalog(extra_models: list[str] | None = None) -> list[CatalogEntry]:
    """Every known model: whisper.cpp's downloadable ones (plus any
    `extra_models`, e.g. locally cached files), then the other backends'."""
    names = list(dict.fromkeys([*_whisper_model_names(), *(extra_models or [])]))
    others = {e.model for e in EXTRA_ENTRIES}
    return [whisper_entry(n) for n in names if n not in others] + list(EXTRA_ENTRIES)


def is_installed(entry: CatalogEntry) -> bool:
    from narrator.paths import find_model_file

    if entry.backend == AsrBackendKind.WHISPER_CPP:
        return find_model_file(entry.model) is not None
    return all(find_model_file(name) is not None for name in entry.files)


def catalog_rows(current_model: str = "") -> list[dict]:
    """The catalog for the web UI: each entry with `installed`."""
    try:
        from narrator.paths import cached_model_names

        cached = cached_model_names()
    except Exception:
        cached = []
    extra = [*cached, *([current_model] if current_model else [])]
    return [entry.to_dict(installed=is_installed(entry)) for entry in catalog(extra)]


def unit_kind_problem(model: str, unit_kind: str) -> str | None:
    """Why `model` cannot run on a unit of `unit_kind` ("cpu", "cuda",
    "vulkan"), None when it can. "auto" fits every model (it is resolved
    among the model's own unit kinds)."""
    if unit_kind == "auto":
        return None
    entry = entry_for(model)
    if entry.supports(unit_kind):
        return None
    return (
        f"model {model!r} ({entry.backend.value}) runs on {', '.join(entry.units)}, not on {unit_kind}"
    )
