"""Processing units (todo C6): the CPU, each CUDA card and each Vulkan
device, as one list the operator picks `asr.unit` from.

Unit ids: "cpu", "cuda:<n>" (nvidia-smi's index, i.e. PCI order) and
"vulkan:<n>" (ggml's Vulkan index, see vulkan_devices.py). An NVIDIA card
appears twice (CUDA and Vulkan), an AMD or Intel GPU once (Vulkan). `auto`
is the recommended unit, resolved when the ASR worker starts: CUDA > a
discrete Vulkan GPU > the CPU > an integrated GPU (an iGPU only when nothing
else can run the model).

Indices shift when cards are added or removed, so a GPU unit also has an
identity: its name plus PCI bus id ("NVIDIA GeForce RTX 3070 @
0000:01:00.0"), recorded in `asr.unit_identity` when the operator picks it.
A recorded identity that no longer matches the device at that index is a
problem (narrator.health) and the worker refuses to start on it -- never a
silent re-map onto another card.

Every unit the machine has is listed, with `usable` and the reason when it
is not (the CPU when the installed build's CPU backend would SIGILL here, a
CUDA card while no CUDA build of whisper.cpp is installed, a Vulkan GPU
without a Vulkan build). The probes are duck-typed (`LocalProbes` here,
narrator.health.checks.Environment for the app, fakes in tests).
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from narrator.asr.vulkan_devices import normalize_pci_bus_id

logger = logging.getLogger(__name__)

AUTO = "auto"
CPU = "cpu"
CUDA = "cuda"
VULKAN = "vulkan"
UNIT_KINDS = (CPU, CUDA, VULKAN)

_UNIT_RE = re.compile(r"^(?:auto|cpu|(?:vulkan|cuda):\d+)$")
UNIT_FORMS = "auto, cpu, vulkan:<n> or cuda:<n>"

# pywhispercpp versions built by packaging/whisper_cpp/build_vulkan_linux.sh
# (Vulkan or CUDA)
# carry this local version tag; the plain PyPI wheel does not.
SOURCE_BUILD_TAG = "+whispercpp."

NVIDIA_SMI_TIMEOUT_SECONDS = 10.0


def normalize_unit(value) -> str:
    """The canonical spelling of a unit id ("Vulkan:0 " -> "vulkan:0");
    raises ValueError for anything else."""
    text = str(value).strip().lower()
    if not _UNIT_RE.match(text):
        raise ValueError(f"asr.unit must be {UNIT_FORMS}, got {value!r}")
    return text


def parse_unit(unit: str) -> tuple[str, int | None]:
    """("vulkan", 0) for "vulkan:0", ("cpu", None), ("auto", None)."""
    kind, _, index = normalize_unit(unit).partition(":")
    return kind, (int(index) if index else None)


def unit_from_legacy(device, gpu_device) -> str:
    """The unit for the retired `asr.device` + `asr.gpu_device` pair:
    cpu -> cpu; an explicit GPU index -> vulkan:<n> (the only GPU backend
    there was); auto/vulkan with gpu_device auto -> auto (the first discrete
    GPU, as before)."""
    device = getattr(device, "value", device)
    device = str(device).strip().lower() if device is not None else AUTO
    if device not in (AUTO, VULKAN, CPU):
        raise ValueError(f"asr.device must be auto, vulkan or cpu, got {device!r}")
    index = None
    if isinstance(gpu_device, int) and not isinstance(gpu_device, bool) and gpu_device >= 0:
        index = gpu_device
    elif isinstance(gpu_device, str) and gpu_device.strip().isdigit():
        index = int(gpu_device)
    elif not (gpu_device is None or (isinstance(gpu_device, str) and gpu_device.strip().lower() == AUTO)):
        raise ValueError(f"asr.gpu_device must be auto or a GPU index, got {gpu_device!r}")
    if device == CPU:
        return CPU
    return f"{VULKAN}:{index}" if index is not None else AUTO


def unit_or_legacy(value) -> str:
    """`value` as a unit id, accepting a retired asr.device value too
    ("vulkan" -> auto, "cpu" -> cpu) for older payloads and caches."""
    try:
        return normalize_unit(value)
    except ValueError:
        return unit_from_legacy(value, None)


def cache_device_key(unit: str) -> str:
    """The device part of narrator-tune's cached bench keys: the unit's kind
    ("vulkan:0" -> "vulkan", as asr.device was spelled), so benches stored
    before asr.unit existed stay valid. Which GPU it was is the separate
    machine identity in the same key."""
    return unit_or_legacy(unit).partition(":")[0]


def identity_of(name: str, pci_bus_id: str | None) -> str:
    return f"{name} @ {pci_bus_id}" if pci_bus_id else name


def split_identity(identity: str) -> tuple[str, str | None]:
    name, sep, pci = identity.rpartition(" @ ")
    if not sep:
        return identity.strip(), None
    return name.strip(), normalize_pci_bus_id(pci)


@dataclass(frozen=True)
class AsrBuild:
    """The installed pywhispercpp, from package metadata only."""

    installed: bool
    version: str | None = None
    has_vulkan: bool = False
    has_cuda: bool = False

    @property
    def source_build(self) -> bool:
        return bool(self.version and SOURCE_BUILD_TAG in self.version)


@dataclass(frozen=True)
class GpuDevice:
    """One GPU from a probe (Vulkan or nvidia-smi)."""

    index: int
    name: str
    discrete: bool
    pci_bus_id: str | None = None
    vram_mb: int | None = None


@dataclass(frozen=True)
class ProcessingUnit:
    id: str  # "cpu", "cuda:0", "vulkan:1"
    kind: str  # cpu | cuda | vulkan
    index: int | None
    name: str
    device_type: str  # cpu | discrete | integrated
    pci_bus_id: str | None = None
    vram_mb: int | None = None
    usable: bool = True
    reason: str | None = None  # why it is not usable

    @property
    def identity(self) -> str:
        """Name + PCI bus id; "" for the CPU (there is only one)."""
        return "" if self.kind == CPU else identity_of(self.name, self.pci_bus_id)

    @property
    def backend_label(self) -> str:
        return {CPU: "CPU", CUDA: "CUDA", VULKAN: "Vulkan"}[self.kind]

    @property
    def label(self) -> str:
        return f"{self.id} ({self.name})" if self.kind != CPU else f"cpu ({self.name})"

    def matches_identity(self, recorded: str) -> bool:
        """Is this the device `recorded` (an asr.unit_identity) names? With a
        PCI bus id on both sides, bus id and name must both match (a card
        moved to another slot is a different identity on purpose: the
        operator confirms it); otherwise the name decides."""
        name, pci = split_identity(recorded)
        if name.casefold() != self.name.casefold():
            return False
        if pci and self.pci_bus_id:
            return pci == normalize_pci_bus_id(self.pci_bus_id)
        return True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "index": self.index,
            "name": self.name,
            "type": self.device_type,
            "backend": self.backend_label,
            "pci_bus_id": self.pci_bus_id,
            "vram_mb": self.vram_mb,
            "usable": self.usable,
            "reason": self.reason,
            "identity": self.identity,
        }


# ----------------------------------------------------------------- probes


def parse_cpuinfo(text: str) -> tuple[set[str] | None, str | None]:
    """(flags, model name) from /proc/cpuinfo."""
    flags = name = None
    for line in text.splitlines():
        if flags is None and line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
        elif name is None and line.startswith("model name"):
            name = line.split(":", 1)[1].strip()
    return flags, name


def parse_nvidia_smi_devices(text: str) -> list[GpuDevice]:
    """`nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total
    --format=csv,noheader,nounits` -> GpuDevices (CUDA units)."""
    devices = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        try:
            vram = int(float(parts[3]))
        except ValueError:
            vram = None
        devices.append(GpuDevice(int(parts[0]), parts[1], True, normalize_pci_bus_id(parts[2]), vram))
    return devices


class LocalProbes:
    """The probes run in the calling process (the ASR worker, bench,
    narrator-tune). The app itself uses narrator.health.checks.Environment,
    which caches them and enumerates Vulkan in a subprocess."""

    def asr_build(self) -> AsrBuild:
        """From package metadata only: importing the native module would
        load whisper.cpp into the caller."""
        import importlib.metadata as metadata

        try:
            dist = metadata.distribution("pywhispercpp")
        except metadata.PackageNotFoundError:
            return AsrBuild(installed=False)
        files = [str(f) for f in (dist.files or [])]
        return AsrBuild(
            installed=True,
            version=dist.version,
            has_vulkan=any("ggml-vulkan" in f for f in files),
            has_cuda=any("ggml-cuda" in f for f in files),
        )

    def _cpuinfo(self) -> tuple[set[str] | None, str | None]:
        if not sys.platform.startswith("linux"):
            return None, None
        try:
            return parse_cpuinfo(Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return None, None

    def cpu_flags(self) -> set[str] | None:
        """The CPU's instruction-set flags (/proc/cpuinfo), None when unknown."""
        return self._cpuinfo()[0]

    def cpu_name(self) -> str | None:
        return self._cpuinfo()[1]

    def vulkan_loader_present(self) -> bool:
        import ctypes

        for name in ("libvulkan.so.1", "vulkan-1.dll"):
            try:
                ctypes.CDLL(name)
                return True
            except OSError:
                continue
        return False

    def vulkan_devices(self) -> list[GpuDevice] | None:
        from narrator.asr.vulkan_devices import vulkan_devices

        return [
            GpuDevice(d.index, d.name, d.is_discrete, d.pci_bus_id, d.vram_mb) for d in vulkan_devices()
        ]

    def cuda_devices(self) -> list[GpuDevice] | None:
        """NVIDIA cards from nvidia-smi ([] without the NVIDIA driver, None
        when nvidia-smi exists but fails)."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_SECONDS, check=True,
            ).stdout
        except FileNotFoundError:
            return []
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("nvidia-smi failed: %s", exc)
            return None
        return parse_nvidia_smi_devices(out)


def cpu_backend_problem(probes) -> str | None:
    """Why the CPU backend would not run here, None when it would.

    The PyPI pywhispercpp wheel is built for AVX CPUs and dies with SIGILL at
    whisper_init on pre-AVX ones (the Pentium G4560 box); the source build
    (build_vulkan_linux.sh, version tagged +whispercpp.X) is compiled
    -march=native, so it runs on the machine it was built on."""
    build = probes.asr_build()
    if not build.installed:
        return "the ASR package (pywhispercpp) is not installed"
    if build.source_build:
        return None
    flags = probes.cpu_flags()
    if flags is not None and "avx" not in flags:
        return (
            f"this CPU has no AVX and the installed pywhispercpp {build.version} is the PyPI wheel, "
            "whose CPU backend crashes with an illegal instruction (SIGILL) here; "
            "rebuild it from source with `make asr-backend`"
        )
    return None


def _vulkan_reason(build: AsrBuild) -> str | None:
    if not build.installed:
        return "the ASR package (pywhispercpp) is not installed"
    if not build.has_vulkan:
        return (
            f"the installed pywhispercpp {build.version} has no Vulkan backend "
            "(build it with `make asr-backend ASR_GPU=vulkan`)"
        )
    return None


def _cuda_reason(build: AsrBuild) -> str | None:
    if not build.installed:
        return "the ASR package (pywhispercpp) is not installed"
    if not build.has_cuda:
        return (
            f"the installed pywhispercpp {build.version} has no CUDA backend "
            "(build it with `make asr-backend ASR_GPU=cuda`; needs the CUDA toolkit)"
        )
    return None


def discover_units(probes) -> list[ProcessingUnit]:
    """Every unit of this machine, usable or not: the CPU first, then the
    CUDA cards, then the Vulkan devices. Probe failures leave that kind's
    units out (the setup checks report the failure)."""
    build = probes.asr_build()
    cpu_reason = cpu_backend_problem(probes)
    try:
        cpu_name = probes.cpu_name() or "CPU"
    except Exception:
        cpu_name = "CPU"
    units = [ProcessingUnit(CPU, CPU, None, cpu_name, "cpu", usable=cpu_reason is None, reason=cpu_reason)]

    cuda_reason = _cuda_reason(build)
    for d in probes.cuda_devices() or []:
        units.append(ProcessingUnit(
            f"{CUDA}:{d.index}", CUDA, d.index, d.name, "discrete", d.pci_bus_id, d.vram_mb,
            usable=cuda_reason is None, reason=cuda_reason,
        ))

    vulkan_reason = _vulkan_reason(build)
    gpus = probes.vulkan_devices() if probes.vulkan_loader_present() else []
    for d in gpus or []:
        units.append(ProcessingUnit(
            f"{VULKAN}:{d.index}", VULKAN, d.index, d.name, "discrete" if d.discrete else "integrated",
            d.pci_bus_id, d.vram_mb, usable=vulkan_reason is None, reason=vulkan_reason,
        ))
    return units


def _rank(unit: ProcessingUnit) -> int:
    if unit.kind == CUDA:
        return 0
    if unit.kind == VULKAN and unit.device_type == "discrete":
        return 1
    if unit.kind == CPU:
        return 2
    return 3  # an integrated GPU: only when nothing else runs the model


def recommended_unit(units: list[ProcessingUnit], kinds: tuple[str, ...] | None = None) -> ProcessingUnit | None:
    """CUDA > discrete Vulkan GPU > CPU > integrated GPU, among the usable
    units of the `kinds` the model can run on (None = any)."""
    candidates = [u for u in units if u.usable and (kinds is None or u.kind in kinds)]
    if not candidates:
        return None
    return min(candidates, key=lambda u: (_rank(u), units.index(u)))


def find_unit(units: list[ProcessingUnit], unit_id: str) -> ProcessingUnit | None:
    return next((u for u in units if u.id == unit_id), None)


def resolve_unit(spec: str, units: list[ProcessingUnit], kinds: tuple[str, ...] | None = None) -> ProcessingUnit | None:
    """The unit `spec` (an asr.unit value) means here: auto -> the
    recommended one; an explicit id -> that unit, None when absent."""
    if normalize_unit(spec) == AUTO:
        return recommended_unit(units, kinds)
    return find_unit(units, normalize_unit(spec))


def same_card(units: list[ProcessingUnit], unit: ProcessingUnit) -> list[ProcessingUnit]:
    """The other units on the same physical card (an NVIDIA GPU's CUDA and
    Vulkan entries), matched by PCI bus id."""
    if not unit.pci_bus_id:
        return []
    return [u for u in units if u is not unit and u.pci_bus_id == unit.pci_bus_id]

