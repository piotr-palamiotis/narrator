"""The setup checks. Each is a pure function of the config, an Environment
(the probes: Vulkan, CPU flags, the installed ASR build, model files, audio
devices, the NDI runtime, free disk) and, for the runtime ones, a
RuntimeState snapshot of the controller. Tests pass a fake Environment, so
no GPU, microphone or NDI runtime is needed.

Two kinds:
  - static checks (`static_problems`): about the machine and the config,
    run by HealthMonitor at startup, on config change and periodically, on
    its own thread;
  - runtime problems (`runtime_problems`): about what the controller
    observed (the ASR worker halted after crashing, audio capture or NDI
    output failed to start, the NDI audio source is missing). Only attribute
    reads, recomputed on every status snapshot so they show at once.

Nothing here changes the config or restarts anything: a problem offers fix
actions (problems.Fix), and only the operator's click applies one.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from narrator.asr import units as unit_mod
from narrator.asr.catalog import unit_kind_problem, whisper_model_size_mb
from narrator.asr.units import (  # noqa: F401  (re-exported: the probe types live in narrator.asr.units)
    SOURCE_BUILD_TAG,
    AsrBuild,
    GpuDevice,
    LocalProbes,
    ProcessingUnit,
    cpu_backend_problem,
)
from narrator.config.schema import RootConfig
from narrator.health.problems import (
    FIX_ASR_PIN_UNIT,
    FIX_ASR_RETRY,
    FIX_ASR_USE_CPU,
    FIX_AUDIO_RETRY,
    FIX_NDI_RETRY,
    FIX_UI_ASR,
    FIX_UI_AUDIO,
    FIX_UI_NDI,
    Fix,
    Problem,
    Severity,
    Subsystem,
    use_unit_fix,
)

logger = logging.getLogger(__name__)

# narrator-tune refuses to cache excerpts below its own min_free_disk_mb
# (tune.yaml); narrator cannot read that file (no narrator_tune import), so
# the hint uses the shipped default.
TUNE_CACHE_MIN_FREE_MB = 2048

# The Vulkan probe subprocess (python start + one vkCreateInstance).
VULKAN_PROBE_TIMEOUT_SECONDS = 30.0

# How long the NDI audio source may stay missing before it is a problem
# (discovery itself takes a second or two).
NDI_SOURCE_GRACE_SECONDS = 5.0

_FIX_RETRY_ASR = Fix(FIX_ASR_RETRY, "Retry")
_FIX_USE_CPU = Fix(FIX_ASR_USE_CPU, "Use CPU instead")
_FIX_PICK_UNIT = Fix(FIX_UI_ASR, "ASR settings")
_FIX_RETRY_AUDIO = Fix(FIX_AUDIO_RETRY, "Retry")
_FIX_PICK_MIC = Fix(FIX_UI_AUDIO, "Audio settings")
_FIX_RETRY_NDI = Fix(FIX_NDI_RETRY, "Retry")
_FIX_NDI_SETTINGS = Fix(FIX_UI_NDI, "NDI settings")


class Environment(LocalProbes):
    """The real probes (narrator.asr.units.LocalProbes plus the app's own).
    Expensive ones (a Vulkan instance, nvidia-smi, the package metadata) are
    cached until `invalidate()`, which the monitor calls on a config change
    or an operator's re-check, so the periodic pass stays cheap. Vulkan is
    enumerated in a subprocess."""

    def __init__(self) -> None:
        self._cache: dict = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def invalidate(self) -> None:
        self._cache.clear()

    def _cached(self, key, fn):
        # Read once, write once: invalidate() may clear the dict from another
        # thread (the web UI's snapshot reads probes too) at any point. One
        # lock per probe: the health thread and the unit-usage thread must
        # not both run the Vulkan subprocess, while the web UI's cheap reads
        # (build info, CPU flags) never wait behind it.
        try:
            return self._cache[key]
        except KeyError:
            pass
        with self._locks_guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            try:
                return self._cache[key]
            except KeyError:
                value = fn()
                self._cache[key] = value
                return value

    def vulkan_loader_present(self) -> bool:
        return self._cached("vk_loader", super().vulkan_loader_present)

    def vulkan_devices(self) -> list[GpuDevice] | None:
        """The Vulkan GPUs in ggml's order, None when the probe failed.
        Enumerated in a subprocess (narrator.health.vulkan_probe): the
        driver is never loaded into the app process (the frozen build, which
        cannot run `-m`, enumerates in-process)."""

        def probe() -> list[GpuDevice] | None:
            if getattr(sys, "frozen", False):
                return LocalProbes.vulkan_devices(self)
            import json
            import subprocess

            try:
                out = subprocess.run(
                    [sys.executable, "-m", "narrator.health.vulkan_probe"],
                    capture_output=True, text=True, timeout=VULKAN_PROBE_TIMEOUT_SECONDS, check=True,
                ).stdout
                return [
                    GpuDevice(
                        int(d["index"]), str(d["name"]), bool(d["discrete"]),
                        d.get("pci_bus_id"), d.get("vram_mb"),
                    )
                    for d in json.loads(out)
                ]
            except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Vulkan device probe failed: %s", exc)
                return None

        return self._cached("vk_devices", probe)

    def cuda_devices(self) -> list[GpuDevice] | None:
        return self._cached("cuda_devices", super().cuda_devices)

    def _cpuinfo(self):
        return self._cached("cpuinfo", super()._cpuinfo)

    def asr_build(self) -> AsrBuild:
        return self._cached("asr_build", super().asr_build)

    def units(self) -> list[ProcessingUnit]:
        """Every processing unit (narrator.asr.units.discover_units), from
        the cached probes."""
        return unit_mod.discover_units(self)

    def model_file(self, model: str) -> Path | None:
        from narrator.paths import find_model_file

        return find_model_file(model)

    def downloadable_models(self) -> set[str]:
        def probe() -> set[str]:
            try:
                from pywhispercpp.constants import AVAILABLE_MODELS
            except Exception:
                return set()
            return set(AVAILABLE_MODELS)

        return self._cached("downloadable", probe)

    def models_dir(self) -> Path | None:
        from narrator.paths import models_dir

        return models_dir()

    def input_devices(self) -> list[tuple[str, int]]:
        """(name, max input channels) of every PortAudio input device."""
        from narrator.audio.devices import list_input_devices

        return [(d.name, int(d.max_input_channels)) for d in list_input_devices()]

    def ndi_runtime_error(self) -> str | None:
        """None when cyndilib and its NDI library load, else the reason.
        A success is cached (the library cannot disappear from a running
        process); a failure is re-probed on every pass."""
        if self._cache.get("ndi_ok"):
            return None
        if importlib.util.find_spec("cyndilib") is None:
            return "the cyndilib package is not installed"
        try:
            from cyndilib import Sender  # noqa: F401  (loads the bundled libndi)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        self._cache["ndi_ok"] = True
        return None

    def free_mb(self, path: Path) -> float | None:
        """Free space on the disk holding `path` (its nearest existing parent)."""
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            return shutil.disk_usage(probe).free / (1024 * 1024)
        except OSError:
            return None

    def var_dir(self) -> Path:
        from narrator.paths import var_dir

        return var_dir()



@dataclass(frozen=True)
class RuntimeState:
    """What the controller observed; built by AppController.runtime_state()."""

    asr_alive: bool = False
    asr_halted: bool = False
    asr_crash_streak: int = 0
    asr_last_exit_code: int | None = None
    asr_last_error: str | None = None
    audio_running: bool = True
    audio_error: str | None = None
    ndi_audio_source: str | None = None  # set when audio.device is ndi:<name>
    ndi_audio_status: str | None = None  # NdiAudioCapture.status
    ndi_audio_status_age_s: float = 0.0
    ndi_audio_detail: str | None = None
    ndi_output_error: str | None = None


# --------------------------------------------------------------- ASR checks


def _unit_fixes(
    config: RootConfig,
    env: Environment,
    units: list[ProcessingUnit] | None = None,
    *,
    retry: bool = False,
    also: tuple[Fix, ...] = (),
) -> tuple[Fix, ...]:
    """Fixes for a unit problem: [Retry], the problem's own fixes, "Use
    <recommended GPU>" (when `units` is known and that is another unit),
    "Use CPU instead" (where the model has a CPU build and it runs here),
    and the ASR tab to pick any other unit."""
    asr = config.asr
    kinds = _model_kinds(config)
    fixes = ([_FIX_RETRY_ASR] if retry else []) + list(also)
    if units is not None:
        best = unit_mod.recommended_unit(units, kinds)
        if best is not None and best.id != asr.unit and best.kind != unit_mod.CPU:
            fix = use_unit_fix(best.id, best.name)
            if fix not in fixes:
                fixes.append(fix)
    if asr.unit_kind != unit_mod.CPU and unit_mod.CPU in kinds and cpu_backend_problem(env) is None:
        fixes.append(_FIX_USE_CPU)
    fixes.append(_FIX_PICK_UNIT)
    return tuple(fixes)


def _model_kinds(config: RootConfig) -> tuple[str, ...]:
    from narrator.asr.catalog import entry_for

    return entry_for(config.asr.model).units


def _vulkan_unavailable(env: Environment, severity: Severity, consequence: str, fixes) -> Problem | None:
    """Why no Vulkan GPU can be used (no Vulkan build, no loader, no GPU),
    None when one can; a failed probe is a warning (the check is skipped)."""
    build = env.asr_build()
    if not build.has_vulkan:
        return Problem(
            "asr.unit.no_vulkan_build", severity, Subsystem.SETUP,
            f"The installed pywhispercpp {build.version} has no Vulkan backend (the PyPI wheel, or a CUDA "
            f"build), so {consequence}. Build the Vulkan backend with `make asr-backend ASR_GPU=vulkan`.",
            fixes(),
        )
    if not env.vulkan_loader_present():
        return Problem(
            "asr.unit.no_vulkan_loader", severity, Subsystem.SETUP,
            "The Vulkan loader (libvulkan.so.1) is not installed, so "
            f"{consequence}. Install libvulkan1 and the GPU driver's Vulkan ICD "
            "(mesa-vulkan-drivers on AMD/Intel, the NVIDIA driver on NVIDIA).",
            fixes(),
        )
    devices = env.vulkan_devices()
    if devices is None:
        return Problem(
            "asr.unit.probe_failed", Severity.WARNING, Subsystem.ASR,
            "The Vulkan GPUs could not be listed (the probe crashed or timed out); "
            "the GPU check is skipped. See the log.",
        )
    if not devices:
        return Problem(
            "asr.unit.no_gpu", severity, Subsystem.ASR,
            f"No Vulkan GPU was found (driver missing, or the user lacks the render/video group), so {consequence}.",
            fixes(),
        )
    return None


def _check_identity(config: RootConfig, env: Environment, units: list[ProcessingUnit], unit: ProcessingUnit) -> list[Problem]:
    """Is the device at the configured index still the card the operator
    picked (asr.unit_identity)? A mismatch is never re-mapped silently: the
    worker refuses it and the operator chooses (the recorded card at its new
    index, or the card now at this index)."""
    recorded = config.asr.unit_identity
    if not recorded:
        return [Problem(
            "asr.unit.identity_unrecorded", Severity.INFO, Subsystem.ASR,
            f"asr.unit {unit.id} ({unit.name}) has no recorded identity (asr.unit_identity), so a shift of the "
            "GPU indices (a card added or removed) would go unnoticed. Remember this card to pin it.",
            (Fix(FIX_ASR_PIN_UNIT, "Remember this card"),),
        )]
    if unit.matches_identity(recorded):
        if unit.device_type == "integrated":
            return [Problem(
                "asr.unit.igpu", Severity.INFO, Subsystem.ASR,
                f"asr.unit {unit.id} ({unit.name}) is an integrated GPU, which is much slower than a discrete card.",
            )]
        return []
    moved = next(
        (u for u in units if u.kind == unit.kind and u.id != unit.id and u.matches_identity(recorded)), None
    )
    also = [use_unit_fix(moved.id, moved.name)] if moved is not None else []
    also.append(Fix(FIX_ASR_PIN_UNIT, f"Use {unit.name} instead"))
    where = f" The recorded card is now {moved.id}." if moved is not None else " The recorded card is not present."
    return [Problem(
        "asr.unit.identity_mismatch", Severity.ERROR, Subsystem.ASR,
        f"asr.unit {unit.id} is recorded as {recorded!r}, but {unit.id} is now {unit.identity!r}: the GPU indices "
        f"changed (a card was added or removed) or asr.unit was edited by hand.{where} ASR does not start on a "
        "different card by itself; pick the unit to use.",
        _unit_fixes(config, env, units, also=tuple(also)),
    )]


def _missing_unit(config: RootConfig, env: Environment, units: list[ProcessingUnit]) -> list[Problem]:
    asr = config.asr
    kind = asr.unit_kind
    same_kind = [u for u in units if u.kind == kind]
    listed = ", ".join(f"{u.id}={u.name}" for u in same_kind) or "none"
    moved = None
    if asr.unit_identity:
        moved = next((u for u in same_kind if u.matches_identity(asr.unit_identity)), None)
    where = f" The recorded card ({asr.unit_identity}) is now {moved.id}." if moved is not None else ""
    label = "Vulkan devices" if kind == unit_mod.VULKAN else "CUDA cards"
    return [Problem(
        "asr.unit.missing", Severity.ERROR, Subsystem.ASR,
        f"asr.unit is {asr.unit}, but that unit is not on this machine ({label} here: {listed}).{where} "
        "No captions from speech until another unit is picked.",
        _unit_fixes(config, env, units, also=(use_unit_fix(moved.id, moved.name),) if moved is not None else ()),
    )]


def check_asr_unit(config: RootConfig, env: Environment) -> list[Problem]:
    """Is the configured processing unit (asr.unit) present, still the card
    the operator picked, usable by the installed build, and one the model
    runs on? Each problem offers other units; nothing is changed here."""
    asr = config.asr
    build = env.asr_build()
    if not build.installed:
        return [Problem(
            "asr.build.missing", Severity.ERROR, Subsystem.SETUP,
            "The ASR package (pywhispercpp) is not installed, so there are no live captions from speech. "
            "Run `make install` (or `make asr-backend`).",
        )]
    kind = asr.unit_kind
    incompatible = unit_kind_problem(asr.model, kind)
    if incompatible is not None:
        return [Problem(
            "asr.unit.model_incompatible", Severity.ERROR, Subsystem.ASR,
            f"asr.unit is {asr.unit}, but {incompatible}.", _unit_fixes(config, env, env.units()),
        )]
    if kind == unit_mod.CPU:
        reason = cpu_backend_problem(env)
        if reason is None:
            return []
        return [Problem(
            "asr.unit.cpu_unsupported", Severity.ERROR, Subsystem.ASR,
            f"asr.unit is cpu, but {reason}.", _unit_fixes(config, env, env.units()),
        )]
    if kind == unit_mod.AUTO:
        return _check_auto(config, env)

    if kind == unit_mod.VULKAN:
        # whisper.cpp does not fail without a GPU, it quietly runs on the
        # CPU -- exactly the silent unit change the operator must hear about.
        problem = _vulkan_unavailable(
            env, Severity.ERROR, "whisper.cpp would quietly run on the CPU instead",
            lambda: _unit_fixes(config, env, env.units()),
        )
        if problem is not None:
            return [problem]
    elif env.cuda_devices() is None:
        return [Problem(
            "asr.unit.cuda_probe_failed", Severity.WARNING, Subsystem.ASR,
            "The NVIDIA cards could not be listed (nvidia-smi failed); the CUDA check is skipped. See the log.",
        )]
    units = env.units()
    unit = unit_mod.find_unit(units, asr.unit)
    if unit is None:
        return _missing_unit(config, env, units)
    if not unit.usable:
        twins = [u for u in unit_mod.same_card(units, unit) if u.usable]
        same = f" The same card is usable as {twins[0].id}." if twins else ""
        return [Problem(
            "asr.unit.no_backend", Severity.ERROR, Subsystem.SETUP,
            f"asr.unit is {unit.label}: the card is present, but {unit.reason}, so ASR cannot run on it.{same}",
            _unit_fixes(config, env, units, also=tuple(use_unit_fix(u.id, u.name) for u in twins)),
        )]
    return _check_identity(config, env, units, unit)


def _check_auto(config: RootConfig, env: Environment) -> list[Problem]:
    """asr.unit auto: the recommended unit is resolved when the worker
    starts. Fine on a CUDA card or a discrete Vulkan GPU; a fallback to the
    CPU or an integrated GPU is a warning saying why."""
    units = env.units()
    chosen = unit_mod.recommended_unit(units, _model_kinds(config))
    if chosen is None:
        reasons = "; ".join(f"{u.id}: {u.reason}" for u in units if u.reason) or "no unit was found"
        return [Problem(
            "asr.unit.none_usable", Severity.ERROR, Subsystem.ASR,
            f"asr.unit is auto, but no processing unit here can run model {config.asr.model!r} ({reasons}).",
            (_FIX_PICK_UNIT,),
        )]
    if chosen.kind == unit_mod.CUDA or chosen.device_type == "discrete":
        return []
    consequence = f"asr.unit auto runs on {chosen.label}"
    problem = _vulkan_unavailable(env, Severity.WARNING, consequence, lambda: (_FIX_PICK_UNIT,))
    if problem is not None:
        return [problem]
    return [Problem(
        "asr.unit.auto_fallback", Severity.WARNING if chosen.kind == unit_mod.CPU else Severity.INFO, Subsystem.ASR,
        f"No discrete GPU is usable here, so {consequence}, which is much slower than a discrete card.",
        (_FIX_PICK_UNIT,),
    )]


def check_model(config: RootConfig, env: Environment) -> list[Problem]:
    """Is the configured model installed (or at least downloadable)?"""
    model = config.asr.model
    if env.model_file(model) is not None:
        return []
    downloadable = env.downloadable_models()
    if downloadable and model not in downloadable:
        return [Problem(
            "asr.model.unknown", Severity.ERROR, Subsystem.ASR,
            f"Model {model!r} is not installed and is not a whisper.cpp model name that can be downloaded.",
            (Fix(FIX_UI_ASR, "Pick another model"),),
        )]
    size = whisper_model_size_mb(model)
    target = env.models_dir()
    free = env.free_mb(target) if target is not None else None
    if size is not None and free is not None and free < size * 1.1:
        return [Problem(
            "asr.model.no_space", Severity.ERROR, Subsystem.ASR,
            f"Model {model!r} is not installed and needs about {size:.0f} MB to download, "
            f"but only {free:.0f} MB are free.",
            (Fix(FIX_UI_ASR, "Pick another model"),),
        )]
    about = f" (about {size:.0f} MB)" if size is not None else ""
    return [Problem(
        "asr.model.not_installed", Severity.WARNING, Subsystem.ASR,
        f"Model {model!r} is not installed; the ASR worker downloads it{about} when it starts, "
        "which needs network access and delays the first captions.",
    )]


# ------------------------------------------------------------- audio / NDI


def check_audio_device(config: RootConfig, env: Environment) -> list[Problem]:
    """Is the configured microphone there, with the selected channels?
    (NDI audio sources are checked at runtime, by the capture itself.)"""
    from narrator.audio.devices import ndi_source_from_spec

    spec = config.audio.device
    if ndi_source_from_spec(spec) is not None:
        return []
    try:
        devices = env.input_devices()
    except Exception as exc:
        return [Problem(
            "audio.portaudio", Severity.ERROR, Subsystem.AUDIO,
            f"Audio devices cannot be listed ({exc}).", (_FIX_RETRY_AUDIO, _FIX_PICK_MIC),
        )]
    if not devices:
        return [Problem(
            "audio.no_devices", Severity.ERROR, Subsystem.AUDIO,
            "No audio input device was found (no microphone, or PipeWire/PulseAudio is not running).",
            (_FIX_RETRY_AUDIO, _FIX_PICK_MIC),
        )]
    if spec.strip().lower() == "default":
        return []
    needle = spec.strip().lower()
    match = next(((n, ch) for n, ch in devices if needle in n.lower()), None)
    if match is None:
        names = ", ".join(n for n, _ in devices)
        return [Problem(
            "audio.device_missing", Severity.ERROR, Subsystem.AUDIO,
            f"The audio input {spec!r} is not connected. Available: {names}.",
            (_FIX_RETRY_AUDIO, Fix(FIX_UI_AUDIO, "Pick another input")),
        )]
    name, max_channels = match
    wanted = max(config.audio.channels)
    if wanted > max_channels:
        return [Problem(
            "audio.channels", Severity.ERROR, Subsystem.AUDIO,
            f"audio.channels asks for channel {wanted}, but {name!r} has {max_channels}.",
            (Fix(FIX_UI_AUDIO, "Pick channels"),),
        )]
    return []


def check_ndi_runtime(env: Environment) -> list[Problem]:
    error = env.ndi_runtime_error()
    if error is None:
        return []
    return [Problem(
        "ndi.runtime", Severity.ERROR, Subsystem.NDI,
        f"The NDI runtime cannot be loaded ({error}): no caption video output and no NDI audio input. "
        "Install the NDI runtime (https://ndi.video) or reinstall cyndilib.",
        (_FIX_RETRY_NDI,),
    )]


# -------------------------------------------------------------------- disk


def check_disk(config: RootConfig, env: Environment) -> list[Problem]:
    problems = []
    tune_cache = env.var_dir() / "tune" / "cache"
    if tune_cache.is_dir():
        free = env.free_mb(tune_cache)
        if free is not None and free < TUNE_CACHE_MIN_FREE_MB:
            problems.append(Problem(
                "setup.tune_cache_disk_low", Severity.INFO, Subsystem.SETUP,
                f"Only {free:.0f} MB free for the narrator-tune cache ({tune_cache}); "
                "`make configure` stops caching excerpts below its min_free_disk_mb.",
            ))
    return problems


def static_problems(config: RootConfig, env: Environment, runtime: RuntimeState | None = None) -> list[Problem]:
    """Every static check; one failing check never hides the others."""
    problems: list[Problem] = []
    checks = [check_asr_unit, check_model, check_disk]
    if runtime is None or not runtime.audio_running:
        # A running capture has its device; PortAudio does not rescan while
        # it is open, so listing again would tell nothing new.
        checks.append(check_audio_device)
    for check in checks:
        try:
            problems += check(config, env)
        except Exception as exc:  # a broken probe must not take the panel down
            logger.debug("health check %s failed", check.__name__, exc_info=True)
            problems.append(Problem(
                f"setup.check_failed.{check.__name__}", Severity.INFO, Subsystem.SETUP,
                f"The {check.__name__.removeprefix('check_').replace('_', ' ')} check could not run: {exc}",
            ))
    try:
        problems += check_ndi_runtime(env)
    except Exception:
        logger.debug("NDI runtime check failed", exc_info=True)
    return problems


# ---------------------------------------------------------------- runtime


def describe_exit_code(code: int | None) -> str:
    if code is None:
        return "an unknown cause"
    if code == -4:
        return (
            "an illegal instruction (SIGILL): the ASR build uses CPU or GPU instructions this machine "
            "lacks (a pre-AVX CPU with the PyPI wheel, or asr.flash_attn on this driver)"
        )
    if code == -11:
        return "a crash in native code (SIGSEGV), usually the GPU driver or a broken ASR build"
    if code == -6:
        return "an abort in native code (SIGABRT)"
    if code == -9:
        return "being killed (SIGKILL), often the system running out of memory"
    if code < 0:
        return f"signal {-code}"
    return f"exit code {code}"


def runtime_problems(runtime: RuntimeState, config: RootConfig, env: Environment) -> list[Problem]:
    problems: list[Problem] = []
    if runtime.asr_halted:
        cause = runtime.asr_last_error or describe_exit_code(runtime.asr_last_exit_code)
        # A model that cannot load fails on every unit: offer another model,
        # not "Use CPU instead".
        model_broken = any(p.severity == Severity.ERROR for p in check_model(config, env))
        fixes = (
            (_FIX_RETRY_ASR, Fix(FIX_UI_ASR, "Pick another model"))
            if model_broken
            else _unit_fixes(config, env, retry=True)
        )
        problems.append(Problem(
            "asr.stopped", Severity.ERROR, Subsystem.ASR,
            f"ASR stopped: the worker on {config.asr.unit} with model {config.asr.model!r} failed "
            f"{runtime.asr_crash_streak} times in a row right after starting (last: {cause}). "
            "No captions from speech until this is fixed; NDI output, manual captions and this page "
            "keep working.",
            fixes,
        ))
    elif runtime.asr_crash_streak > 0 and not runtime.asr_alive:
        cause = runtime.asr_last_error or describe_exit_code(runtime.asr_last_exit_code)
        problems.append(Problem(
            "asr.restarting", Severity.WARNING, Subsystem.ASR,
            f"The ASR worker stopped ({cause}); restarting it.",
        ))

    if runtime.audio_error:
        problems.append(Problem(
            "audio.capture_failed", Severity.ERROR, Subsystem.AUDIO,
            f"Audio input is not running: {runtime.audio_error}",
            (_FIX_RETRY_AUDIO, _FIX_PICK_MIC),
        ))
    elif runtime.ndi_audio_source is not None:
        source, status = runtime.ndi_audio_source, runtime.ndi_audio_status
        if status == "runtime_unavailable":
            problems.append(Problem(
                "audio.ndi_runtime", Severity.ERROR, Subsystem.AUDIO,
                f"NDI audio input {source!r} cannot start: the NDI runtime is unavailable"
                f"{f' ({runtime.ndi_audio_detail})' if runtime.ndi_audio_detail else ''}.",
                (_FIX_RETRY_AUDIO, _FIX_PICK_MIC),
            ))
        elif status == "searching" and runtime.ndi_audio_status_age_s >= NDI_SOURCE_GRACE_SECONDS:
            problems.append(Problem(
                "audio.ndi_source_missing", Severity.WARNING, Subsystem.AUDIO,
                f"NDI audio source {source!r} is not on the network (for {runtime.ndi_audio_status_age_s:.0f} s); "
                "captions resume by themselves when it appears.",
                (_FIX_PICK_MIC,),
            ))
        elif status == "channel_error":
            problems.append(Problem(
                "audio.ndi_channels", Severity.ERROR, Subsystem.AUDIO,
                runtime.ndi_audio_detail or f"NDI source {source!r} lacks the selected audio channels.",
                (Fix(FIX_UI_AUDIO, "Pick channels"),),
            ))

    if runtime.ndi_output_error:
        problems.append(Problem(
            "ndi.output_failed", Severity.ERROR, Subsystem.NDI,
            f"NDI caption output is not running: {runtime.ndi_output_error}",
            (_FIX_RETRY_NDI, _FIX_NDI_SETTINGS),
        ))
    return problems
