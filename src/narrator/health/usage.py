"""Live usage of every processing unit (todo C6), for the web UI's unit
picker and `/api/state`: GPU utilisation, VRAM used/total, temperature and
power; per-core CPU load and RAM.

Sampled on its own thread at ~1 Hz, never on the ASR, render or event-drain
path; readers (the web UI's 10 Hz state snapshot) get the last result,
swapped atomically. Sources, each optional (missing -> "n/a", i.e. None):
  - NVIDIA: one long-lived `nvidia-smi --query-gpu=... -lms 1000` process
    (a fresh nvidia-smi every second costs real CPU on the Pentium host);
  - AMD: sysfs (`gpu_busy_percent`, `mem_info_vram_used/total`, hwmon
    temperature and power), matched to the unit by PCI bus id -- no amd-smi
    or rocm-smi needed;
  - CPU: /proc/stat (per-core load from the difference of two samples) and
    /proc/meminfo (psutil is not a dependency).
An Intel iGPU has no utilisation counter in sysfs: n/a.

The unit list itself comes from the setup checks' Environment (cached
probes; re-probed after its invalidate()).
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from narrator.asr import units as unit_mod
from narrator.asr.vulkan_devices import normalize_pci_bus_id

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 1.0
# An nvidia-smi sample older than this is n/a (the process stalled or died).
_NVIDIA_STALE_SECONDS = 5.0
# How long to wait before starting nvidia-smi again after it failed.
_NVIDIA_RETRY_SECONDS = 60.0
# nvidia-smi runs only while someone reads the usage (a web UI tab open, an
# /api/state poller); it is stopped after this long without a reader.
_IDLE_STOP_SECONDS = 30.0
_NVIDIA_FIELDS = "pci.bus_id,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
_AMD_VENDOR = "0x1002"


def empty_usage() -> dict:
    return {"util_pct": None, "mem_used_mb": None, "mem_total_mb": None, "temp_c": None, "power_w": None}


def _number(text: str) -> float | None:
    text = text.strip()
    if not text or text.startswith("[") or text.lower() in ("n/a", "not supported"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_nvidia_smi_usage(line: str) -> tuple[str, dict] | None:
    """One line of `nvidia-smi --query-gpu=<_NVIDIA_FIELDS>
    --format=csv,noheader,nounits` -> (pci bus id, usage)."""
    parts = line.split(",")
    if len(parts) < 6:
        return None
    pci = normalize_pci_bus_id(parts[0])
    if not pci or ":" not in pci:
        return None
    util, used, total, temp, power = (_number(p) for p in parts[1:6])
    return pci, {"util_pct": util, "mem_used_mb": used, "mem_total_mb": total, "temp_c": temp, "power_w": power}


def parse_proc_stat(text: str) -> dict[str, tuple[int, int]]:
    """{"cpu0": (idle, total), ...} jiffies per core from /proc/stat (the
    aggregate "cpu" line is left out)."""
    cores = {}
    for line in text.splitlines():
        if not line.startswith("cpu") or line.startswith("cpu "):
            continue
        name, *fields = line.split()
        try:
            values = [int(v) for v in fields]
        except ValueError:
            continue
        if len(values) < 4:
            continue
        idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
        cores[name] = (idle, sum(values[:8]))  # guest time is already in user/nice
    return cores


def core_loads(previous: dict[str, tuple[int, int]], current: dict[str, tuple[int, int]]) -> list[float | None]:
    """Per-core busy % between two parse_proc_stat samples, in core order."""
    loads = []
    for name in sorted(current, key=lambda n: int(n[3:]) if n[3:].isdigit() else 0):
        if name not in previous:
            loads.append(None)
            continue
        idle = current[name][0] - previous[name][0]
        total = current[name][1] - previous[name][1]
        loads.append(None if total <= 0 else round(100.0 * (total - idle) / total, 1))
    return loads


def parse_meminfo(text: str) -> tuple[float | None, float | None]:
    """(used MB, total MB) from /proc/meminfo (used = total - available)."""
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            try:
                values[key] = int(parts[0])  # kB
            except ValueError:
                pass
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None:
        return None, None
    used = None if available is None else round((total - available) / 1024)
    return used, round(total / 1024)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def read_amd_sysfs(sys_root: Path = Path("/sys")) -> dict[str, dict]:
    """{pci bus id: usage} for every AMD GPU under /sys/class/drm."""
    found: dict[str, dict] = {}
    drm = sys_root / "class" / "drm"
    try:
        cards = sorted(p for p in drm.iterdir() if p.name.startswith("card") and "-" not in p.name)
    except OSError:
        return found
    for card in cards:
        device = card / "device"
        if _read(device / "vendor") != _AMD_VENDOR:
            continue
        try:
            pci = normalize_pci_bus_id(device.resolve().name)
        except OSError:
            continue
        usage = empty_usage()
        usage["util_pct"] = _number(_read(device / "gpu_busy_percent") or "")
        used = _number(_read(device / "mem_info_vram_used") or "")
        total = _number(_read(device / "mem_info_vram_total") or "")
        usage["mem_used_mb"] = None if used is None else round(used / 2**20)
        usage["mem_total_mb"] = None if total is None else round(total / 2**20)
        try:
            hwmons = sorted((device / "hwmon").iterdir())
        except OSError:
            hwmons = []
        for hwmon in hwmons[:1]:
            temp = _number(_read(hwmon / "temp1_input") or "")
            power = _number(_read(hwmon / "power1_average") or _read(hwmon / "power1_input") or "")
            usage["temp_c"] = None if temp is None else temp / 1000.0
            usage["power_w"] = None if power is None else round(power / 1e6, 1)
        if pci:
            found[pci] = usage
    return found


class NvidiaSmiStream:
    """One `nvidia-smi ... -lms <interval>` process and the reader thread
    keeping its latest sample per card. Started on first use, restarted
    (at most every _NVIDIA_RETRY_SECONDS) when it dies; missing nvidia-smi
    simply means no NVIDIA usage."""

    def __init__(self, interval_ms: int = 1000, command: list[str] | None = None) -> None:
        self._command = command or [
            "nvidia-smi", f"--query-gpu={_NVIDIA_FIELDS}", "--format=csv,noheader,nounits", f"--loop-ms={interval_ms}",
        ]
        self._process: subprocess.Popen | None = None
        self._samples: dict[str, tuple[float, dict]] = {}
        self._next_try = 0.0
        self.error: str | None = None

    def ensure_running(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            return
        now = time.monotonic()
        if now < self._next_try:
            return
        self._next_try = now + _NVIDIA_RETRY_SECONDS
        try:
            process = subprocess.Popen(
                self._command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            return
        self._process = process
        self.error = None
        threading.Thread(target=self._read, args=(process,), name="nvidia-smi", daemon=True).start()

    def _read(self, process: subprocess.Popen) -> None:
        try:
            for line in process.stdout:
                parsed = parse_nvidia_smi_usage(line)
                if parsed is not None:
                    self._samples[parsed[0]] = (time.monotonic(), parsed[1])
        except (OSError, ValueError):
            pass

    def latest(self) -> dict[str, dict]:
        now = time.monotonic()
        return {pci: usage for pci, (t, usage) in list(self._samples.items()) if now - t <= _NVIDIA_STALE_SECONDS}

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        self._next_try = 0.0  # a deliberate stop: the next ensure_running() starts at once
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


class UnitUsageMonitor:
    def __init__(
        self,
        env,
        get_config: Callable,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        *,
        proc_root: Path = Path("/proc"),
        sys_root: Path = Path("/sys"),
        nvidia: NvidiaSmiStream | None = None,
    ) -> None:
        self._env = env
        self._get_config = get_config
        self._interval = interval
        self._proc = proc_root
        self._sys = sys_root
        self._nvidia = nvidia if nvidia is not None else NvidiaSmiStream(int(interval * 1000))
        self._prev_stat: dict[str, tuple[int, int]] = {}
        self._snapshot: dict = {"units": [], "recommended": None, "resolved": None}
        self._last_read = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="unit-usage", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._nvidia.stop()

    def snapshot(self) -> dict:
        """{"units": [unit dict + "usage", "recommended", "selected"],
        "recommended": id, "resolved": id the configured asr.unit means}."""
        self._last_read = time.monotonic()
        return self._snapshot

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:
                logger.debug("unit usage sample failed", exc_info=True)
            self._stop.wait(self._interval)

    def sample(self) -> dict:
        """One pass (the thread's body; tests call it directly)."""
        from narrator.asr.catalog import entry_for

        found = unit_mod.discover_units(self._env)
        config = self._get_config()
        kinds = entry_for(config.asr.model).units if config is not None else None
        best = unit_mod.recommended_unit(found, kinds)
        resolved = None
        if config is not None:
            try:
                unit = unit_mod.resolve_unit(config.asr.unit, found, kinds)
                resolved = unit.id if unit is not None else None
            except ValueError:
                pass
        nvidia = {}
        read_recently = time.monotonic() - self._last_read < _IDLE_STOP_SECONDS
        if not read_recently:
            self._nvidia.stop()
        elif any(u.kind == unit_mod.CUDA or "nvidia" in u.name.lower() for u in found if u.kind != unit_mod.CPU):
            self._nvidia.ensure_running()
            nvidia = self._nvidia.latest()
        amd = read_amd_sysfs(self._sys) if any(u.kind == unit_mod.VULKAN for u in found) else {}
        cpu = self._cpu_usage()
        rows = []
        for unit in found:
            row = unit.to_dict()
            if unit.kind == unit_mod.CPU:
                usage = cpu
            else:
                usage = dict(nvidia.get(unit.pci_bus_id) or amd.get(unit.pci_bus_id) or empty_usage())
                if usage["mem_total_mb"] is None and unit.vram_mb and unit.device_type == "discrete":
                    usage["mem_total_mb"] = unit.vram_mb
            row["usage"] = usage
            row["recommended"] = best is not None and unit.id == best.id
            row["selected"] = unit.id == resolved
            rows.append(row)
        snapshot = {"units": rows, "recommended": best.id if best else None, "resolved": resolved}
        self._snapshot = snapshot  # one reference swap
        return snapshot

    def _cpu_usage(self) -> dict:
        usage = empty_usage()
        stat = _read(self._proc / "stat")
        cores: list[float | None] = []
        if stat is not None:
            current = parse_proc_stat(stat)
            cores = core_loads(self._prev_stat, current)
            self._prev_stat = current
        known = [c for c in cores if c is not None]
        usage["util_pct"] = round(sum(known) / len(known), 1) if known else None
        usage["per_core_pct"] = cores
        meminfo = _read(self._proc / "meminfo")
        if meminfo is not None:
            usage["mem_used_mb"], usage["mem_total_mb"] = parse_meminfo(meminfo)
        return usage
