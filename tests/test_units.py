"""Processing units, the model catalog, live unit usage and the backend
interface (todo C6). GPU-free: every probe is faked (no Vulkan loader,
nvidia-smi or sysfs of this machine is read)."""
from __future__ import annotations

from pathlib import Path

import pytest

from narrator.asr import catalog, units
from narrator.asr.units import AsrBuild, GpuDevice, ProcessingUnit
from narrator.asr.vulkan_devices import format_pci_bus_id, normalize_pci_bus_id
from narrator.config.schema import AsrConfig
from narrator.health import usage

SOURCE_BUILD = AsrBuild(installed=True, version="1.5.1+whispercpp.1.9.5", has_vulkan=True)


class FakeProbes:
    def __init__(self, **kw) -> None:
        self.build = kw.get("build", SOURCE_BUILD)
        self.flags = kw.get("flags", {"sse4_2"})
        self.loader = kw.get("loader", True)
        self.gpus = kw.get("gpus", [
            GpuDevice(0, "NVIDIA GeForce RTX 3070", True, "0000:01:00.0", 8192),
            GpuDevice(1, "Intel(R) HD Graphics 610", False, "0000:00:02.0", 5820),
        ])
        self.cuda = kw.get("cuda", [GpuDevice(0, "NVIDIA GeForce RTX 3070", True, "0000:01:00.0", 8192)])

    def asr_build(self):
        return self.build

    def cpu_flags(self):
        return self.flags

    def cpu_name(self):
        return "Intel(R) Pentium(R) CPU G4560 @ 3.50GHz"

    def vulkan_loader_present(self):
        return self.loader

    def vulkan_devices(self):
        return self.gpus

    def cuda_devices(self):
        return self.cuda


# ---------------------------------------------------------------- discovery


def test_pci_bus_ids_are_normalized_across_tools():
    assert normalize_pci_bus_id("00000000:01:00.0") == "0000:01:00.0"  # nvidia-smi
    assert normalize_pci_bus_id("0000:03:00.0") == "0000:03:00.0"  # sysfs
    assert normalize_pci_bus_id("01:00.0") == "0000:01:00.0"
    assert normalize_pci_bus_id("") is None
    assert format_pci_bus_id(0, 1, 0, 0) == "0000:01:00.0"
    assert format_pci_bus_id(0, 0x2b, 0x1f, 3) == "0000:2b:1f.3"


def test_parse_nvidia_smi_devices():
    text = "0, NVIDIA GeForce RTX 3070, 00000000:01:00.0, 8192\n1, NVIDIA RTX A2000, 00000000:02:00.0, [N/A]\ngarbage\n"
    rtx, a2000 = units.parse_nvidia_smi_devices(text)
    assert (rtx.index, rtx.name, rtx.pci_bus_id, rtx.vram_mb) == (0, "NVIDIA GeForce RTX 3070", "0000:01:00.0", 8192)
    assert a2000.index == 1 and a2000.vram_mb is None


def test_parse_cpuinfo():
    flags, name = units.parse_cpuinfo("processor\t: 0\nmodel name\t: Intel(R) Pentium(R) CPU G4560 @ 3.50GHz\nflags\t\t: fpu sse4_2\n")
    assert flags == {"fpu", "sse4_2"} and name.startswith("Intel(R) Pentium")


def test_discovery_lists_every_unit_with_why_it_is_not_usable():
    found = units.discover_units(FakeProbes())
    assert [u.id for u in found] == ["cpu", "cuda:0", "vulkan:0", "vulkan:1"]
    cpu, cuda, rtx, igpu = found
    assert cpu.usable and cpu.name.startswith("Intel(R) Pentium")
    assert not cuda.usable and "no CUDA backend" in cuda.reason  # present, no backend installed
    assert rtx.usable and rtx.device_type == "discrete" and rtx.vram_mb == 8192
    assert igpu.device_type == "integrated"
    assert rtx.identity == "NVIDIA GeForce RTX 3070 @ 0000:01:00.0" and cpu.identity == ""
    assert units.same_card(found, cuda) == [rtx]  # one physical card, two backends


def test_discovery_without_vulkan_build_or_loader():
    pypi = AsrBuild(installed=True, version="1.5.1", has_vulkan=False)
    found = units.discover_units(FakeProbes(build=pypi))
    cpu = units.find_unit(found, "cpu")
    assert not cpu.usable and "SIGILL" in cpu.reason  # pre-AVX CPU + the PyPI wheel
    assert all(not u.usable and "no Vulkan backend" in u.reason for u in found if u.kind == "vulkan")
    assert [u.id for u in units.discover_units(FakeProbes(loader=False, cuda=[]))] == ["cpu"]
    assert [u.id for u in units.discover_units(FakeProbes(gpus=None, cuda=None))] == ["cpu"]  # failed probes


def test_recommendation_order_cuda_then_dgpu_then_cpu_never_igpu_first():
    found = units.discover_units(FakeProbes())
    assert units.recommended_unit(found).id == "vulkan:0"  # cuda:0 has no backend
    cuda_build = AsrBuild(True, "1.5.1+whispercpp.1.9.5", has_vulkan=True, has_cuda=True)
    assert units.recommended_unit(units.discover_units(FakeProbes(build=cuda_build))).id == "cuda:0"
    igpu_only = FakeProbes(gpus=[GpuDevice(0, "Intel(R) HD Graphics 610", False)], cuda=[])
    assert units.recommended_unit(units.discover_units(igpu_only)).id == "cpu"
    no_cpu = FakeProbes(gpus=[GpuDevice(0, "Intel(R) HD Graphics 610", False)], cuda=[],
                        build=AsrBuild(True, "1.5.1", has_vulkan=True))
    assert units.recommended_unit(units.discover_units(no_cpu)).id == "vulkan:0"  # the only one left
    # restricted to the kinds a model runs on (an ONNX model: no Vulkan)
    assert units.recommended_unit(found, ("cpu", "cuda")).id == "cpu"
    assert units.resolve_unit("auto", found).id == "vulkan:0"
    assert units.resolve_unit("vulkan:1", found).id == "vulkan:1"
    assert units.resolve_unit("vulkan:5", found) is None


def test_identity_matching():
    rtx = ProcessingUnit("vulkan:0", "vulkan", 0, "NVIDIA GeForce RTX 3070", "discrete", "0000:01:00.0")
    assert rtx.matches_identity("NVIDIA GeForce RTX 3070 @ 0000:01:00.0")
    assert rtx.matches_identity("nvidia geforce rtx 3070 @ 00000000:01:00.0")  # case, nvidia-smi spelling
    assert rtx.matches_identity("NVIDIA GeForce RTX 3070")  # recorded without a PCI id
    assert not rtx.matches_identity("NVIDIA GeForce RTX 3070 @ 0000:02:00.0")  # another slot
    assert not rtx.matches_identity("Intel(R) HD Graphics 610 @ 0000:01:00.0")
    assert units.split_identity("A @ B @ 0000:01:00.0") == ("A @ B", "0000:01:00.0")


@pytest.mark.parametrize(
    "device,gpu,unit",
    [("cpu", "auto", "cpu"), ("vulkan", "auto", "auto"), ("auto", 1, "vulkan:1"), (None, None, "auto"), ("vulkan", "0", "vulkan:0")],
)
def test_unit_from_legacy(device, gpu, unit):
    assert units.unit_from_legacy(device, gpu) == unit


def test_unit_parsing_and_cache_keys():
    assert units.parse_unit("Vulkan:2") == ("vulkan", 2)
    assert units.parse_unit("auto") == ("auto", None)
    with pytest.raises(ValueError):
        units.parse_unit("gpu0")
    assert units.unit_or_legacy("vulkan") == "auto"  # an asr.device value from an older payload
    assert units.unit_or_legacy("cuda:1") == "cuda:1"
    # narrator-tune's cached benches were keyed by asr.device ("vulkan"/"cpu"/"auto")
    assert [units.cache_device_key(u) for u in ("vulkan:1", "cpu", "auto", "vulkan")] == ["vulkan", "cpu", "auto", "auto"]


# ------------------------------------------------------------------ catalog


def test_catalog_entries_are_valid():
    entries = catalog.catalog()
    assert entries, "the catalog lists whisper.cpp's models"
    names = [e.model for e in entries]
    assert len(names) == len(set(names))
    assert {"tiny", "small", "medium", "large-v3"} <= set(names)
    for entry in entries:
        assert entry.units and set(entry.units) <= set(units.UNIT_KINDS)
        assert set(entry.units) <= set(catalog.BACKEND_UNITS[entry.backend])
        assert entry.files
    small = catalog.entry_for("small")
    assert small.backend == catalog.AsrBackendKind.WHISPER_CPP and small.units == ("cpu", "cuda", "vulkan")
    assert small.files == ("ggml-small.bin",) and small.size_mb == 488
    assert catalog.entry_for("/models/custom.bin").backend == catalog.AsrBackendKind.WHISPER_CPP
    assert catalog.whisper_model_size_mb("large-v3-q5_0") == 1550
    # the unit rules of the backends to come (todo C6)
    assert catalog.BACKEND_UNITS[catalog.AsrBackendKind.ONNX] == ("cpu", "cuda")
    assert catalog.BACKEND_UNITS[catalog.AsrBackendKind.NEMO] == ("cuda",)


def test_catalog_installed_comes_from_narrator_paths(tmp_path, monkeypatch):
    (tmp_path / "ggml-small.bin").write_bytes(b"x")
    monkeypatch.setattr("narrator.paths.model_search_dirs", lambda: [tmp_path])
    rows = {r["model"]: r for r in catalog.catalog_rows("my-model")}
    assert rows["small"]["installed"] is True and rows["medium"]["installed"] is False
    assert "my-model" in rows and rows["my-model"]["backend"] == "whisper.cpp"


def test_catalog_unit_problem(monkeypatch):
    assert catalog.unit_kind_problem("large-v3", "vulkan") is None
    monkeypatch.setattr(catalog, "EXTRA_ENTRIES", [catalog.CatalogEntry("parakeet", catalog.AsrBackendKind.ONNX, ("p.onnx",))])
    assert "not on vulkan" in catalog.unit_kind_problem("parakeet", "vulkan")
    assert catalog.unit_kind_problem("parakeet", "auto") is None
    assert "parakeet" in [e.model for e in catalog.catalog()]


# -------------------------------------------------------------------- usage


def test_parse_nvidia_smi_usage():
    pci, row = usage.parse_nvidia_smi_usage("00000000:01:00.0, 37, 1423, 8192, 55, 41.27\n")
    assert pci == "0000:01:00.0"
    assert row == {"util_pct": 37.0, "mem_used_mb": 1423.0, "mem_total_mb": 8192.0, "temp_c": 55.0, "power_w": 41.27}
    _, row = usage.parse_nvidia_smi_usage("00000000:01:00.0, [N/A], 100, 4096, [Not Supported], [N/A]")
    assert row["util_pct"] is None and row["temp_c"] is None and row["power_w"] is None
    assert usage.parse_nvidia_smi_usage("No devices were found") is None


def test_cpu_core_loads_and_meminfo():
    first = usage.parse_proc_stat("cpu  10 0 10 80 0 0 0 0\ncpu0 5 0 5 40 0 0 0 0\ncpu1 5 0 5 40 0 0 0 0\nintr 1\n")
    second = usage.parse_proc_stat("cpu  0 0 0 0\ncpu0 15 0 15 60 0 0 0 0\ncpu1 5 0 5 140 0 0 0 0\n")
    assert usage.core_loads(first, second) == [50.0, 0.0]
    assert usage.core_loads({}, second) == [None, None]
    used, total = usage.parse_meminfo("MemTotal:  8000000 kB\nMemFree: 1 kB\nMemAvailable:  6000000 kB\n")
    assert (used, total) == (1953, 7812)
    assert usage.parse_meminfo("nothing") == (None, None)


def _amd_tree(root: Path) -> None:
    device = root / "devices" / "0000:03:00.0"
    (device / "hwmon" / "hwmon4").mkdir(parents=True)
    for name, value in {
        "vendor": "0x1002", "gpu_busy_percent": "64", "mem_info_vram_used": str(3 * 2**30),
        "mem_info_vram_total": str(12 * 2**30),
    }.items():
        (device / name).write_text(value + "\n")
    (device / "hwmon" / "hwmon4" / "temp1_input").write_text("61000\n")
    (device / "hwmon" / "hwmon4" / "power1_average").write_text("123400000\n")
    intel = root / "devices" / "0000:00:02.0"
    intel.mkdir(parents=True)
    (intel / "vendor").write_text("0x8086\n")
    drm = root / "class" / "drm"
    drm.mkdir(parents=True)
    (drm / "card0").mkdir()
    (drm / "card0" / "device").symlink_to(device)
    (drm / "card1").mkdir()
    (drm / "card1" / "device").symlink_to(intel)
    (drm / "card0-DP-1").mkdir()


def test_amd_sysfs_usage(tmp_path):
    _amd_tree(tmp_path)
    found = usage.read_amd_sysfs(tmp_path)
    assert list(found) == ["0000:03:00.0"]  # the Intel card has no counters
    row = found["0000:03:00.0"]
    assert row == {"util_pct": 64.0, "mem_used_mb": 3072, "mem_total_mb": 12288, "temp_c": 61.0, "power_w": 123.4}
    assert usage.read_amd_sysfs(tmp_path / "missing") == {}


class FakeNvidia:
    def __init__(self, samples):
        self.samples, self.started, self.stopped = samples, 0, 0

    def ensure_running(self):
        self.started += 1

    def latest(self):
        return dict(self.samples)

    def stop(self):
        self.stopped += 1


def test_usage_monitor_sample(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text("cpu0 1 0 1 8 0 0 0 0\ncpu1 1 0 1 8 0 0 0 0\n")
    (proc / "meminfo").write_text("MemTotal: 4096000 kB\nMemAvailable: 2048000 kB\n")
    nvidia = FakeNvidia({"0000:01:00.0": {"util_pct": 12.0, "mem_used_mb": 900.0, "mem_total_mb": 8192.0,
                                          "temp_c": 50.0, "power_w": 30.0}})
    config = AsrConfig(unit="auto")
    root = type("Cfg", (), {"asr": config})()
    monitor = usage.UnitUsageMonitor(FakeProbes(), lambda: root, proc_root=proc, sys_root=tmp_path / "sys", nvidia=nvidia)
    monitor.snapshot()  # a reader: nvidia-smi is wanted
    snap = monitor.sample()
    rows = {r["id"]: r for r in snap["units"]}
    assert snap["recommended"] == "vulkan:0" and snap["resolved"] == "vulkan:0"
    assert rows["vulkan:0"]["selected"] and rows["vulkan:0"]["recommended"]
    # one card, two units: both show the card's nvidia-smi numbers
    assert rows["vulkan:0"]["usage"]["util_pct"] == 12.0 and rows["cuda:0"]["usage"]["power_w"] == 30.0
    assert rows["vulkan:1"]["usage"]["util_pct"] is None  # no counters for the iGPU: n/a
    assert rows["vulkan:1"]["usage"]["mem_total_mb"] is None  # its "VRAM" is shared system RAM
    assert rows["cpu"]["usage"]["mem_total_mb"] == 4000 and rows["cpu"]["usage"]["per_core_pct"] == [None, None]
    (proc / "stat").write_text("cpu0 2 0 2 16 0 0 0 0\ncpu1 11 0 1 8 0 0 0 0\n")
    cpu = {r["id"]: r for r in monitor.sample()["units"]}["cpu"]["usage"]
    assert cpu["per_core_pct"] == [20.0, 100.0] and cpu["util_pct"] == 60.0
    assert nvidia.started == 2

    # nobody reading: nvidia-smi is stopped, GPU usage is n/a
    monitor._last_read = 0.0
    rows = {r["id"]: r for r in monitor.sample()["units"]}
    assert nvidia.stopped and rows["vulkan:0"]["usage"]["util_pct"] is None
    assert rows["vulkan:0"]["usage"]["mem_total_mb"] == 8192  # the probe's VRAM size stays


def test_usage_monitor_resolves_an_explicit_unit():
    config = type("Cfg", (), {"asr": AsrConfig(unit="vulkan:1")})()
    monitor = usage.UnitUsageMonitor(FakeProbes(), lambda: config, nvidia=FakeNvidia({}))
    snap = monitor.sample()
    assert snap["resolved"] == "vulkan:1"
    assert [r["id"] for r in snap["units"] if r["selected"]] == ["vulkan:1"]


# --------------------------------------------------------------- interface


def test_whisper_backend_implements_the_interface_and_factory_dispatches(monkeypatch):
    from narrator.asr import asr_backend
    from narrator.asr.backend import WhisperBackend

    assert asr_backend.AsrBackend in WhisperBackend.__mro__
    for name in ("transcribe", "decode_segment", "detect_language", "set_vocabulary", "close"):
        assert callable(getattr(WhisperBackend, name))
    created = []
    monkeypatch.setattr("narrator.asr.backend.WhisperBackend", lambda config: created.append(config) or "whisper")
    assert asr_backend.create_backend(AsrConfig(model="small")) == "whisper"
    monkeypatch.setattr(catalog, "EXTRA_ENTRIES", [catalog.CatalogEntry("parakeet", catalog.AsrBackendKind.ONNX, ("p.onnx",))])
    with pytest.raises(asr_backend.ModelLoadError, match="onnx"):
        asr_backend.create_backend(AsrConfig(model="parakeet", unit="cpu"))


def test_backend_refuses_a_shifted_gpu_index(monkeypatch):
    """The worker side of the identity check: a recorded card that is no
    longer at the configured index is refused (GpuSelectionError), never
    loaded on whatever card is there now."""
    from narrator.asr import backend
    from narrator.asr.vulkan_devices import VulkanDevice

    monkeypatch.setattr(backend, "_asr_build", lambda: SOURCE_BUILD)
    monkeypatch.setattr(backend, "_vulkan_devices", lambda: [
        VulkanDevice(0, "Intel(R) HD Graphics 610", 1, "0000:00:02.0"),
        VulkanDevice(1, "NVIDIA GeForce RTX 3070", 2, "0000:01:00.0"),
    ])
    loads = []
    monkeypatch.setattr(backend.WhisperBackend, "_load_model", lambda self, use_gpu: loads.append(use_gpu) or object())
    config = AsrConfig(model="small", unit="vulkan:0", unit_identity="NVIDIA GeForce RTX 3070 @ 0000:01:00.0")
    with pytest.raises(backend.GpuSelectionError, match="recorded as"):
        backend.WhisperBackend(config)
    assert loads == []
    ok = backend.WhisperBackend(config.model_copy(update={"unit": "vulkan:1"}))
    assert ok.resolved_unit == "vulkan:1" and ok._gpu_index == 1 and loads == [True]
    with pytest.raises(backend.GpuSelectionError, match="vulkan:4"):
        backend.WhisperBackend(AsrConfig(model="small", unit="vulkan:4"))
    with pytest.raises(backend.DevicePolicyError, match="CUDA build"):
        backend.WhisperBackend(AsrConfig(model="small", unit="cuda:0"))
    cpu = backend.WhisperBackend(AsrConfig(model="small", unit="cpu"))
    assert cpu.resolved_unit == "cpu" and loads[-1] is False


CUDA_BUILD = AsrBuild(installed=True, version="1.5.1+whispercpp.1.9.5", has_cuda=True)


def test_backend_loads_on_a_cuda_card(monkeypatch):
    """cuda:<n> with a CUDA build: gpu_device = nvidia-smi's index, device
    order pinned to PCI; an absent index or a shifted identity is refused;
    a Vulkan unit on a CUDA-only build is refused (it would land on another
    device), never loaded."""
    from narrator.asr import backend

    monkeypatch.setattr(backend, "_asr_build", lambda: CUDA_BUILD)
    monkeypatch.setattr(backend, "_cuda_devices", lambda: [
        GpuDevice(0, "NVIDIA GeForce RTX 3070", True, "0000:01:00.0", 8192),
    ])
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    loads = []
    monkeypatch.setattr(backend.WhisperBackend, "_load_model", lambda self, use_gpu: loads.append(use_gpu) or object())

    ok = backend.WhisperBackend(AsrConfig(model="large-v3", unit="cuda:0",
                                          unit_identity="NVIDIA GeForce RTX 3070 @ 0000:01:00.0"))
    assert ok.resolved_unit == "cuda:0" and ok._gpu_index == 0 and loads == [True]
    assert "CUDA" in ok.resolved_device_description
    import os
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    with pytest.raises(backend.GpuSelectionError, match="cuda:1"):
        backend.WhisperBackend(AsrConfig(model="large-v3", unit="cuda:1"))
    with pytest.raises(backend.GpuSelectionError, match="recorded as"):
        backend.WhisperBackend(AsrConfig(model="large-v3", unit="cuda:0",
                                         unit_identity="NVIDIA GeForce RTX 4090 @ 0000:02:00.0"))
    with pytest.raises(backend.DevicePolicyError, match="Vulkan build"):
        backend.WhisperBackend(AsrConfig(model="large-v3", unit="vulkan:0"))
    assert loads == [True]
