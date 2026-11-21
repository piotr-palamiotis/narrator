"""Setup checks + problems panel (narrator.health, todo C8) and the failure
policy: nothing stops the app, a fix restarts only the affected part.
No GPU, microphone or NDI runtime needed -- every probe is faked."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from narrator.app_controller import AppController
from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.config.loader import load_config, save_config
from narrator.config.schema import RootConfig
from narrator.health import checks
from narrator.health.checks import AsrBuild, GpuDevice, RuntimeState
from narrator.health.monitor import HealthMonitor
from narrator.health.problems import Severity, sort_problems

RTX = GpuDevice(0, "NVIDIA GeForce RTX 3070", True, "0000:01:00.0", 8192)
IGPU = GpuDevice(1, "Intel(R) HD Graphics 610", False, "0000:00:02.0", 5820)
RTX_CUDA = GpuDevice(0, "NVIDIA GeForce RTX 3070", True, "0000:01:00.0", 8192)
RTX_ID = "NVIDIA GeForce RTX 3070 @ 0000:01:00.0"
SOURCE_BUILD = AsrBuild(installed=True, version="1.5.1+whispercpp.1.9.5", has_vulkan=True)
PYPI_WHEEL = AsrBuild(installed=True, version="1.5.1", has_vulkan=False)


class FakeEnv(checks.Environment):
    """Every probe as a plain attribute."""

    def __init__(self, tmp: Path | None = None, **kw) -> None:
        super().__init__()
        self.build = kw.get("build", SOURCE_BUILD)
        self.loader = kw.get("loader", True)
        self.gpus = kw.get("gpus", [RTX, IGPU])
        self.cuda = kw.get("cuda", [])
        self.flags = kw.get("flags", {"sse4_2"})
        self.models = kw.get("models", {"small", "large-v3"})
        self.downloadable = kw.get("downloadable", {"tiny", "small", "medium", "large-v3"})
        self.inputs = kw.get("inputs", [("USB Mic", 2), ("default", 2)])
        self.ndi_error = kw.get("ndi_error")
        self.free = kw.get("free", 50_000.0)
        self.tmp = tmp or Path("/nonexistent-narrator-test")
        self.invalidations = 0

    def invalidate(self):
        self.invalidations += 1

    def vulkan_loader_present(self):
        return self.loader

    def vulkan_devices(self):
        return None if self.gpus is None else list(self.gpus)

    def cuda_devices(self):
        return None if self.cuda is None else list(self.cuda)

    def cpu_flags(self):
        return self.flags

    def cpu_name(self):
        return "Test CPU"

    def asr_build(self):
        return self.build

    def model_file(self, model):
        return Path(f"/models/ggml-{model}.bin") if model in self.models else None

    def downloadable_models(self):
        return set(self.downloadable)

    def models_dir(self):
        return self.tmp / "models"

    def input_devices(self):
        if isinstance(self.inputs, Exception):
            raise self.inputs
        return list(self.inputs)

    def ndi_runtime_error(self):
        return self.ndi_error

    def free_mb(self, path):
        return self.free

    def var_dir(self):
        return self.tmp / "var"



def cfg(**sections) -> RootConfig:
    return RootConfig.model_validate(sections)


def ids(problems) -> list[str]:
    return [p.id for p in problems]


def fix_actions(problem) -> list[str]:
    return [f.action for f in problem.fixes]


# ------------------------------------------------------------- ASR unit


def unit_cfg(unit, identity="", **asr):
    return cfg(asr={"unit": unit, "unit_identity": identity, **asr})


def test_gpu_present_is_fine():
    assert checks.check_asr_unit(unit_cfg("auto"), FakeEnv()) == []
    assert checks.check_asr_unit(unit_cfg("vulkan:0", RTX_ID), FakeEnv()) == []


def test_vulkan_without_gpu_is_an_error_with_cpu_fix():
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:0"), FakeEnv(gpus=[]))
    assert problem.id == "asr.unit.no_gpu" and problem.severity == Severity.ERROR
    assert "quietly run on the CPU" in problem.message
    assert fix_actions(problem) == ["asr.use_cpu", "ui:asr"]


def test_auto_without_gpu_is_a_warning():
    [problem] = checks.check_asr_unit(unit_cfg("auto"), FakeEnv(gpus=[]))
    assert problem.severity == Severity.WARNING and "auto runs on cpu" in problem.message


def test_auto_with_only_an_igpu_prefers_the_cpu_and_says_so():
    [problem] = checks.check_asr_unit(unit_cfg("auto"), FakeEnv(gpus=[IGPU]))
    assert problem.id == "asr.unit.auto_fallback" and "cpu" in problem.message
    # the iGPU only when nothing else can run the model (pre-AVX + PyPI CPU path)
    env = FakeEnv(gpus=[GpuDevice(0, "Intel(R) HD Graphics 610", False)], build=AsrBuild(True, "1.5.1", has_vulkan=True))
    [problem] = checks.check_asr_unit(unit_cfg("auto"), env)
    assert problem.id == "asr.unit.auto_fallback" and problem.severity == Severity.INFO
    assert "vulkan:0" in problem.message


def test_auto_with_nothing_usable():
    env = FakeEnv(gpus=[], build=AsrBuild(True, "1.5.1", has_vulkan=False))
    [problem] = checks.check_asr_unit(unit_cfg("auto"), env)
    assert problem.id == "asr.unit.none_usable" and problem.severity == Severity.ERROR
    assert "SIGILL" in problem.message


def test_missing_vulkan_loader():
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:0"), FakeEnv(loader=False))
    assert problem.id == "asr.unit.no_vulkan_loader"
    assert "libvulkan" in problem.message


def test_pypi_wheel_has_no_vulkan_and_no_cpu_fix_on_pre_avx():
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:0"), FakeEnv(build=PYPI_WHEEL))
    assert problem.id == "asr.unit.no_vulkan_build"
    # pre-AVX CPU + PyPI wheel: CPU would SIGILL, so "Use CPU" is not offered
    assert "asr.use_cpu" not in fix_actions(problem)
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:0"), FakeEnv(build=PYPI_WHEEL, flags={"avx", "avx2"}))
    assert "asr.use_cpu" in fix_actions(problem)


def test_cpu_backend_instruction_support():
    env = FakeEnv(build=PYPI_WHEEL, flags={"sse4_2"})
    [problem] = checks.check_asr_unit(unit_cfg("cpu"), env)
    assert problem.id == "asr.unit.cpu_unsupported"
    assert "SIGILL" in problem.message and "make asr-backend" in problem.message
    assert checks.check_asr_unit(unit_cfg("cpu"), FakeEnv(build=PYPI_WHEEL, flags={"avx"})) == []
    # the source build is -march=native: fine on its own (pre-AVX) host
    assert checks.check_asr_unit(unit_cfg("cpu"), FakeEnv(build=SOURCE_BUILD)) == []
    # unknown flags (not Linux): no claim either way
    assert checks.check_asr_unit(unit_cfg("cpu"), FakeEnv(build=PYPI_WHEEL, flags=None)) == []


def test_asr_package_missing():
    [problem] = checks.check_asr_unit(cfg(), FakeEnv(build=AsrBuild(installed=False)))
    assert problem.id == "asr.build.missing" and problem.severity == Severity.ERROR


def test_gpu_probe_failure_is_not_reported_as_no_gpu():
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:0"), FakeEnv(gpus=None))
    assert problem.id == "asr.unit.probe_failed" and problem.severity == Severity.WARNING


def test_unit_not_present_offers_the_recorded_card_at_its_new_index():
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:3"), FakeEnv())
    assert problem.id == "asr.unit.missing" and "vulkan:0=NVIDIA" in problem.message
    assert fix_actions(problem) == ["asr.unit:vulkan:0", "asr.use_cpu", "ui:asr"]  # the recommended unit
    # the recorded card moved from index 3 to index 1: say so, offer it first
    moved = FakeEnv(gpus=[GpuDevice(0, IGPU.name, False, IGPU.pci_bus_id), GpuDevice(1, RTX.name, True, RTX.pci_bus_id)])
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:3", RTX_ID), moved)
    assert "is now vulkan:1" in problem.message
    assert fix_actions(problem)[0] == "asr.unit:vulkan:1"


def test_identity_mismatch_is_an_error_never_a_remap():
    """vulkan:0 was the RTX when picked; a card added in front shifts the
    indices. The check reports it and offers both readings; nothing is
    re-mapped by itself."""
    shifted = FakeEnv(gpus=[GpuDevice(0, IGPU.name, False, IGPU.pci_bus_id), GpuDevice(1, RTX.name, True, RTX.pci_bus_id)])
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:0", RTX_ID), shifted)
    assert problem.id == "asr.unit.identity_mismatch" and problem.severity == Severity.ERROR
    assert "now vulkan:1" in problem.message
    assert fix_actions(problem) == ["asr.unit:vulkan:1", "asr.pin_unit", "asr.use_cpu", "ui:asr"]
    # the same card in another PCI slot is a different identity on purpose
    moved_slot = FakeEnv(gpus=[GpuDevice(0, RTX.name, True, "0000:02:00.0")])
    assert ids(checks.check_asr_unit(unit_cfg("vulkan:0", RTX_ID), moved_slot)) == ["asr.unit.identity_mismatch"]
    # no PCI id on one side: the name decides
    no_pci = FakeEnv(gpus=[GpuDevice(0, RTX.name, True)])
    assert checks.check_asr_unit(unit_cfg("vulkan:0", RTX_ID), no_pci) == []


def test_unrecorded_identity_and_igpu_are_info():
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:0"), FakeEnv())
    assert problem.id == "asr.unit.identity_unrecorded" and problem.severity == Severity.INFO
    assert fix_actions(problem) == ["asr.pin_unit"]
    [problem] = checks.check_asr_unit(unit_cfg("vulkan:1", "Intel(R) HD Graphics 610 @ 0000:00:02.0"), FakeEnv())
    assert problem.id == "asr.unit.igpu" and problem.severity == Severity.INFO


def test_cuda_unit_without_a_cuda_build_offers_the_same_card_through_vulkan():
    env = FakeEnv(cuda=[RTX_CUDA])
    [problem] = checks.check_asr_unit(unit_cfg("cuda:0", RTX_ID), env)
    assert problem.id == "asr.unit.no_backend" and problem.severity == Severity.ERROR
    assert "no CUDA backend" in problem.message and "usable as vulkan:0" in problem.message
    assert fix_actions(problem) == ["asr.unit:vulkan:0", "asr.use_cpu", "ui:asr"]
    # with a CUDA build the unit is fine
    cuda_build = AsrBuild(True, "1.5.1+whispercpp.1.9.5", has_vulkan=True, has_cuda=True)
    assert checks.check_asr_unit(unit_cfg("cuda:0", RTX_ID), FakeEnv(cuda=[RTX_CUDA], build=cuda_build)) == []
    # no NVIDIA card at all / nvidia-smi failing
    assert ids(checks.check_asr_unit(unit_cfg("cuda:0"), FakeEnv())) == ["asr.unit.missing"]
    assert ids(checks.check_asr_unit(unit_cfg("cuda:0"), FakeEnv(cuda=None))) == ["asr.unit.cuda_probe_failed"]


def test_model_unit_incompatible(monkeypatch):
    from narrator.asr import catalog

    config = unit_cfg("vulkan:0", RTX_ID)
    monkeypatch.setattr(catalog, "EXTRA_ENTRIES", [catalog.CatalogEntry("large-v3", catalog.AsrBackendKind.ONNX, ("x.onnx",))])
    [problem] = checks.check_asr_unit(config, FakeEnv())
    assert problem.id == "asr.unit.model_incompatible" and "not on vulkan" in problem.message


# ------------------------------------------------------------------ model


def test_model_installed_downloadable_unknown_and_no_space():
    assert checks.check_model(cfg(asr={"model": "small"}), FakeEnv()) == []
    [problem] = checks.check_model(cfg(asr={"model": "medium"}), FakeEnv())
    assert problem.id == "asr.model.not_installed" and problem.severity == Severity.WARNING
    assert "downloads it" in problem.message
    [problem] = checks.check_model(cfg(asr={"model": "large-v9"}), FakeEnv())
    assert problem.id == "asr.model.unknown" and problem.severity == Severity.ERROR
    assert fix_actions(problem) == ["ui:asr"]
    [problem] = checks.check_model(cfg(asr={"model": "medium"}), FakeEnv(free=800.0))
    assert problem.id == "asr.model.no_space" and "800 MB" in problem.message


# -------------------------------------------------------------- audio / NDI


def test_audio_device_checks():
    assert checks.check_audio_device(cfg(audio={"device": "usb mic"}), FakeEnv()) == []
    assert checks.check_audio_device(cfg(audio={"device": "default"}), FakeEnv()) == []
    [problem] = checks.check_audio_device(cfg(audio={"device": "Zoom H6"}), FakeEnv())
    assert problem.id == "audio.device_missing" and "USB Mic" in problem.message
    assert fix_actions(problem) == ["audio.retry", "ui:audio"]
    [problem] = checks.check_audio_device(cfg(audio={"device": "USB", "channels": [1, 4]}), FakeEnv())
    assert problem.id == "audio.channels"
    [problem] = checks.check_audio_device(cfg(), FakeEnv(inputs=[]))
    assert problem.id == "audio.no_devices"
    [problem] = checks.check_audio_device(cfg(), FakeEnv(inputs=OSError("PortAudio not initialized")))
    assert problem.id == "audio.portaudio"
    # NDI audio sources are checked at runtime, not by listing PortAudio devices
    assert checks.check_audio_device(cfg(audio={"device": "ndi:MIXER (Program)"}), FakeEnv(inputs=[])) == []


def test_ndi_runtime_check():
    assert checks.check_ndi_runtime(FakeEnv()) == []
    [problem] = checks.check_ndi_runtime(FakeEnv(ndi_error="OSError: libndi.so.6: cannot open"))
    assert problem.id == "ndi.runtime" and fix_actions(problem) == ["ndi.retry"]


# -------------------------------------------------------------------- disk


def test_disk_checks(tmp_path):
    config = cfg()
    assert checks.check_disk(config, FakeEnv(tmp_path, free=500.0)) == []  # no tune cache: nothing to warn about
    (tmp_path / "var" / "tune" / "cache").mkdir(parents=True)
    assert checks.check_disk(config, FakeEnv(tmp_path)) == []
    assert ids(checks.check_disk(config, FakeEnv(tmp_path, free=500.0))) == ["setup.tune_cache_disk_low"]


def test_static_problems_survive_a_broken_probe():
    class Broken(FakeEnv):
        def vulkan_devices(self):
            raise RuntimeError("driver exploded")

    found = checks.static_problems(cfg(asr={"model": "large-v9"}), Broken())
    assert "asr.model.unknown" in ids(found)  # the other checks still ran
    assert any(p.id.startswith("setup.check_failed.check_asr_unit") for p in found)


# ----------------------------------------------------------------- runtime


def test_runtime_problems():
    config = cfg(asr={"unit": "vulkan:0"})
    env = FakeEnv()
    [problem] = checks.runtime_problems(
        RuntimeState(asr_halted=True, asr_crash_streak=3, asr_last_exit_code=-4), config, env
    )
    assert problem.id == "asr.stopped" and "SIGILL" in problem.message
    assert fix_actions(problem) == ["asr.retry", "asr.use_cpu", "ui:asr"]
    [problem] = checks.runtime_problems(  # the model is the cause: another model, not another unit
        RuntimeState(asr_halted=True, asr_crash_streak=3), cfg(asr={"model": "large-v9"}), env
    )
    assert fix_actions(problem) == ["asr.retry", "ui:asr"]
    [problem] = checks.runtime_problems(
        RuntimeState(asr_halted=True, asr_crash_streak=3, asr_last_error="GPU backend init failed"), config, env
    )
    assert "GPU backend init failed" in problem.message
    assert ids(checks.runtime_problems(RuntimeState(asr_crash_streak=1), config, env)) == ["asr.restarting"]
    assert checks.runtime_problems(RuntimeState(asr_alive=True), config, env) == []

    found = checks.runtime_problems(
        RuntimeState(asr_alive=True, audio_running=False, audio_error="No input device matching 'x'",
                     ndi_output_error="libndi missing"),
        config, env,
    )
    assert ids(found) == ["audio.capture_failed", "ndi.output_failed"]

    def ndi(status, age=10.0, detail=None):
        return checks.runtime_problems(
            RuntimeState(asr_alive=True, ndi_audio_source="MIXER", ndi_audio_status=status,
                         ndi_audio_status_age_s=age, ndi_audio_detail=detail),
            config, env,
        )

    assert ids(ndi("searching")) == ["audio.ndi_source_missing"]
    assert ndi("searching", age=1.0) == []  # discovery grace
    assert ndi("receiving") == []
    assert ids(ndi("runtime_unavailable")) == ["audio.ndi_runtime"]
    assert ids(ndi("channel_error", detail="NDI source 'MIXER' has 2 channels")) == ["audio.ndi_channels"]


def test_sort_and_dedupe():
    env = FakeEnv(gpus=[])
    found = sort_problems(
        checks.static_problems(cfg(asr={"model": "medium"}), env)
        + checks.runtime_problems(RuntimeState(audio_running=False, audio_error="x"), cfg(), env)
    )
    severities = [p.severity for p in found]
    assert severities == sorted(severities, key=["error", "warning", "info"].index)
    assert len(ids(found)) == len(set(ids(found)))


def test_monitor_runs_on_its_own_thread_and_rechecks_on_request():
    holder = {"config": cfg(asr={"model": "small"})}
    env = FakeEnv()
    monitor = HealthMonitor(lambda: holder["config"], lambda: RuntimeState(asr_alive=True), env=env, interval=60)
    monitor.start()
    try:
        deadline = time.monotonic() + 5
        while monitor.checks_run < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert monitor.problems() == []
        holder["config"] = cfg(asr={"model": "large-v9"})  # a config change
        monitor.request_check(invalidate=True)
        while "asr.model.unknown" not in ids(monitor.problems()) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "asr.model.unknown" in ids(monitor.problems())
        assert env.invalidations >= 1
    finally:
        monitor.stop()


def test_real_environment_probes_do_not_raise():
    """The real probes on whatever machine runs the tests (no assertions on
    the hardware itself, only that a check pass completes)."""
    found = checks.static_problems(RootConfig(), checks.Environment())
    assert all(p.message for p in found)


# ------------------------------------------- the app stays up, fixes restart one part


class DeadProcess:
    """An ASR worker that died right after spawn (e.g. SIGILL)."""

    def __init__(self, exitcode=-4):
        self.exitcode = exitcode

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass

    def terminate(self):
        pass

    def kill(self):
        pass


class LiveProcess(DeadProcess):
    def __init__(self):
        super().__init__(None)
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.alive = False
        self.exitcode = 0


class FakeSender:
    created = 0

    def __init__(self, ndi, canvas):
        FakeSender.created += 1

    def send_frame(self, data):
        pass

    def close(self):
        pass


class FakeMic:
    def __init__(self):
        self.level = 0.1
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def frames(self, timeout=1.0, stop_event=None):
        while stop_event is None or not stop_event.is_set():
            time.sleep(0.02)
            if False:
                yield None


@pytest.fixture
def broken_app(tmp_path, monkeypatch, request):
    """A real AppController.start() on a machine with nothing working: the
    NDI runtime missing, the configured mic missing, the model unknown, no
    GPU, and an ASR worker that dies on every spawn."""
    config_path = tmp_path / "config.yaml"
    save_config(
        RootConfig.model_validate({
            "audio": {"device": "Zoom H6"},
            "asr": {"preset": "custom", "model": getattr(request, "param", "large-v9"), "unit": "vulkan:0"},
            "web": {"enabled": False},
        }),
        config_path,
    )
    env = FakeEnv(tmp_path, gpus=[], inputs=[("USB Mic", 2)])
    monkeypatch.setattr("narrator.health.monitor.Environment", lambda: env)

    def no_ndi(ndi, canvas):
        raise OSError("libndi.so.6: cannot open shared object file")

    def no_mic(audio, own_ndi_name=None):
        raise ValueError(f"No input device matching {audio.device!r}. Available: USB Mic")

    monkeypatch.setattr("narrator.app_controller.NdiCaptionSender", no_ndi)
    monkeypatch.setattr("narrator.app_controller.create_capture", no_mic)
    monkeypatch.setattr("narrator.app_controller._ASR_RESTART_BACKOFF_SECONDS", 0.01)
    spawned: list = []
    plan = {"next": lambda: DeadProcess(-4)}

    def fake_process(*args, **kwargs):
        proc = plan["next"]()
        spawned.append(proc)
        return proc

    monkeypatch.setattr("narrator.app_controller.mp.Process", fake_process)
    controller = AppController(config_path)
    controller.start()
    yield controller, spawned, plan, env
    controller.stop()


def wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_app_stays_up_with_nothing_working(broken_app):
    controller, spawned, plan, env = broken_app
    assert wait_for(lambda: controller._asr_halted), "ASR must halt after the crash loop"
    assert not controller.stop_requested, "a crash loop must never stop the app"
    count = len(spawned)
    time.sleep(1.5)
    assert len(spawned) == count == 3, "no respawn once halted"

    state = controller.status_snapshot()
    assert state["asr_active"] is False and state["asr_halted"] is True
    assert state["audio_running"] is False and state["ndi_running"] is False
    found = {p["id"]: p for p in state["problems"]}
    for expected in ("asr.stopped", "asr.unit.no_gpu", "asr.model.unknown",
                     "audio.capture_failed", "ndi.output_failed"):
        assert expected in found, expected
    assert "SIGILL" in found["asr.stopped"]["message"]
    # the model cannot load on any unit: no "Use CPU instead" for that
    assert [f["action"] for f in found["asr.stopped"]["fixes"]] == ["asr.retry", "ui:asr"]
    assert state["problems"][0]["severity"] == "error"

    # manual captions still work
    controller.set_caption_mode("manual")
    controller.submit_manual_text("Dzień dobry")
    assert " ".join(controller.status_snapshot()["on_air"]) == "Dzień dobry"


def test_worker_error_text_reaches_the_problem(broken_app):
    controller, *_ = broken_app
    assert wait_for(lambda: controller._asr_halted)
    controller._handle_caption_event(CaptionEvent(CaptionEventType.ERROR, "GPU backend init failed on vulkan: boom"))
    asr = next(p for p in controller.problems() if p.id == "asr.stopped")
    assert "GPU backend init failed on vulkan: boom" in asr.message


def test_each_fix_restarts_only_its_part(broken_app, monkeypatch):
    controller, spawned, plan, env = broken_app
    assert wait_for(lambda: controller._asr_halted)
    asr_spawns = len(spawned)

    with pytest.raises(ValueError):
        controller.apply_fix("rm.rf")
    with pytest.raises(ValueError):  # ui: fixes only navigate the web UI
        controller.apply_fix("ui:asr")

    # NDI: the runtime is back -> only the NDI stream restarts
    FakeSender.created = 0
    monkeypatch.setattr("narrator.app_controller.NdiCaptionSender", FakeSender)
    controller.apply_fix("ndi.retry")
    assert FakeSender.created == 1 and controller._sender is not None
    assert controller._mic is None and len(spawned) == asr_spawns
    assert "ndi.output_failed" not in [p.id for p in controller.problems()]
    with pytest.raises(LookupError):
        controller.apply_fix("ndi.retry")  # no longer offered

    # Audio: the mic is plugged in -> rescan + only audio capture restarts
    mic = FakeMic()
    rescans = []
    monkeypatch.setattr("narrator.audio.devices.rescan_devices", lambda: rescans.append(1))
    monkeypatch.setattr("narrator.app_controller.create_capture", lambda audio, own_ndi_name=None: mic)
    sender = controller._sender
    controller.apply_fix("audio.retry")
    assert controller._mic is mic and rescans == [1]
    assert controller._sender is sender and len(spawned) == asr_spawns
    assert controller.status_snapshot()["audio_running"] is True

    # ASR: Retry spawns one fresh generation, nothing else restarts
    plan["next"] = LiveProcess
    controller.apply_fix("asr.retry")
    assert len(spawned) == asr_spawns + 1
    assert not controller._asr_halted and controller.status_snapshot()["asr_active"] is True
    assert controller._mic is mic and controller._sender is sender
    assert "asr.stopped" not in [p.id for p in controller.problems()]


@pytest.mark.parametrize("broken_app", ["small"], indirect=True)  # model fine, the GPU is the problem
def test_use_cpu_fix_writes_config_like_a_save(broken_app):
    controller, spawned, plan, env = broken_app
    assert wait_for(lambda: controller._asr_halted)
    plan["next"] = LiveProcess
    before = len(spawned)
    controller.apply_fix("asr.use_cpu")
    # the file is the source of truth; the watcher applies it (asr change ->
    # only the ASR worker restarts)
    assert load_config(controller.config_path).asr.unit == "cpu"
    assert load_config(controller.config_path).asr.model == "small"  # nothing else changed
    assert wait_for(lambda: controller.config.asr.unit == "cpu", timeout=10)
    assert wait_for(lambda: len(spawned) == before + 1, timeout=10)
    assert not controller._asr_halted


@pytest.mark.parametrize("broken_app", ["small"], indirect=True)
def test_use_cpu_is_not_offered_when_the_cpu_backend_cannot_run(broken_app):
    controller, spawned, plan, env = broken_app
    assert wait_for(lambda: controller._asr_halted)
    asr = next(p for p in controller.problems() if p.id == "asr.stopped")
    assert "asr.use_cpu" in [f.action for f in asr.fixes]  # source build: the CPU path runs here
    env.build = PYPI_WHEEL  # pre-AVX flags in FakeEnv: the CPU path would SIGILL
    asr = next(p for p in controller.problems() if p.id == "asr.stopped")
    assert "asr.use_cpu" not in [f.action for f in asr.fixes]
    with pytest.raises(LookupError):
        controller.apply_fix("asr.use_cpu")
    assert load_config(controller.config_path).asr.unit == "vulkan:0"


@pytest.mark.parametrize("broken_app", ["small"], indirect=True)
def test_unit_fixes_pin_and_switch_write_unit_and_identity(broken_app):
    """"Remember this card" records the identity of the device at the
    configured index; "Use vulkan:<n>" switches to the recorded card at its
    new index. Both only by the operator's click, through config.yaml."""
    controller, spawned, plan, env = broken_app
    assert wait_for(lambda: controller._asr_halted)
    plan["next"] = LiveProcess
    env.gpus = [GpuDevice(0, IGPU.name, False, IGPU.pci_bus_id), GpuDevice(1, RTX.name, True, RTX.pci_bus_id)]
    assert "asr.unit.identity_unrecorded" in ids(controller._health.check_now())
    controller.apply_fix("asr.pin_unit")
    saved = load_config(controller.config_path).asr
    assert (saved.unit, saved.unit_identity) == ("vulkan:0", "Intel(R) HD Graphics 610 @ 0000:00:02.0")
    assert wait_for(lambda: controller.config.asr.unit_identity == saved.unit_identity, timeout=10)

    env.gpus = [RTX, IGPU]  # the iGPU moved to index 1
    mismatch = next(p for p in controller._health.check_now() if p.id == "asr.unit.identity_mismatch")
    assert "asr.unit:vulkan:1" in fix_actions(mismatch)
    with pytest.raises(LookupError):
        controller.apply_fix("asr.unit:vulkan:7")  # not offered
    controller.apply_fix("asr.unit:vulkan:1")
    saved = load_config(controller.config_path).asr
    assert (saved.unit, saved.unit_identity) == ("vulkan:1", "Intel(R) HD Graphics 610 @ 0000:00:02.0")
    assert saved.model == "small"


def test_ndi_audio_source_status_is_reported():
    from narrator.audio.ndi_capture import NdiAudioCapture

    class NoSources:
        def source_names(self, timeout_s):
            time.sleep(0.01)
            return []

        def close(self):
            pass

    capture = NdiAudioCapture("MIXER", backend_factory=NoSources, discovery_timeout_s=0.01)
    capture.start()
    try:
        assert wait_for(lambda: capture.status == "searching", timeout=5)
    finally:
        capture.stop()

    def no_runtime():
        raise OSError("libndi missing")

    capture = NdiAudioCapture("MIXER", backend_factory=no_runtime)
    capture.start()
    try:
        assert wait_for(lambda: capture.status == "runtime_unavailable", timeout=5)
        assert "libndi missing" in capture.status_detail
    finally:
        capture.stop()


def test_worker_reports_its_error_before_dying(monkeypatch):
    """run_asr_worker puts an ERROR event on the queue, then re-raises (the
    supervisor sees the exit, the panel shows the text)."""
    import queue

    from narrator.asr import worker

    def boom(*args, **kwargs):
        raise RuntimeError("GPU backend init failed on vulkan: no device")

    monkeypatch.setattr(worker, "_run", boom)
    monkeypatch.setattr("narrator.logsetup.attach_worker_file_log", lambda: None)
    events: queue.Queue = queue.Queue()
    with pytest.raises(RuntimeError):
        worker.run_asr_worker(RootConfig().asr, queue.Queue(), events, threading.Event())
    event = events.get_nowait()
    assert event.type == CaptionEventType.ERROR
    assert "no device" in event.text
