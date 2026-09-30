from narrator.app_controller import AppController
from narrator.asr.events import CaptionEvent, CaptionEventType


class FakeRenderer:
    def __init__(self) -> None:
        self.final_texts: list[str] = []
        self.partial_preview: str | None = None
        self.cleared = False
        self.pruned = False

    def append_final_text(self, text: str) -> None:
        self.final_texts.append(text)

    def set_partial_preview(self, text: str) -> None:
        self.partial_preview = text

    def prune_history(self) -> int:
        self.pruned = True
        return 0

    def clear(self) -> None:
        self.cleared = True
        self.final_texts = []
        self.partial_preview = None


def make_controller() -> tuple[AppController, FakeRenderer]:
    # AppController.__init__ wires up real audio/NDI/config-watching infra;
    # this exercises only the caption-mode/manual/clear logic, which only
    # needs `_renderer` and `_caption_mode`, so we bypass __init__ entirely.
    controller = AppController.__new__(AppController)
    renderer = FakeRenderer()
    controller._renderer = renderer
    controller._caption_mode = "auto"
    controller.resolved_asr_backend = "unresolved"
    return controller, renderer


def test_auto_mode_applies_asr_final_and_partial_events():
    controller, renderer = make_controller()

    controller._handle_caption_event(CaptionEvent(CaptionEventType.PARTIAL, "hell"))
    controller._handle_caption_event(CaptionEvent(CaptionEventType.FINAL, "hello"))

    assert renderer.final_texts == ["hello"]


def test_manual_mode_ignores_asr_final_partial_and_clear_events():
    controller, renderer = make_controller()
    controller.set_caption_mode("manual")

    controller._handle_caption_event(CaptionEvent(CaptionEventType.PARTIAL, "hell"))
    controller._handle_caption_event(CaptionEvent(CaptionEventType.FINAL, "hello"))
    controller._handle_caption_event(CaptionEvent(CaptionEventType.CLEAR))

    assert renderer.final_texts == []
    assert renderer.partial_preview is None
    assert not renderer.cleared


def test_manual_mode_still_applies_status_events():
    controller, renderer = make_controller()
    controller.set_caption_mode("manual")

    controller._handle_caption_event(CaptionEvent(CaptionEventType.STATUS, "CPU"))

    assert controller.resolved_asr_backend == "CPU"


def test_submit_manual_text_appends_and_clears_partial_preview():
    controller, renderer = make_controller()
    controller.set_caption_mode("manual")

    controller.submit_manual_text("  operator typed this  ")

    assert renderer.final_texts == ["operator typed this"]
    assert renderer.partial_preview == ""


def test_submit_manual_text_ignores_blank_input():
    controller, renderer = make_controller()

    controller.submit_manual_text("   ")

    assert renderer.final_texts == []


def test_clear_captions_works_in_either_mode():
    controller, renderer = make_controller()

    controller.clear_captions()

    assert renderer.cleared is True


def test_asr_clear_event_prunes_history_instead_of_wiping_screen():
    """The ASR worker's silence-timeout CLEAR must not blank the lower-third.
    It only prunes off-screen history; the wipe path stays operator-only
    (Ctrl+N / the GUI Clear button)."""
    controller, renderer = make_controller()
    controller._handle_caption_event(CaptionEvent(CaptionEventType.FINAL, "hello"))
    controller._handle_caption_event(CaptionEvent(CaptionEventType.CLEAR))

    assert renderer.pruned is True
    assert renderer.cleared is False
    assert renderer.final_texts == ["hello"]


def test_set_caption_mode_rejects_unknown_mode():
    controller, _renderer = make_controller()
    try:
        controller.set_caption_mode("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an invalid caption mode")


# --- ASR worker lifecycle -------------------------------------------------
#
# Regression tests for a cluster of restart bugs with one operator-visible
# symptom: GPU worked on first launch, then any config edit dropped it to CPU
# permanently and leaked a worker process per edit until the machine ran out
# of memory.

import threading
import time

from narrator.config.schema import AsrDevice, RootConfig


class FakeProcess:
    """Stands in for mp.Process. Stays 'alive' until stopped, so a test can
    model a worker that is slow to honour its stop event."""

    def __init__(self, ignore_stop: bool = False) -> None:
        self.alive = True
        self.exitcode = None
        self.terminated = False
        self.killed = False
        self._ignore_stop = ignore_stop

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout=None) -> None:
        if not self._ignore_stop:
            self.alive = False
            self.exitcode = 0

    def terminate(self) -> None:
        self.terminated = True
        if self._ignore_stop:
            self.alive = False
            self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        self.alive = False
        self.exitcode = -9


def make_lifecycle_controller(monkeypatch, processes):
    """Controller with just enough state for the ASR lifecycle (and stop()),
    and mp.Process replaced by a queue of FakeProcesses."""
    controller = AppController.__new__(AppController)
    controller._config = RootConfig()
    controller._asr_lock = threading.Lock()
    controller._stop_event = threading.Event()
    controller._asr_process = None
    controller._asr_stop_event = None
    controller._asr_spawn_time = 0.0
    controller._consecutive_quick_crashes = 0
    controller._audio_queue = None
    controller._event_queue = None
    # stop() support (the crash-loop policy path calls it)
    controller._stopped = False
    controller._stop_lock = threading.Lock()
    controller._ndi_stop_event = None
    controller._audio_stop_event = None
    controller._watcher = None
    controller._mic = None
    controller._ndi_thread = None
    controller._audio_thread = None
    controller._sender = None
    controller._threads = []

    spawned = []
    made = iter(processes)

    def fake_process(*args, **kwargs):
        proc = next(made)
        spawned.append(proc)
        return proc

    monkeypatch.setattr("narrator.app_controller.mp.Process", fake_process)
    monkeypatch.setattr("narrator.app_controller.mp.Event", lambda: threading.Event())
    return controller, spawned


def test_restart_gives_each_generation_its_own_stop_event(monkeypatch):
    """The old shared event was cleared by the incoming generation, which
    un-stopped any outgoing worker that hadn't exited yet."""
    controller, spawned = make_lifecycle_controller(monkeypatch, [FakeProcess(), FakeProcess()])

    with controller._asr_lock:
        controller._spawn_asr_process()
    first_stop_event = controller._asr_stop_event

    controller._restart_asr_process()

    assert controller._asr_stop_event is not first_stop_event
    assert first_stop_event.is_set(), "outgoing worker must stay stopped"
    assert not controller._asr_stop_event.is_set(), "incoming worker must not start stopped"


def test_restart_does_not_count_as_a_crash(monkeypatch):
    """Three config edits used to trip the crash-loop breaker and pin the app
    to CPU, because a deliberate restart looked identical to a crash."""
    controller, _ = make_lifecycle_controller(
        monkeypatch, [FakeProcess() for _ in range(4)]
    )
    with controller._asr_lock:
        controller._spawn_asr_process()

    for _ in range(3):
        controller._restart_asr_process()

    assert controller._consecutive_quick_crashes == 0
    assert not controller._stopped, "config edits must not terminate the app"


def test_restart_escalates_to_terminate_when_worker_ignores_stop(monkeypatch):
    """A worker mid-decode or loading a 3GB model won't exit cooperatively;
    leaving it running leaked a whole whisper model per config edit."""
    stubborn = FakeProcess(ignore_stop=True)
    controller, _ = make_lifecycle_controller(monkeypatch, [stubborn, FakeProcess()])
    with controller._asr_lock:
        controller._spawn_asr_process()

    controller._restart_asr_process()

    assert stubborn.terminated, "outgoing worker must be terminated, not orphaned"
    assert not stubborn.is_alive()


def test_supervisor_ignores_a_worker_detached_for_restart(monkeypatch):
    """While a deliberate restart is in flight `_asr_process` is None, which
    is what stops the supervisor from racing it and spawning a duplicate."""
    controller, spawned = make_lifecycle_controller(monkeypatch, [FakeProcess()])
    with controller._asr_lock:
        controller._spawn_asr_process()
    controller._asr_process = None  # simulate restart in flight

    supervisor = threading.Thread(target=controller._supervise_asr_loop, daemon=True)
    supervisor.start()
    time.sleep(0.2)
    controller._stop_event.set()
    supervisor.join(timeout=5)

    assert len(spawned) == 1, "supervisor must not spawn while a restart is in flight"


def test_crash_loop_terminates_app_instead_of_forcing_cpu(monkeypatch):
    """The configured device is final by policy. The old behavior -- forcing
    CPU after 3 quick crashes -- silently downgraded a GPU config the operator
    explicitly chose. Now the supervisor stops the app with an explanation."""
    dead = FakeProcess()
    dead.alive = False
    dead.exitcode = -6
    controller, spawned = make_lifecycle_controller(monkeypatch, [FakeProcess(), FakeProcess()])
    controller._consecutive_quick_crashes = 2  # one more quick crash trips it
    with controller._asr_lock:
        controller._spawn_asr_process()
    controller._asr_process.alive = False  # the generation just died quickly
    controller._asr_process.exitcode = -6

    supervisor = threading.Thread(target=controller._supervise_asr_loop, daemon=True)
    supervisor.start()
    supervisor.join(timeout=10)

    assert not supervisor.is_alive(), "supervisor must exit after tripping the breaker"
    assert controller._stopped, "app must be stopped, not downgraded to CPU"
    assert controller._stop_event.is_set()
    # No force-CPU state exists anymore; a fresh spawn with the same device
    # config is what stop() teardown implies.
    assert controller._config.asr.device != AsrDevice.CPU


def test_drain_audio_queue_drops_stale_frames():
    """Frames queued for the outgoing worker must not reach the successor's
    decoder: they would produce a caption for pre-restart speech."""
    import queue as queue_mod

    controller = AppController.__new__(AppController)
    controller._audio_queue = queue_mod.Queue()
    for i in range(3):
        controller._audio_queue.put(i)

    drained = controller._drain_audio_queue()

    assert drained == 3
    assert controller._audio_queue.empty()
