"""Headless mode: SIGTERM/SIGINT take the one shutdown path
(AppController.stop), and the exit status tells systemd whether to restart."""
import os
import signal
import threading
import time

import pytest

from narrator import main as narrator_main


class FakeController:
    def __init__(self):
        self.stop_calls = 0
        self.stop_requested = False

    def stop(self):
        self.stop_calls += 1
        self.stop_requested = True


def _run_in_thread(controller, shutdown):
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(code=narrator_main.run_headless(controller, shutdown, poll_seconds=0.02))
    )
    thread.start()
    return thread, result


def test_shutdown_event_stops_controller_and_exits_zero():
    controller, shutdown = FakeController(), threading.Event()
    thread, result = _run_in_thread(controller, shutdown)
    time.sleep(0.1)
    assert thread.is_alive() and controller.stop_calls == 0
    shutdown.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert controller.stop_calls == 1
    assert result["code"] == 0


def test_app_stopping_itself_exits_nonzero_so_systemd_restarts():
    """Safety net only: since the 2026-10-17 failure policy no code path
    stops the app by itself (an ASR crash loop halts ASR and shows a problem,
    see test_health.py). Should one ever appear, systemd must restart it."""
    controller, shutdown = FakeController(), threading.Event()
    thread, result = _run_in_thread(controller, shutdown)
    controller.stop_requested = True
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert controller.stop_calls == 1  # idempotent controller.stop() is still run
    assert result["code"] == 1


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="needs POSIX signals")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_real_signal_reaches_the_shutdown_event(signum):
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    shutdown = threading.Event()
    try:
        narrator_main.install_signal_handlers(lambda _signum: shutdown.set())
        os.kill(os.getpid(), signum)
        assert shutdown.wait(timeout=5)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="needs POSIX signals")
def test_sigterm_to_a_headless_process_shuts_down_cleanly():
    """What `systemctl --user stop narrator` does, end to end minus hardware."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import threading
        from narrator import main

        class C:
            stop_requested = False
            def stop(self):
                print("STOPPED", flush=True)

        shutdown = threading.Event()
        main.install_signal_handlers(lambda signum: shutdown.set())
        print("READY", flush=True)
        raise SystemExit(main.run_headless(C(), shutdown))
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0
    assert "STOPPED" in out
