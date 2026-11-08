import logging
import os

from narrator import logsetup


def test_enable_file_logging_writes_a_rotating_log_and_exports_dir_for_worker(tmp_path, monkeypatch):
    root = logging.getLogger()
    before_handlers, before_level = list(root.handlers), root.level
    monkeypatch.delenv(logsetup.LOG_DIR_ENV, raising=False)
    try:
        path = logsetup.enable_file_logging(tmp_path / "log")
        logging.getLogger("narrator.test").info("hello from the app")
        assert os.environ[logsetup.LOG_DIR_ENV] == str(tmp_path / "log")

        logsetup.attach_worker_file_log()  # what the spawned ASR worker does
        logging.getLogger("narrator.test").warning("hello from the worker")
        for h in root.handlers:
            h.flush()
        assert "hello from the app" in path.read_text(encoding="utf-8")
        assert "hello from the worker" in (tmp_path / "log" / "narrator-asr.log").read_text(encoding="utf-8")
    finally:
        for h in list(root.handlers):
            if h not in before_handlers:
                root.removeHandler(h)
                h.close()
        root.setLevel(before_level)
        os.environ.pop(logsetup.LOG_DIR_ENV, None)
