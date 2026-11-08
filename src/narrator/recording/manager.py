"""RecordingManager: session lifecycle, the disk guard, retention purge and
the draft scheduler. Owned by AppController.

Threads: `feed()` (the audio feed thread) and `on_air_*()` (the event-drain
thread, the web thread for manual text) only read `self._session` and hand
the data on without blocking. start/stop/delete run on whichever thread asks
(web loop executor, OSC, the controller) under `_lock`. One maintenance
thread ticks every couple of seconds: it stops a session whose disk ran low,
purges by `recording.retention_days`, and runs at most one draft subprocess.

Draft scheduling: drafts are queued in each session's session.json (so they
survive a restart) and run one at a time in a low-priority subprocess
(`python -m narrator.recording.draft`). A queued draft starts only when the
live ASR worker leaves headroom on the device it would share (`draft_gate`):
ASR not running, the draft on the CPU while ASR is on the GPU, or ASR's
measured busy fraction at most `recording.draft_max_asr_busy`. A running
draft that loses that headroom is suspended (SIGSTOP) until it returns, so
speech starting mid-draft gets the GPU back. "Generate draft now" bypasses
the gate and is never suspended.
"""
from __future__ import annotations

import collections
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from narrator.asr.events import CaptionEvent
from narrator.config.schema import AsrDevice, RootConfig
from narrator.paths import recordings_dir, reference_library_dir
from narrator.recording import store
from narrator.recording.draft import build_request, write_request
from narrator.recording.session import (
    AUDIO_FILE,
    DRAFT_SRT,
    RecordingError,
    RecordingSession,
    free_disk_mb,
    read_meta,
    soundfile_flac_available,
)

logger = logging.getLogger(__name__)
draft_log = logging.getLogger("narrator.recording.draft")

_TICK_SECONDS = 2.0
_PURGE_EVERY_SECONDS = 3600.0
_RESUME_FRACTION = 0.7  # hysteresis: resume a suspended draft below 70% of the limit


def draft_gate(asr_load: dict, draft_device: AsrDevice, max_busy: float) -> tuple[bool, str]:
    """May a (non-forced) draft run now? -> (allowed, reason shown in the UI).

    asr_load: {"active": bool, "busy": float | None, "device": AsrDevice}.
    """
    if not asr_load.get("active"):
        return True, "live ASR is not running"
    asr_device = asr_load.get("device")
    if draft_device == AsrDevice.CPU and asr_device not in (AsrDevice.CPU, None):
        return True, "draft on the CPU, live ASR on the GPU"
    if max_busy <= 0:
        return False, "waiting until live ASR stops (draft_max_asr_busy is 0)"
    busy = asr_load.get("busy")
    if busy is None:
        return False, "waiting: live ASR load not measured yet"
    if busy <= max_busy:
        return True, f"live ASR busy {busy:.0%} (limit {max_busy:.0%})"
    return False, f"waiting: live ASR busy {busy:.0%} (limit {max_busy:.0%})"


def _draft_command(session_dir: Path) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--draft-recording", str(session_dir)]
    return [sys.executable, "-m", "narrator.recording.draft", str(session_dir)]


def language_name(code: str) -> str:
    from narrator.asr.whisper_languages import _LANGUAGE_NAMES

    return _LANGUAGE_NAMES.get(code, "Polish")


class RecordingManager:
    def __init__(
        self,
        config_getter: Callable[[], RootConfig],
        *,
        asr_load: Callable[[], dict] = lambda: {"active": False, "busy": None, "device": None},
        vocabulary: Callable[[], list[str]] = lambda: [],
        on_change: Callable[[], None] | None = None,
        launcher: Callable[[list[str]], subprocess.Popen] | None = None,
        session_factory=RecordingSession,
        free_mb: Callable[[Path], float] = free_disk_mb,
        flac_available: Callable[[], bool] = soundfile_flac_available,
    ) -> None:
        self._config = config_getter
        self._asr_load = asr_load
        self._vocabulary = vocabulary
        self._on_change = on_change
        self._launch = launcher or self._default_launch
        self._session_factory = session_factory
        self._free_mb = free_mb
        self._flac_available = flac_available
        self._available: bool | None = None
        self._lock = threading.RLock()
        self._session: RecordingSession | None = None
        self._disk_low = threading.Event()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_purge = float("-inf")
        self._last_stop_reason: str | None = None
        # The one running draft subprocess.
        self._draft_proc: subprocess.Popen | None = None
        self._draft_id: str | None = None
        self._draft_forced = False
        self._draft_suspended = False
        self._draft_tail: collections.deque = collections.deque(maxlen=20)
        self.draft_wait_reason: str | None = None

    # ------------------------------------------------------------ basics

    @property
    def root(self) -> Path:
        directory = self._config().recording.directory
        return Path(directory).expanduser() if directory else recordings_dir()

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = bool(self._flac_available())
        return self._available

    @property
    def recording(self) -> bool:
        return self._session is not None

    @property
    def active_id(self) -> str | None:
        session = self._session
        return session.id if session is not None else None

    def _changed(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                logger.exception("recording state callback failed")

    # ------------------------------------------------------------ sessions

    def start(self) -> str:
        """Start a session (no-op when one is running); returns its id."""
        with self._lock:
            if self._session is not None:
                return self._session.id
            if not self.available:
                raise RecordingError(
                    "recording needs FLAC support from the 'soundfile' package (pip install soundfile)"
                )
            config = self._config()
            root = self.root
            free = self._free_mb(root)
            if free < config.recording.min_free_disk_mb:
                raise RecordingError(
                    f"only {free:.0f} MB free on the recordings disk "
                    f"(recording.min_free_disk_mb is {config.recording.min_free_disk_mb})"
                )
            self._disk_low.clear()
            self._session = self._session_factory(
                root, config, on_disk_low=self._on_disk_low, free_mb=self._free_mb
            )
            self._last_stop_reason = None
            logger.info("recording started: %s", self._session.directory)
        self._changed()
        return self._session.id

    def stop(self, reason: str = "stopped by the operator") -> dict | None:
        with self._lock:
            session, self._session = self._session, None
            if session is None:
                return None
            meta = session.stop(reason, queue_draft=self._config().recording.draft_enabled)
            self._last_stop_reason = meta.get("stop_reason")
        logger.info(
            "recording stopped (%s): %s, %.1f s of audio, %d dropped frame(s)",
            meta.get("stop_reason"), session.directory, meta.get("duration_s", 0.0), meta.get("dropped_frames", 0),
        )
        self._changed()
        self._wake.set()  # a queued draft may be able to start
        return meta

    def set_recording(self, enabled: bool) -> None:
        if enabled:
            self.start()
        else:
            self.stop()

    def _on_disk_low(self) -> None:
        # Called on the writer thread: never stop from there (stop() joins it).
        self._disk_low.set()
        self._wake.set()

    def feed(self, frame) -> None:
        session = self._session
        if session is not None:
            session.feed(frame)

    def on_air_final(self, event: CaptionEvent) -> None:
        session = self._session
        if session is not None:
            session.on_air(event)

    def on_air_text(self, text: str) -> None:
        session = self._session
        if session is not None:
            session.on_air_text(text)

    def note_asr_change(self, config: RootConfig) -> None:
        session = self._session
        if session is not None:
            session.note_asr_change(config)

    def status(self) -> dict:
        session = self._session
        return {
            "available": self.available,
            "recording": session is not None,
            "session_id": session.id if session is not None else None,
            "elapsed_s": round(session.elapsed_s, 1) if session is not None else None,
            "dropped_frames": session.audio.dropped_frames if session is not None else 0,
            "last_stop_reason": self._last_stop_reason,
            "draft_running": self._draft_id,
            "draft_suspended": self._draft_suspended,
            "draft_waiting": self.draft_wait_reason,
        }

    # ------------------------------------------------------------ library

    def list(self) -> list[dict]:
        items = store.list_recordings(self.root, self.active_id)
        for item in items:
            if item["id"] == self._draft_id:
                item["draft"]["suspended"] = self._draft_suspended
            elif item["status"] == "draft pending" and self.draft_wait_reason:
                item["draft"]["waiting"] = self.draft_wait_reason
        return items

    def disk(self) -> dict:
        info = store.disk_usage(self.root)
        info["min_free_disk_mb"] = self._config().recording.min_free_disk_mb
        info["directory"] = str(self.root)
        return info

    def delete(self, recording_id: str) -> None:
        with self._lock:
            if recording_id == self._draft_id:
                self._kill_draft(requeue=False)
            store.delete_recording(self.root, recording_id, self.active_id)

    def delete_all(self) -> int:
        with self._lock:
            self._kill_draft(requeue=False)
            keep = {self.active_id} if self.active_id else set()
            return store.delete_all(self.root, keep=keep)

    def save_corrected(self, recording_id: str, data: bytes | str) -> int:
        return store.save_corrected(self.root, recording_id, data, self.active_id)

    def use_as_reference(self, recording_id: str) -> dict:
        library = reference_library_dir()
        if library is None:
            raise RecordingError("the reference library (scripts/test_audio) only exists in a repository checkout")
        meta = read_meta(store.session_dir(self.root, recording_id))
        whitelist = meta.get("language_whitelist") or []
        name = language_name(whitelist[0]) if len(whitelist) == 1 else "Polish"
        return store.use_as_reference(self.root, recording_id, library, name)

    def request_draft(self, recording_id: str, force: bool = True) -> None:
        """Queue (or re-queue) a draft; force = "Generate draft now"."""
        with self._lock:
            if recording_id == self.active_id:
                raise PermissionError("this recording is still running")
            directory = store.session_dir(self.root, recording_id)
            if not (directory / AUDIO_FILE).is_file():
                raise FileNotFoundError("this recording has no audio")
            if recording_id == self._draft_id:
                if force and self._draft_suspended:
                    self._resume_draft()
                self._draft_forced = self._draft_forced or force
                return
            store.update_meta(directory, draft={"state": "queued", "queued_at": time.time(), "forced": force})
        self._wake.set()

    # ------------------------------------------------------------ background

    def start_background(self) -> None:
        self._recover_interrupted()
        self._thread = threading.Thread(target=self._run, name="narrator-recordings", daemon=True)
        self._thread.start()

    def _recover_interrupted(self) -> None:
        """Sessions left "recording" by a crash become "interrupted"; drafts
        left "running" are queued again (or ready, if the file got written)."""
        for directory in store.session_dirs(self.root):
            meta = read_meta(directory)
            if not meta:
                continue
            changes = {}
            if meta.get("status") == "recording":
                changes["status"] = "interrupted"
                changes["stop_reason"] = "the app stopped while recording"
                try:
                    import soundfile

                    info = soundfile.info(str(directory / AUDIO_FILE))
                    changes["duration_s"] = round(info.frames / info.samplerate, 3)
                except Exception:
                    pass
                if self._config().recording.draft_enabled:
                    changes["draft"] = {"state": "queued", "queued_at": time.time()}
            draft = meta.get("draft") or {}
            if draft.get("state") == "running":
                if (directory / DRAFT_SRT).is_file():
                    changes["draft"] = {**draft, "state": "ready"}
                else:
                    changes["draft"] = {"state": "queued", "queued_at": time.time()}
            if changes:
                store.update_meta(directory, **changes)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("recordings maintenance failed")
            self._wake.wait(_TICK_SECONDS)
            self._wake.clear()

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self._disk_low.is_set() and self._session is not None:
            self._disk_low.clear()
            self.stop(reason="low disk space")
        if now - self._last_purge >= _PURGE_EVERY_SECONDS:
            self._last_purge = now
            days = self._config().recording.retention_days
            if days > 0:
                keep = {i for i in (self.active_id, self._draft_id) if i}
                store.purge_older_than(self.root, days, keep=keep)
        self._schedule_drafts()

    def _schedule_drafts(self) -> None:
        with self._lock:
            config = self._config().recording
            if self._draft_proc is not None:
                code = self._draft_proc.poll()
                if code is not None:
                    self._finish_draft(code)
                else:
                    self._regulate_running_draft(config)
                    return
            queued = self._queued_drafts()
            if not queued:
                self.draft_wait_reason = None
                return
            forced, directory = queued[0]
            if not forced:
                allowed, reason = draft_gate(self._asr_load(), config.draft_device, config.draft_max_asr_busy)
                self.draft_wait_reason = None if allowed else reason
                if not allowed:
                    return
            self.draft_wait_reason = None
            self._launch_draft(directory, forced)

    def _queued_drafts(self) -> list[tuple[bool, Path]]:
        found = []
        for directory in store.session_dirs(self.root):
            if directory.name == self.active_id:
                continue
            draft = read_meta(directory).get("draft") or {}
            if draft.get("state") == "queued":
                found.append((bool(draft.get("forced")), draft.get("queued_at") or 0.0, directory))
        found.sort(key=lambda item: (not item[0], item[1]))  # forced first, then oldest
        return [(forced, directory) for forced, _, directory in found]

    def _launch_draft(self, directory: Path, forced: bool) -> None:
        config = self._config()
        meta = read_meta(directory)
        whitelist = meta.get("language_whitelist") or list(config.asr.language_whitelist)
        request = build_request(config, whitelist, self._vocabulary())
        write_request(directory, request)
        try:
            proc = self._launch(_draft_command(directory))
        except OSError as exc:
            store.update_meta(directory, draft={"state": "failed", "error": f"cannot start: {exc}"})
            return
        self._draft_proc, self._draft_id = proc, directory.name
        self._draft_forced, self._draft_suspended = forced, False
        self._draft_tail.clear()
        store.update_meta(
            directory,
            draft={"state": "running", "model": request["model"], "started_at": time.time(), "forced": forced},
        )
        logger.info("draft for recording %s started (model %s, pid %s)", directory.name, request["model"], proc.pid)
        if proc.stdout is not None:
            threading.Thread(target=self._pump_output, args=(proc,), daemon=True).start()

    def _pump_output(self, proc) -> None:
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip() if isinstance(raw, bytes) else str(raw).rstrip()
            if line:
                self._draft_tail.append(line)
                draft_log.info("%s", line)

    def _finish_draft(self, code: int) -> None:
        directory = self.root / self._draft_id
        meta = read_meta(directory)
        previous = meta.get("draft") or {}
        if code == 0 and (directory / DRAFT_SRT).is_file():
            draft = {**previous, "state": "ready", "finished_at": time.time()}
            logger.info("draft for recording %s ready", self._draft_id)
        else:
            tail = [line for line in self._draft_tail if "failed" in line.lower()] or list(self._draft_tail)[-3:]
            draft = {**previous, "state": "failed", "finished_at": time.time(),
                     "error": (tail[-1] if tail else f"exit code {code}")[:500]}
            logger.error("draft for recording %s failed (exit code %s)", self._draft_id, code)
        if directory.is_dir():
            store.update_meta(directory, draft=draft)
        self._draft_proc, self._draft_id = None, None
        self._draft_forced = self._draft_suspended = False

    def _regulate_running_draft(self, config) -> None:
        if self._draft_forced or not hasattr(signal, "SIGSTOP"):
            return
        load = self._asr_load()
        allowed, reason = draft_gate(load, config.draft_device, config.draft_max_asr_busy)
        if not self._draft_suspended and not allowed:
            self._signal_draft(signal.SIGSTOP)
            self._draft_suspended = True
            self.draft_wait_reason = f"suspended -- {reason}"
            logger.info("draft for %s suspended (%s)", self._draft_id, reason)
        elif self._draft_suspended:
            relaxed = config.draft_max_asr_busy * _RESUME_FRACTION
            busy = load.get("busy")
            if not load.get("active") or (allowed and (busy is None or busy <= relaxed)) or (
                allowed and config.draft_device == AsrDevice.CPU
            ):
                self._resume_draft()

    def _resume_draft(self) -> None:
        if hasattr(signal, "SIGCONT"):
            self._signal_draft(signal.SIGCONT)
        self._draft_suspended = False
        self.draft_wait_reason = None
        logger.info("draft for %s resumed", self._draft_id)

    def _signal_draft(self, signum) -> None:
        try:
            self._draft_proc.send_signal(signum)
        except (OSError, AttributeError):
            pass

    def _kill_draft(self, requeue: bool) -> None:
        proc, recording_id = self._draft_proc, self._draft_id
        if proc is None:
            return
        if self._draft_suspended and hasattr(signal, "SIGCONT"):
            self._signal_draft(signal.SIGCONT)  # a stopped process cannot act on SIGTERM
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        self._draft_proc, self._draft_id = None, None
        self._draft_forced = self._draft_suspended = False
        directory = self.root / recording_id
        if requeue and directory.is_dir():
            store.update_meta(directory, draft={"state": "queued", "queued_at": time.time()})

    @staticmethod
    def _default_launch(command: list[str]) -> subprocess.Popen:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    def shutdown(self) -> None:
        """Stop the session (flushing audio and captions), the maintenance
        thread, and any draft (queued again for the next run)."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.stop(reason="the app stopped")
        with self._lock:
            self._kill_draft(requeue=True)
