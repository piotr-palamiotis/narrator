"""One recording session: a folder with the audio the ASR received, what
went on air, and its metadata.

    <recordings dir>/<YYYY-MM-DD_HH-MM-SS>/
        audio.flac      mono, the exact signal fed to the ASR worker (after
                        channel selection / resampling), 24-bit FLAC
        live.srt        the captions that went on air (post-blocklist)
        session.json    start/end, model, language whitelist, status, draft
        draft.srt       offline best-effort pass, written after the session
        corrected.srt   the operator's fix, uploaded from the web UI

Audio never blocks the caller: AppController's feed thread hands each frame
to `AudioRecorder.offer`, a put_nowait into a bounded queue; a writer thread
encodes it. A full queue (the disk stalled for longer than the queue holds)
drops the frame and counts it -- a gap in the recording beats a stall in the
live captions.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.captions.transcript import TranscriptWriter
from narrator.config.schema import RootConfig, TranscriptConfig, TranscriptFormat

logger = logging.getLogger(__name__)

AUDIO_FILE = "audio.flac"
LIVE_SRT = "live.srt"
DRAFT_SRT = "draft.srt"
CORRECTED_SRT = "corrected.srt"
META_FILE = "session.json"
DRAFT_REQUEST = "draft_request.json"

SESSION_ID_FORMAT = "%Y-%m-%d_%H-%M-%S"
# ~30 s of 30 ms frames: a disk can stall this long without losing audio.
_QUEUE_FRAMES = 1000
_DISK_CHECK_SECONDS = 5.0


class RecordingError(RuntimeError):
    """Recording cannot start (low disk, no FLAC support, ...)."""


def free_disk_mb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024 * 1024)


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_meta(directory: Path) -> dict:
    try:
        return json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def soundfile_flac_available() -> bool:
    try:
        import soundfile
    except Exception:  # missing package or libsndfile
        return False
    return "FLAC" in soundfile.available_formats()


class AudioRecorder:
    """Bounded, non-blocking FLAC writer.

    `offer()` is called from the audio feed thread and never waits: it is a
    `put_nowait`, and a full queue drops the frame (`dropped_frames`). The
    writer thread batches queued frames into the file and checks free disk
    space every few seconds; below `min_free_mb` it stops writing and calls
    `on_disk_low` (the manager then stops the session from its own thread).
    """

    def __init__(
        self,
        path: Path,
        sample_rate: int,
        *,
        min_free_mb: float = 0.0,
        on_disk_low: Callable[[], None] | None = None,
        max_queue_frames: int = _QUEUE_FRAMES,
        free_mb: Callable[[Path], float] = free_disk_mb,
        file_factory: Callable[[Path, int], object] | None = None,
    ) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.dropped_frames = 0
        self.frames_written = 0
        self.samples_written = 0
        self.disk_low = False
        self.error: str | None = None
        self._min_free_mb = min_free_mb
        self._on_disk_low = on_disk_low
        self._free_mb = free_mb
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_frames)
        self._closing = threading.Event()
        self._file = (file_factory or _open_flac)(path, sample_rate)
        self._thread = threading.Thread(target=self._run, name="narrator-recorder", daemon=True)
        self._thread.start()

    def offer(self, frame: np.ndarray) -> bool:
        """Queue one frame for writing; False (and counted) when dropped."""
        if self._closing.is_set() or self.disk_low:
            return False
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            self.dropped_frames += 1
            return False

    def _run(self) -> None:
        next_disk_check = 0.0
        while True:
            try:
                first = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._closing.is_set():
                    break
                continue
            batch = [first]
            while len(batch) < 200:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if self.disk_low or self.error:
                continue  # stopped writing; drain until closed
            now = time.monotonic()
            if now >= next_disk_check:
                next_disk_check = now + _DISK_CHECK_SECONDS
                if self._min_free_mb > 0 and self._free_mb(self.path.parent) < self._min_free_mb:
                    self.disk_low = True
                    logger.error(
                        "recording: less than %d MB free on the recordings disk; stopping the recording",
                        self._min_free_mb,
                    )
                    if self._on_disk_low is not None:
                        self._on_disk_low()
                    continue
            data = np.clip(np.concatenate(batch).astype(np.float32, copy=False), -1.0, 1.0)
            try:
                self._file.write(data)
            except Exception as exc:  # disk full, I/O error: never reaches captions
                self.error = str(exc)
                logger.exception("recording: writing %s failed; recording stopped", self.path)
                if self._on_disk_low is not None:
                    self._on_disk_low()
                continue
            self.frames_written += len(batch)
            self.samples_written += len(data)

    def close(self, timeout: float = 10.0) -> None:
        """Write what is queued, then close the file."""
        self._closing.set()
        self._thread.join(timeout=timeout)
        try:
            self._file.close()
        except Exception:
            logger.exception("recording: closing %s failed", self.path)

    @property
    def duration_s(self) -> float:
        return self.samples_written / self.sample_rate


def _open_flac(path: Path, sample_rate: int):
    import soundfile

    return soundfile.SoundFile(
        str(path), mode="w", samplerate=sample_rate, channels=1, format="FLAC", subtype="PCM_24"
    )


def new_session_dir(root: Path, now: datetime) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = now.strftime(SESSION_ID_FORMAT)
    candidate = root / base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base}-{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


class RecordingSession:
    """An active recording. Created by RecordingManager.start(); `feed` and
    `on_air` are called from the app's threads and never block on disk."""

    def __init__(
        self,
        root: Path,
        config: RootConfig,
        *,
        on_disk_low: Callable[[], None] | None = None,
        now: Callable[[], datetime] = datetime.now,
        clock: Callable[[], float] = time.monotonic,
        file_factory=None,
        free_mb: Callable[[Path], float] = free_disk_mb,
    ) -> None:
        started = now()
        self.directory = new_session_dir(root, started)
        self.id = self.directory.name
        self.started_monotonic = clock()
        self._clock = clock
        self._now = now
        self.meta: dict = {
            "id": self.id,
            "started_at": started.astimezone().isoformat(timespec="seconds"),
            "ended_at": None,
            "status": "recording",
            "stop_reason": None,
            "sample_rate": config.audio.sample_rate,
            "audio_device": config.audio.device,
            "model": config.asr.model,
            "language_whitelist": list(config.asr.language_whitelist),
            "asr_unit": config.asr.unit,
            "asr_changes": [],
            "duration_s": 0.0,
            "dropped_frames": 0,
            "draft": {"state": "none"},
        }
        self._write_meta()
        self.audio = AudioRecorder(
            self.directory / AUDIO_FILE,
            config.audio.sample_rate,
            min_free_mb=config.recording.min_free_disk_mb,
            on_disk_low=on_disk_low,
            file_factory=file_factory,
            free_mb=free_mb,
        )
        self._live = TranscriptWriter(
            TranscriptConfig(enabled=True, format=TranscriptFormat.SRT),
            clock=clock,
            path=self.directory / LIVE_SRT,
            origin=self.started_monotonic,
        )
        self._live_lock = threading.Lock()

    def feed(self, frame: np.ndarray) -> None:
        self.audio.offer(frame)

    def on_air(self, event: CaptionEvent) -> None:
        """A caption that went on air (an applied ASR FINAL or a manual line)."""
        with self._live_lock:
            self._live.add_final(event)

    def on_air_text(self, text: str) -> None:
        self.on_air(CaptionEvent(CaptionEventType.FINAL, text, timestamp=self._clock()))

    def note_asr_change(self, config: RootConfig) -> None:
        self.meta["asr_changes"].append(
            {
                "at_s": round(self._clock() - self.started_monotonic, 1),
                "model": config.asr.model,
                "language_whitelist": list(config.asr.language_whitelist),
            }
        )
        self._write_meta()

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self.started_monotonic

    def stop(self, reason: str = "stopped", queue_draft: bool = True) -> dict:
        """Flush and close everything; returns the final metadata."""
        self.audio.close()
        with self._live_lock:
            self._live.close()
        status = "complete"
        if self.audio.disk_low:
            status, reason = "stopped_low_disk", "low disk space"
        elif self.audio.error:
            status, reason = "stopped_error", f"write error: {self.audio.error}"
        self.meta.update(
            ended_at=self._now().astimezone().isoformat(timespec="seconds"),
            status=status,
            stop_reason=reason,
            duration_s=round(self.audio.duration_s, 3),
            dropped_frames=self.audio.dropped_frames,
        )
        if queue_draft and self.audio.samples_written > 0:
            self.meta["draft"] = {"state": "queued", "queued_at": time.time()}
        self._write_meta()
        return dict(self.meta)

    def _write_meta(self) -> None:
        try:
            write_json_atomic(self.directory / META_FILE, self.meta)
        except OSError:
            logger.exception("recording: cannot write %s", self.directory / META_FILE)
