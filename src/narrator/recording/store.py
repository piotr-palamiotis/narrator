"""The recordings folder: listing, status, delete/purge, corrected-SRT
upload and "Use as reference". Plain functions over a directory, so the web
API, the manager and the tests share them."""
from __future__ import annotations

import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from narrator.recording.session import (
    AUDIO_FILE,
    CORRECTED_SRT,
    DRAFT_SRT,
    LIVE_SRT,
    META_FILE,
    read_meta,
    write_json_atomic,
)
from narrator.recording.srt import format_srt, parse_srt_text

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d(-\d+)?$")
DOWNLOADABLE = (AUDIO_FILE, LIVE_SRT, DRAFT_SRT, CORRECTED_SRT)
# The reference-library tag for a human-corrected recording (narrator_tune
# real_fixtures maps "_corrected" to an objective reference).
CORRECTED_TAG_SUFFIX = "_corrected"


class RecordingNotFound(LookupError):
    pass


def valid_id(recording_id: str) -> bool:
    return bool(_ID_RE.match(recording_id or ""))


def session_dir(root: Path, recording_id: str) -> Path:
    """The folder of one recording; refuses anything that is not a session
    id (no path traversal through the API)."""
    if not valid_id(recording_id):
        raise RecordingNotFound(f"invalid recording id {recording_id!r}")
    directory = root / recording_id
    if not directory.is_dir():
        raise RecordingNotFound(f"no recording {recording_id!r}")
    return directory


def session_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and valid_id(p.name)), reverse=True)


def dir_size(directory: Path) -> int:
    total = 0
    for p in directory.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def status_of(meta: dict, files: dict[str, bool], active: bool) -> str:
    """recording / corrected / draft ready / draft pending / draft failed /
    live (only the on-air captions so far)."""
    if active:
        return "recording"
    if files.get(CORRECTED_SRT):
        return "corrected"
    if files.get(DRAFT_SRT):
        return "draft ready"
    state = (meta.get("draft") or {}).get("state")
    if state in ("queued", "running"):
        return "draft pending"
    if state == "failed":
        return "draft failed"
    return "live"


def describe(directory: Path, active_id: str | None = None) -> dict:
    meta = read_meta(directory)
    files = {name: (directory / name).is_file() for name in DOWNLOADABLE}
    active = directory.name == active_id
    draft = dict(meta.get("draft") or {"state": "none"})
    return {
        "id": directory.name,
        "started_at": meta.get("started_at"),
        "ended_at": meta.get("ended_at"),
        "duration_s": meta.get("duration_s") or 0.0,
        "size_bytes": dir_size(directory),
        "status": status_of(meta, files, active),
        "session_status": meta.get("status"),
        "stop_reason": meta.get("stop_reason"),
        "model": meta.get("model"),
        "language_whitelist": meta.get("language_whitelist") or [],
        "dropped_frames": meta.get("dropped_frames") or 0,
        "draft": draft,
        "files": files,
        "reference": meta.get("reference"),
    }


def list_recordings(root: Path, active_id: str | None = None) -> list[dict]:
    return [describe(d, active_id) for d in session_dirs(root)]


def disk_usage(root: Path) -> dict:
    from narrator.recording.session import free_disk_mb

    used = sum(dir_size(d) for d in session_dirs(root))
    probe = root if root.exists() else root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "recordings_bytes": used,
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "free_mb": round(free_disk_mb(root), 1),
    }


def update_meta(directory: Path, **changes) -> dict:
    meta = read_meta(directory)
    meta.update(changes)
    write_json_atomic(directory / META_FILE, meta)
    return meta


def delete_recording(root: Path, recording_id: str, active_id: str | None = None) -> None:
    if recording_id == active_id:
        raise PermissionError("this recording is still running; stop it first")
    directory = session_dir(root, recording_id)
    shutil.rmtree(directory)
    logger.info("recording %s deleted", recording_id)


def delete_all(root: Path, keep: set[str] = frozenset()) -> int:
    count = 0
    for directory in session_dirs(root):
        if directory.name in keep:
            continue
        shutil.rmtree(directory, ignore_errors=True)
        count += 1
    logger.info("deleted %d recording(s)", count)
    return count


def _session_time(directory: Path) -> float:
    """When the session ended (else started, else the folder's mtime), as an
    epoch timestamp."""
    meta = read_meta(directory)
    for key in ("ended_at", "started_at"):
        value = meta.get(key)
        if value:
            try:
                return datetime.fromisoformat(value).timestamp()
            except ValueError:
                pass
    try:
        return datetime.strptime(directory.name[:19], "%Y-%m-%d_%H-%M-%S").timestamp()
    except ValueError:
        return directory.stat().st_mtime


def purge_older_than(root: Path, days: float, *, now: float | None = None, keep: set[str] = frozenset()) -> list[str]:
    """Delete sessions that ended more than `days` ago (0 = keep all)."""
    if days <= 0:
        return []
    cutoff = (time.time() if now is None else now) - days * 86400
    purged = []
    for directory in session_dirs(root):
        if directory.name in keep:
            continue
        if _session_time(directory) < cutoff:
            shutil.rmtree(directory, ignore_errors=True)
            purged.append(directory.name)
    if purged:
        logger.info("retention: deleted %d recording(s) older than %g day(s)", len(purged), days)
    return purged


def save_corrected(root: Path, recording_id: str, data: bytes | str, active_id: str | None = None) -> int:
    """Validate an uploaded SRT against the recording and store it as
    corrected.srt; returns the cue count. Raises SrtError when invalid."""
    if recording_id == active_id:
        raise PermissionError("this recording is still running")
    directory = session_dir(root, recording_id)
    meta = read_meta(directory)
    duration = meta.get("duration_s") or None
    cues = parse_srt_text(data, duration_s=duration)
    tmp = directory / (CORRECTED_SRT + ".tmp")
    tmp.write_text(format_srt(cues), encoding="utf-8")
    tmp.replace(directory / CORRECTED_SRT)
    update_meta(directory, corrected={"uploaded_at": time.time(), "cues": len(cues)})
    logger.info("recording %s: corrected SRT uploaded (%d cues)", recording_id, len(cues))
    return len(cues)


def reference_title(recording_id: str) -> str:
    return f"narrator {recording_id}"


def use_as_reference(root: Path, recording_id: str, library_dir: Path, language_name: str = "Polish") -> dict:
    """Copy a corrected recording into the reference library as
    "<title> (FLAC).flac" + "<title> (<Language>_corrected).srt" (the title is
    what pairs them; real_fixtures strips the trailing "(...)" tag)."""
    directory = session_dir(root, recording_id)
    corrected = directory / CORRECTED_SRT
    audio = directory / AUDIO_FILE
    if not corrected.is_file():
        raise FileNotFoundError("upload a corrected SRT first")
    if not audio.is_file():
        raise FileNotFoundError("this recording has no audio")
    if not re.fullmatch(r"[A-Za-z][A-Za-z \-]*", language_name):
        raise ValueError(f"invalid language name {language_name!r}")
    library_dir.mkdir(parents=True, exist_ok=True)
    title = reference_title(recording_id)
    audio_dest = library_dir / f"{title} (FLAC).flac"
    srt_dest = library_dir / f"{title} ({language_name}{CORRECTED_TAG_SUFFIX}).srt"
    shutil.copyfile(audio, audio_dest)
    shutil.copyfile(corrected, srt_dest)
    info = {"copied_at": time.time(), "audio": str(audio_dest), "srt": str(srt_dest)}
    update_meta(directory, reference=info)
    logger.info("recording %s copied into the reference library (%s)", recording_id, library_dir)
    return info
