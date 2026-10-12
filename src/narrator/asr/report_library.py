"""Autoconfigure run reports.

Every `narrator --autoconfigure` run writes one self-contained JSON summary
to scripts/autoconfigure_reports/: every configuration it evaluated, with
its scores per excerpt set (pooled WER, the substitution/deletion/insertion
breakdown, caption delay, worker load, the most frequent inserted phrases),
the offline ceiling, the decision and the reason for it. The raw captions
behind those scores stay in the result store
(scripts/autoconfigure_cache/results.jsonl). Both are plain JSON, ready for
any later analysis -- `narrator --analyze-reports` hands the summaries to a
local Ollama model.

A manifest next to the reports records which audio files each run saw, so a
gradually growing library shows which recordings are new. Files are keyed
by name relative to the fixture folder (moving the repository does not make
every file "new") and identified by a cheap content fingerprint (size plus
the first and last 64 KiB) -- hashing all ~6 GB of audio on every run took
minutes for no benefit.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from narrator.asr.autoconfigure import MANIFEST_PATH, REPORTS_DIR
from narrator.asr.library_eval import audio_fingerprint

logger = logging.getLogger(__name__)


def audio_files_in(directory: Path | None) -> list[Path]:
    """All audio fixture files in `directory` (sorted)."""
    from narrator.asr.real_fixtures import AUDIO_EXTENSIONS

    if directory is None or not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()
    )


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """The report manifest, or an empty skeleton when there is none (or it
    predates relative keys and fingerprints -- such entries are ignored)."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"audio_files": {}, "reports": []}
    manifest.setdefault("audio_files", {})
    manifest.setdefault("reports", [])
    return manifest


def save_manifest(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    """Atomically persist the manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def detect_new_audio(
    paths: list[Path], manifest: dict[str, Any], base: Path
) -> tuple[list[str], list[str]]:
    """(new, changed) file names relative to `base`: not in the manifest at
    all, or present with a different fingerprint."""
    known = manifest.get("audio_files", {})
    new: list[str] = []
    changed: list[str] = []
    for path in paths:
        key = str(path.relative_to(base))
        entry = known.get(key)
        if entry is None or "fingerprint" not in entry:
            new.append(key)
        elif entry["fingerprint"] != audio_fingerprint(path):
            changed.append(key)
    return new, changed


def register_audio_files(paths: list[Path], manifest: dict[str, Any], base: Path) -> None:
    """Record the files a run saw (first/last seen, fingerprint, size)."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = manifest.setdefault("audio_files", {})
    for path in paths:
        key = str(path.relative_to(base))
        previous = files.get(key, {})
        files[key] = {
            "fingerprint": audio_fingerprint(path),
            "size_bytes": path.stat().st_size,
            "first_seen": previous.get("first_seen", now),
            "last_seen": now,
        }


def save_summary(
    report: dict[str, Any],
    fixtures_dir: Path | None,
    reports_dir: Path = REPORTS_DIR,
    manifest_path: Path = MANIFEST_PATH,
) -> Path:
    """Write one run's summary report and update the manifest. Returns the
    report path."""
    manifest = load_manifest(manifest_path)
    paths = audio_files_in(fixtures_dir)
    if fixtures_dir is not None and paths:
        new, changed = detect_new_audio(paths, manifest, fixtures_dir)
        report = {**report, "new_audio": new, "changed_audio": changed}
        if new:
            logger.info("%d audio file(s) are new since the last run", len(new))
        register_audio_files(paths, manifest, fixtures_dir)
    # Drop entries from the pre-fingerprint manifest format (absolute keys).
    manifest["audio_files"] = {
        k: v for k, v in manifest["audio_files"].items() if "fingerprint" in v
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    path = reports_dir / f"{now.strftime('%Y%m%dT%H%M%S')}_summary.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    decision = report.get("decision", {})
    manifest["reports"].append(
        {
            "timestamp": now.isoformat(timespec="seconds"),
            "filename": path.name,
            "written": decision.get("written"),
            "winner_model": (decision.get("winner") or {}).get("model"),
            "audio_files_count": len(paths),
        }
    )
    save_manifest(manifest, manifest_path)
    logger.info("saved autoconfigure report: %s", path)
    return path
