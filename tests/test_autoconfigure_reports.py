"""Autoconfigure run reports and the audio manifest (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

from narrator.asr.report_library import (
    audio_files_in,
    detect_new_audio,
    load_manifest,
    register_audio_files,
    save_manifest,
    save_summary,
)


def make_audio(directory: Path, name: str, content: bytes = b"audio") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def test_audio_files_in_filters_audio_extensions(tmp_path):
    make_audio(tmp_path, "speech.wav")
    make_audio(tmp_path, "speech.m4a")
    make_audio(tmp_path, "notes.txt")
    assert [p.name for p in audio_files_in(tmp_path)] == ["speech.m4a", "speech.wav"]
    assert audio_files_in(tmp_path / "missing") == []
    assert audio_files_in(None) == []


def test_manifest_roundtrip_and_empty_default(tmp_path):
    path = tmp_path / "manifest.json"
    assert load_manifest(path) == {"audio_files": {}, "reports": []}
    save_manifest({"audio_files": {"a.wav": {"fingerprint": "x"}}, "reports": []}, path)
    assert load_manifest(path)["audio_files"] == {"a.wav": {"fingerprint": "x"}}


def test_new_and_changed_audio_are_detected_by_relative_name(tmp_path):
    fixtures = tmp_path / "fixtures"
    first = make_audio(fixtures, "a.wav", b"one")
    manifest = load_manifest(tmp_path / "none.json")
    new, changed = detect_new_audio([first], manifest, fixtures)
    assert (new, changed) == (["a.wav"], [])

    register_audio_files([first], manifest, fixtures)
    assert detect_new_audio([first], manifest, fixtures) == ([], [])

    first.write_bytes(b"different content")
    second = make_audio(fixtures, "b.wav", b"two")
    new, changed = detect_new_audio([first, second], manifest, fixtures)
    assert new == ["b.wav"] and changed == ["a.wav"]


def test_moving_the_library_does_not_make_files_new(tmp_path):
    old = tmp_path / "old"
    path = make_audio(old, "a.wav", b"same bytes")
    manifest = load_manifest(tmp_path / "none.json")
    register_audio_files([path], manifest, old)
    moved = make_audio(tmp_path / "new", "a.wav", b"same bytes")
    assert detect_new_audio([moved], manifest, tmp_path / "new") == ([], [])


def test_save_summary_writes_report_and_updates_manifest(tmp_path):
    fixtures = tmp_path / "fixtures"
    make_audio(fixtures, "a.wav")
    reports = tmp_path / "reports"
    manifest_path = reports / "manifest.json"
    report = {"decision": {"written": True, "winner": {"model": "large-v3-turbo"}}}
    path = save_summary(report, fixtures, reports_dir=reports, manifest_path=manifest_path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["new_audio"] == ["a.wav"]
    manifest = load_manifest(manifest_path)
    assert manifest["reports"][-1]["winner_model"] == "large-v3-turbo"
    assert "a.wav" in manifest["audio_files"]

    second = save_summary(report, fixtures, reports_dir=reports, manifest_path=manifest_path)
    assert json.loads(second.read_text(encoding="utf-8"))["new_audio"] == []


def test_legacy_absolute_manifest_entries_are_dropped(tmp_path):
    reports = tmp_path / "reports"
    manifest_path = reports / "manifest.json"
    save_manifest(
        {"audio_files": {"/old/abs/path.m4a": {"sha256": "abc"}}, "reports": []}, manifest_path
    )
    save_summary({"decision": {}}, None, reports_dir=reports, manifest_path=manifest_path)
    assert load_manifest(manifest_path)["audio_files"] == {}
