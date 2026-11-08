"""Session recording (narrator.recording): GPU-, mic- and NDI-free."""
from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from narrator.app_controller import AppController
from narrator.asr.events import CaptionEvent, CaptionEventType
from narrator.asr.offline import planned_speech_spans, segments_to_cues, speech_spans
from narrator.asr.worker import _BusyMeter
from narrator.config.schema import AsrDevice, RootConfig
from narrator.recording import store
from narrator.recording.draft import build_request, generate_draft
from narrator.recording.manager import RecordingManager, draft_gate
from narrator.recording.session import (
    AudioRecorder,
    RecordingError,
    RecordingSession,
    read_meta,
    write_json_atomic,
)
from narrator.recording.srt import SrtError, parse_srt_text
from narrator.render.canvas import CaptionRenderer

soundfile = pytest.importorskip("soundfile")

FRAME = np.full(480, 0.1, dtype=np.float32)  # 30 ms at 16 kHz
PLENTY_MB = lambda path: 1e9  # noqa: E731


def make_config(tmp_path: Path, **recording) -> RootConfig:
    return RootConfig.model_validate(
        {
            "asr": {"model": "small", "language_whitelist": ["pl"]},
            "recording": {"directory": str(tmp_path / "rec"), "min_free_disk_mb": 100, **recording},
            "captions": {"blocklist": ["kurwa"]},
        }
    )


class BlockingFile:
    """A file whose write() blocks until released (a stalled disk)."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.written = 0

    def write(self, data) -> None:
        self.release.wait(10)
        self.written += len(data)

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ writer


def test_recorder_never_blocks_the_feed_and_counts_drops(tmp_path):
    blocking = BlockingFile()
    recorder = AudioRecorder(
        tmp_path / "a.flac", 16000, max_queue_frames=50, file_factory=lambda p, r: blocking, free_mb=PLENTY_MB
    )
    started = time.monotonic()
    accepted = sum(recorder.offer(FRAME) for _ in range(2000))
    elapsed = time.monotonic() - started
    assert elapsed < 0.5  # 2000 offers against a stalled disk: no waiting
    assert accepted <= 50 + 200 + 1  # queue bound (+ one batch in the writer's hands)
    assert recorder.dropped_frames == 2000 - accepted > 0
    blocking.release.set()
    recorder.close()


def test_feed_thread_keeps_feeding_asr_while_recording_disk_stalls(tmp_path):
    """AppController's feed thread with a recording whose disk is stalled:
    every frame still reaches the ASR queue, the recording drops and counts."""
    blocking = BlockingFile()
    config = make_config(tmp_path)

    def factory(root, cfg, **kwargs):
        return RecordingSession(root, cfg, file_factory=lambda p, r: blocking, **kwargs)

    manager = RecordingManager(lambda: config, session_factory=factory, free_mb=PLENTY_MB, flac_available=lambda: True)
    manager.start()
    controller = AppController.__new__(AppController)
    controller._audio_queue = queue.Queue(maxsize=5000)
    controller._recordings = manager

    class FakeMic:
        def frames(self, timeout, stop_event):
            for _ in range(3000):
                yield FRAME

    started = time.monotonic()
    controller._feed_audio_loop(FakeMic(), threading.Event())
    assert time.monotonic() - started < 2.0
    assert controller._audio_queue.qsize() == 3000
    assert manager._session.audio.dropped_frames > 0
    blocking.release.set()
    manager.stop()


# ------------------------------------------------------------------ session layout


def test_session_layout_audio_live_srt_and_metadata(tmp_path):
    config = make_config(tmp_path)
    clock = [1000.0]
    session = RecordingSession(
        tmp_path / "rec", config, clock=lambda: clock[0],
        now=lambda: datetime(2026, 10, 16, 18, 0, 0), free_mb=PLENTY_MB,
    )
    assert session.id == "2026-10-16_18-00-00"
    for _ in range(100):  # 3 s
        session.feed(FRAME)
    session.on_air(CaptionEvent(CaptionEventType.FINAL, "Dzień dobry państwu.", timestamp=1002.0, start=1000.5, end=1001.8))
    clock[0] = 1003.0
    session.on_air_text("Ręcznie wpisany tekst")
    meta = session.stop()

    directory = tmp_path / "rec" / "2026-10-16_18-00-00"
    info = soundfile.info(str(directory / "audio.flac"))
    assert (info.samplerate, info.channels, info.frames) == (16000, 1, 48000)
    audio, _ = soundfile.read(str(directory / "audio.flac"), dtype="float32")
    assert np.allclose(audio, 0.1, atol=1e-4)
    live = (directory / "live.srt").read_text(encoding="utf-8")
    assert "00:00:00,500 --> 00:00:01,800\nDzień dobry państwu." in live
    assert "Ręcznie wpisany tekst" in live
    saved = read_meta(directory)
    assert saved == meta
    assert saved["status"] == "complete"
    assert saved["model"] == "small" and saved["language_whitelist"] == ["pl"]
    assert saved["duration_s"] == pytest.approx(3.0)
    assert saved["draft"]["state"] == "queued"
    assert store.describe(directory)["status"] == "draft pending"


def test_status_progression(tmp_path):
    directory = tmp_path / "2026-10-16_18-00-00"
    directory.mkdir()
    write_json_atomic(directory / "session.json", {"draft": {"state": "none"}})
    assert store.describe(directory)["status"] == "live"
    assert store.describe(directory, active_id=directory.name)["status"] == "recording"
    store.update_meta(directory, draft={"state": "failed", "error": "x"})
    assert store.describe(directory)["status"] == "draft failed"
    (directory / "draft.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n")
    assert store.describe(directory)["status"] == "draft ready"
    (directory / "corrected.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n")
    assert store.describe(directory)["status"] == "corrected"


# ------------------------------------------------------------------ SRT upload validation

GOOD_SRT = "﻿1\r\n00:00:01,000 --> 00:00:02,500\r\nDzień dobry\r\nwszystkim\r\n\r\n2\r\n00:00:03,000 --> 00:00:04,000\r\nTak.\r\n"


def test_srt_parses_bom_crlf_multiline():
    cues = parse_srt_text(GOOD_SRT.encode("utf-8"), duration_s=10)
    assert [(c.start, c.end, c.text) for c in cues] == [(1.0, 2.5, "Dzień dobry\nwszystkim"), (3.0, 4.0, "Tak.")]


@pytest.mark.parametrize(
    "content, fragment",
    [
        (b"\xff\xfe garbage", "UTF-8"),
        (b"", "no subtitle cues"),
        (b"hello world\n", "expected a cue number"),
        (b"1\n00:00:01,000 -> 00:00:02,000\nx\n", "expected 'HH:MM:SS"),
        (b"1\n00:00:05,000 --> 00:00:02,000\nx\n", "ends before it starts"),
        (b"1\n00:00:05,000 --> 00:00:06,000\nx\n\n2\n00:00:01,000 --> 00:00:02,000\ny\n", "out of order"),
        (b"1\n01:00:00,000 --> 01:00:02,000\nx\n", "past the end of the recording"),
        (b"1\n00:00:01,000 --> 00:00:02,000\n\n", "no subtitle cues"),
    ],
)
def test_srt_validation_errors(content, fragment):
    with pytest.raises(SrtError, match=fragment):
        parse_srt_text(content, duration_s=60)


def test_save_corrected_and_use_as_reference(tmp_path):
    root = tmp_path / "rec"
    directory = root / "2026-10-16_18-00-00"
    directory.mkdir(parents=True)
    write_json_atomic(directory / "session.json", {"duration_s": 10.0})
    soundfile.write(str(directory / "audio.flac"), np.zeros(16000, dtype=np.float32), 16000, format="FLAC")
    with pytest.raises(FileNotFoundError):
        store.use_as_reference(root, directory.name, tmp_path / "lib")
    with pytest.raises(SrtError):
        store.save_corrected(root, directory.name, b"1\n00:05:00,000 --> 00:05:01,000\nx\n")
    assert store.save_corrected(root, directory.name, GOOD_SRT.encode("utf-8")) == 2
    stored = (directory / "corrected.srt").read_text(encoding="utf-8")
    assert stored.startswith("1\n00:00:01,000 --> 00:00:02,500\nDzień dobry\nwszystkim\n")
    with pytest.raises(PermissionError):
        store.save_corrected(root, directory.name, GOOD_SRT, active_id=directory.name)

    info = store.use_as_reference(root, directory.name, tmp_path / "lib")
    names = sorted(p.name for p in (tmp_path / "lib").iterdir())
    assert names == ["narrator 2026-10-16_18-00-00 (FLAC).flac", "narrator 2026-10-16_18-00-00 (Polish_corrected).srt"]
    assert read_meta(directory)["reference"]["srt"] == info["srt"]


def test_session_ids_cannot_escape_the_folder(tmp_path):
    for bad in ("..", "../etc", "2026-10-16_18-00-00/../../x", "", "abc"):
        with pytest.raises(store.RecordingNotFound):
            store.session_dir(tmp_path, bad)


# ------------------------------------------------------------------ purge / retention / delete


def _fake_session(root: Path, name: str, ended: datetime) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    write_json_atomic(directory / "session.json", {"ended_at": ended.isoformat()})
    return directory


def test_retention_purge_keeps_recent_and_active(tmp_path):
    now = datetime(2026, 10, 16, 12, 0, 0)
    old = _fake_session(tmp_path, "2026-09-01_10-00-00", now - timedelta(days=40))
    active = _fake_session(tmp_path, "2026-09-02_10-00-00", now - timedelta(days=39))
    recent = _fake_session(tmp_path, "2026-10-15_10-00-00", now - timedelta(days=1))
    assert store.purge_older_than(tmp_path, 0, now=now.timestamp()) == []  # 0 = keep forever
    purged = store.purge_older_than(tmp_path, 30, now=now.timestamp(), keep={active.name})
    assert purged == [old.name]
    assert active.exists() and recent.exists() and not old.exists()


def test_delete_and_delete_all(tmp_path):
    a = _fake_session(tmp_path, "2026-10-01_10-00-00", datetime.now())
    b = _fake_session(tmp_path, "2026-10-02_10-00-00", datetime.now())
    c = _fake_session(tmp_path, "2026-10-03_10-00-00", datetime.now())
    with pytest.raises(PermissionError):
        store.delete_recording(tmp_path, c.name, active_id=c.name)
    store.delete_recording(tmp_path, a.name)
    assert store.delete_all(tmp_path, keep={c.name}) == 1
    assert not a.exists() and not b.exists() and c.exists()


# ------------------------------------------------------------------ disk guard


def test_recording_refuses_to_start_below_min_free_disk(tmp_path):
    config = make_config(tmp_path, min_free_disk_mb=500)
    manager = RecordingManager(lambda: config, free_mb=lambda p: 100.0, flac_available=lambda: True)
    with pytest.raises(RecordingError, match="100 MB free"):
        manager.start()
    assert not manager.recording


def test_recording_stops_when_disk_runs_low(tmp_path):
    config = make_config(tmp_path, min_free_disk_mb=500)
    free = [10_000.0]
    changes = []
    manager = RecordingManager(
        lambda: config, free_mb=lambda p: free[0], flac_available=lambda: True, on_change=lambda: changes.append(1)
    )
    session_id = manager.start()
    free[0] = 100.0  # the disk fills up
    manager.feed(FRAME)
    deadline = time.monotonic() + 5
    while not manager._disk_low.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    manager.tick()
    assert not manager.recording
    meta = read_meta(manager.root / session_id)
    assert meta["status"] == "stopped_low_disk"
    assert manager.status()["last_stop_reason"] == "low disk space"
    assert len(changes) == 2  # started, stopped


def test_interrupted_session_is_recovered_on_startup(tmp_path):
    config = make_config(tmp_path)
    directory = tmp_path / "rec" / "2026-10-16_18-00-00"
    directory.mkdir(parents=True)
    soundfile.write(str(directory / "audio.flac"), np.zeros(32000, dtype=np.float32), 16000, format="FLAC")
    write_json_atomic(directory / "session.json", {"status": "recording", "draft": {"state": "none"}})
    manager = RecordingManager(lambda: config, flac_available=lambda: True)
    manager._recover_interrupted()
    meta = read_meta(directory)
    assert meta["status"] == "interrupted"
    assert meta["duration_s"] == pytest.approx(2.0)
    assert meta["draft"]["state"] == "queued"


# ------------------------------------------------------------------ draft scheduling


class FakeProc:
    _next_pid = 90000

    def __init__(self, command) -> None:
        self.command = command
        self.pid = FakeProc._next_pid = FakeProc._next_pid + 1
        self.returncode = None
        self.signals: list[int] = []
        self.stdout = None

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_draft_gate():
    busy = lambda b, active=True, device=AsrDevice.AUTO: {"active": active, "busy": b, "device": device}  # noqa: E731
    assert draft_gate(busy(0.9, active=False), AsrDevice.AUTO, 0.3)[0]
    assert draft_gate(busy(0.1), AsrDevice.AUTO, 0.3)[0]
    assert not draft_gate(busy(0.5), AsrDevice.AUTO, 0.3)[0]
    assert not draft_gate(busy(None), AsrDevice.AUTO, 0.3)[0]  # load not measured yet
    assert not draft_gate(busy(0.0), AsrDevice.AUTO, 0.0)[0]  # 0 = only while ASR is stopped
    assert draft_gate(busy(0.9, device=AsrDevice.VULKAN), AsrDevice.CPU, 0.3)[0]  # different device
    assert not draft_gate(busy(0.9, device=AsrDevice.CPU), AsrDevice.CPU, 0.3)[0]


def _queued_recording(manager: RecordingManager, name: str) -> Path:
    directory = manager.root / name
    directory.mkdir(parents=True)
    soundfile.write(str(directory / "audio.flac"), np.zeros(1600, dtype=np.float32), 16000, format="FLAC")
    write_json_atomic(directory / "session.json", {"language_whitelist": ["pl"], "draft": {"state": "queued", "queued_at": 1.0}})
    return directory


def test_draft_scheduler_waits_for_busy_asr_and_suspends(tmp_path):
    config = make_config(tmp_path, draft_model="small")
    load = {"active": True, "busy": 0.8, "device": AsrDevice.AUTO}
    launched: list[FakeProc] = []
    manager = RecordingManager(
        lambda: config, asr_load=lambda: dict(load), flac_available=lambda: True,
        launcher=lambda cmd: launched.append(FakeProc(cmd)) or launched[-1],
    )
    directory = _queued_recording(manager, "2026-10-16_18-00-00")

    manager.tick()
    assert launched == []  # live ASR busy: the draft waits
    assert "busy 80%" in manager.draft_wait_reason
    assert manager.list()[0]["status"] == "draft pending"

    load["busy"] = 0.1
    manager.tick()
    assert len(launched) == 1
    assert launched[0].command[-1] == str(directory)
    request = json.loads((directory / "draft_request.json").read_text())
    assert request["model"] == "small" and request["language_whitelist"] == ["pl"]
    assert read_meta(directory)["draft"]["state"] == "running"

    import signal

    load["busy"] = 0.9  # speech resumes: the running draft gives the GPU back
    manager.tick()
    assert launched[0].signals == [signal.SIGSTOP]
    load["busy"] = 0.25  # still above the resume threshold (0.3 * 0.7)
    manager.tick()
    assert launched[0].signals == [signal.SIGSTOP]
    load["active"] = False
    manager.tick()
    assert launched[0].signals == [signal.SIGSTOP, signal.SIGCONT]

    (directory / "draft.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n")
    launched[0].returncode = 0
    manager.tick()
    assert read_meta(directory)["draft"]["state"] == "ready"
    assert manager.list()[0]["status"] == "draft ready"


def test_generate_draft_now_bypasses_the_gate(tmp_path):
    config = make_config(tmp_path)
    launched: list[FakeProc] = []
    manager = RecordingManager(
        lambda: config, asr_load=lambda: {"active": True, "busy": 1.0, "device": AsrDevice.AUTO},
        flac_available=lambda: True, launcher=lambda cmd: launched.append(FakeProc(cmd)) or launched[-1],
    )
    directory = _queued_recording(manager, "2026-10-16_18-00-00")
    store.update_meta(directory, draft={"state": "failed", "error": "boom"})
    manager.request_draft(directory.name, force=True)
    manager.tick()
    assert len(launched) == 1
    manager.tick()  # forced drafts are never suspended
    assert launched[0].signals == []
    launched[0].returncode = 1  # failed without writing draft.srt
    manager.tick()
    assert read_meta(directory)["draft"]["state"] == "failed"


def test_draft_bypasses_blocklist_and_text_filters(tmp_path):
    config = RootConfig.model_validate(
        {
            "captions": {"blocklist": ["kurwa"], "blocklist_builtin": True},
            "asr": {"text_filters": [{"pattern": "dzień", "replacement": "NOC"}], "language_whitelist": ["pl"]},
        }
    )
    request = build_request(config, ["pl"], ["Kowalski"])
    assert "blocklist" not in json.dumps(request) and "text_filters" not in request
    directory = tmp_path / "s"
    directory.mkdir()
    audio = np.zeros(16000 * 4, dtype=np.float32)
    soundfile.write(str(directory / "audio.flac"), audio, 16000, format="FLAC")

    class FakeVad:
        def process(self, frame):  # speech everywhere
            return [0.9] * (len(frame) // 512)

    class Segment:
        def __init__(self, t0, t1, text):
            self.t0, self.t1, self.text = t0, t1, text

    calls = []

    def transcribe(backend, chunk, language, prompt):
        calls.append((language, prompt))
        return [Segment(0, 150, "Dzień dobry, kurwa mać."), Segment(150, 300, "tak tak tak")]

    count = generate_draft(
        directory, request, backend_factory=lambda r: object(), vad_factory=lambda r: FakeVad(),
        transcribe=transcribe, log=lambda line: None,
    )
    draft = (directory / "draft.srt").read_text(encoding="utf-8")
    assert "Dzień dobry, kurwa mać." in draft  # raw: no blocklist, no text filter
    assert "\ntak\n" in draft  # whisper's repetition loop collapsed
    assert count == 2
    assert calls == [("pl", "Kowalski.")]  # the vocabulary prompt (build_initial_prompt)


# ------------------------------------------------------------------ offline span planning


def test_planned_spans_cut_long_speech_at_pauses():
    window = 512 / 16000
    n = int(150 / window)  # 150 s
    probs = np.full(n, 0.9, dtype=np.float32)
    pause = slice(int(50 / window), int(51 / window))  # a 1 s pause at 50 s
    probs[pause] = 0.0
    probs[int(140 / window):] = 0.0  # trailing silence
    spans = planned_speech_spans(probs, window, 150.0)
    assert all(e - s <= 60.0 + 1e-6 for s, e in spans)
    assert spans[0][1] == pytest.approx(50.5, abs=0.1)  # cut in the pause, not at 60 s
    assert spans[-1][1] == pytest.approx(140.3, abs=0.1)  # silence is not decoded


def test_speech_spans_shared_with_eval_transcript():
    assert speech_spans([(10.0, 20.0)], 100.0, max_span_s=60.0) == [(0.0, 9.7), (20.3, 80.3), (80.3, 100.0)]


def test_segments_to_cues_split_and_dedupe():
    long_text = " ".join(f"słowo{i}" for i in range(40))
    cues = segments_to_cues([(0.0, 10.0, long_text), (10.0, 12.0, long_text)])
    assert len(cues) >= 3 and all(len(text) <= 84 for _, _, text in cues)
    assert cues[-1][1] <= 10.0 + 1e-6  # the repeated segment was dropped


# ------------------------------------------------------------------ worker busy fraction, NDI dot


def test_busy_meter_publishes_fraction():
    class Value:
        value = -1.0

    now = [0.0]
    value = Value()
    meter = _BusyMeter(value, clock=lambda: now[0])
    meter.add(1.0)
    assert value.value == -1.0  # window not complete
    now[0] = 5.0
    meter.add(1.0)
    assert value.value == pytest.approx(0.4)


def test_on_air_dot_in_render():
    config = RootConfig()
    renderer = CaptionRenderer(config.canvas, config.layout, config.style)
    w, h = config.canvas.width_px, config.canvas.height_px
    probe = (w - round(h * 0.015) - round(h * 0.008), round(h * 0.015) + round(h * 0.008))
    assert renderer.get_frame().getpixel(probe)[3] == 0
    renderer.set_recording_indicator(True)
    r, g, b, a = renderer.get_frame().getpixel(probe)
    assert r > 200 and g < 60 and a > 200
    renderer.set_recording_indicator(False)
    assert renderer.get_frame().getpixel(probe)[3] == 0


def test_controller_sends_only_on_air_captions_to_the_recording(tmp_path):
    from narrator.app_controller import FeatureUnavailableError

    class Recorder:
        recording = True
        available = True

        def __init__(self):
            self.finals, self.texts = [], []

        def on_air_final(self, event):
            self.finals.append(event.text)

        def on_air_text(self, text):
            self.texts.append(text)

    class Renderer:
        def append_final_text(self, text): pass
        def set_partial_preview(self, text): pass
        def set_recording_indicator(self, on): self.dot = on

    controller = AppController.__new__(AppController)
    controller._renderer = Renderer()
    controller._caption_mode = "auto"
    controller._transcript = None
    controller._transcript_lock = threading.Lock()
    controller._recordings = recorder = Recorder()
    controller._handle_caption_event(CaptionEvent(CaptionEventType.FINAL, "na antenie"))
    controller.set_captions_paused(True)
    controller._handle_caption_event(CaptionEvent(CaptionEventType.FINAL, "wstrzymane"))
    controller.submit_manual_text("ręcznie")
    assert recorder.finals == ["na antenie"] and recorder.texts == ["ręcznie"]

    controller._config = make_config(tmp_path, on_air_dot=True)
    controller._update_recording_indicator()
    assert controller._renderer.dot is True

    controller._recordings = None
    with pytest.raises(FeatureUnavailableError):
        controller.set_recording(True)
