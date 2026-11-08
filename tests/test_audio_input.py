"""Channel selection, NDI audio capture and capture-class selection.
No microphone, NDI runtime or display needed: PortAudio and cyndilib are
replaced by fakes."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from pydantic import ValidationError

from narrator.audio import capture as capture_module
from narrator.audio.capture import MicCapture
from narrator.audio.devices import ndi_source_from_spec
from narrator.audio.factory import capture_class_for, create_capture
from narrator.audio.mixing import (
    ChannelSelectionError,
    FrameChunker,
    StreamResampler,
    check_channels_available,
    mix_channels,
)
from narrator.audio.ndi_capture import NdiAudioCapture, find_source, is_own_source
from narrator.config.schema import AudioConfig, RootConfig


# --- schema ---------------------------------------------------------------

def test_channels_default_is_first_channel_only():
    assert AudioConfig().channels == [1]
    assert RootConfig().audio.channels == [1]


def test_channels_accept_stereo_downmix_and_arbitrary_order():
    assert AudioConfig(channels=[1, 2]).channels == [1, 2]
    assert AudioConfig(channels=[4, 2]).channels == [4, 2]


@pytest.mark.parametrize("bad", [[], [0], [-1, 2], [1, 1], [1, 2, 2]])
def test_channels_validation_rejects(bad):
    with pytest.raises(ValidationError):
        AudioConfig(channels=bad)


def test_defaults_yaml_carries_channels():
    import yaml
    from importlib import resources

    text = (resources.files("narrator.config") / "defaults.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(text)["audio"]["channels"] == [1]


# --- mixing math ----------------------------------------------------------

def test_mix_single_channel_selects_that_channel():
    data = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=np.float32)
    assert mix_channels(data, [1]).tolist() == [1.0, 2.0]
    assert mix_channels(data, [3]).tolist() == [100.0, 200.0]


def test_mix_averages_chosen_channels_only():
    data = np.array([[1.0, 3.0, 100.0], [2.0, 4.0, 100.0]], dtype=np.float32)
    assert mix_channels(data, [1, 2]).tolist() == [2.0, 3.0]
    assert mix_channels(data, [2, 1]).tolist() == [2.0, 3.0]  # order irrelevant


def test_mix_result_is_float32_and_independent_of_source():
    data = np.ones((4, 2), dtype=np.float32)
    mono = mix_channels(data, [1])
    data[:] = 9.0
    assert mono.dtype == np.float32 and mono.tolist() == [1.0] * 4


def test_check_channels_available_errors_clearly():
    check_channels_available([1, 2], 2, "device")
    with pytest.raises(ChannelSelectionError, match=r"needs 4 input channel.*only 2"):
        check_channels_available([1, 4], 2, "input device 'X'")


# --- MicCapture (fake PortAudio) ------------------------------------------

class FakeStream:
    instances: list["FakeStream"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_portaudio(monkeypatch):
    FakeStream.instances = []
    monkeypatch.setattr(capture_module.sd, "InputStream", FakeStream)
    monkeypatch.setattr(
        capture_module.sd, "query_devices",
        lambda device=None, kind=None: {"name": "Fake 4ch", "max_input_channels": 4, "default_samplerate": 48000},
    )


def test_mic_opens_with_max_channels_and_mixes_in_callback(fake_portaudio):
    mic = MicCapture("default", frame_ms=30, channels=[2, 3])
    mic.start()
    assert FakeStream.instances[0].kwargs["channels"] == 3
    indata = np.zeros((480, 3), dtype=np.float32)
    indata[:, 0] = 1.0  # unselected channel must not leak into the mix
    indata[:, 1] = 0.2
    indata[:, 2] = 0.4
    mic._on_audio(indata, 480, None, None)
    frame = next(iter(mic.frames(timeout=0.1)))
    assert frame.shape == (480,)
    np.testing.assert_allclose(frame, 0.3, atol=1e-6)
    assert mic.level == pytest.approx(0.3, abs=1e-5)  # meter reads the mix
    mic.stop()
    assert mic.level == 0.0


def test_mic_default_is_single_channel_stream(fake_portaudio):
    mic = MicCapture("default")
    mic.start()
    assert FakeStream.instances[0].kwargs["channels"] == 1


def test_mic_start_rejects_missing_channel_with_clear_error(fake_portaudio):
    mic = MicCapture("default", channels=[1, 5])
    with pytest.raises(ChannelSelectionError, match=r"needs 5 input channel.*only 4"):
        mic.start()
    assert FakeStream.instances == []  # nothing opened


# --- resampling / chunking -------------------------------------------------

def _sine(freq, seconds, rate, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_stream_resampler_48k_to_16k_preserves_tone_and_length():
    x = _sine(440, 1.0, 48000)
    rs = StreamResampler(48000, 16000)
    # awkward chunk sizes (not multiples of 3) must not slip the phase
    out = np.concatenate([rs.process(x[i:i + 1001]) for i in range(0, len(x), 1001)])
    assert abs(len(out) - 16000) <= 2
    spectrum = np.abs(np.fft.rfft(out[1000:-1000]))
    peak_hz = np.argmax(spectrum) * 16000 / len(out[1000:-1000])
    assert abs(peak_hz - 440) < 3
    assert np.sqrt(np.mean(out[1000:-1000] ** 2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.03)


def test_stream_resampler_chunked_matches_one_shot_without_boundary_clicks():
    x = _sine(300, 0.5, 48000)
    whole = StreamResampler(48000, 16000).process(x)
    rs = StreamResampler(48000, 16000)
    chunked = np.concatenate([rs.process(x[i:i + 480]) for i in range(0, len(x), 480)])
    n = min(len(whole), len(chunked))
    assert np.max(np.abs(whole[:n] - chunked[:n])) < 0.02
    assert np.max(np.abs(np.diff(chunked))) < 0.1  # no clicks at chunk joins


def test_stream_resampler_handles_44k1_and_passthrough_16k():
    out = StreamResampler(44100, 16000).process(_sine(440, 1.0, 44100))
    assert abs(len(out) - 16000) <= 2
    same = _sine(440, 0.1, 16000)
    assert np.array_equal(StreamResampler(16000, 16000).process(same), same)


def test_frame_chunker_emits_fixed_frames_and_keeps_remainder():
    ch = FrameChunker(480)
    assert ch.push(np.zeros(300, dtype=np.float32)) == []
    frames = ch.push(np.ones(700, dtype=np.float32))
    assert [f.shape for f in frames] == [(480,), (480,)]
    assert ch.push(np.zeros(40, dtype=np.float32)) == []  # 1000 - 960 = 40 left, +40 = 80


# --- NDI capture with a fake backend --------------------------------------

class FakeStreamSource:
    """Plays a list of chunks, then (optionally) goes silent like a vanished source."""

    def __init__(self, chunks, rate=48000):
        self.chunks = list(chunks)
        self.rate = rate
        self.closed = False

    def read(self, timeout_ms):
        if self.chunks:
            return self.chunks.pop(0), self.rate
        time.sleep(timeout_ms / 1000)
        return None

    def close(self):
        self.closed = True


class FakeBackend:
    def __init__(self, script):
        # script: list of (visible_source_names, stream-or-None) per discovery round
        self.script = list(script)
        self.connected: list[str] = []
        self.rounds = 0
        self.closed = False
        self._next_stream = None

    def source_names(self, timeout_s):
        self.rounds += 1
        time.sleep(0.005)
        if self.script:
            names, stream = self.script[0]
            if stream is not None and names:
                self._next_stream = stream
                self.script.pop(0)
            elif not stream:
                self.script.pop(0) if len(self.script) > 1 else None
            return list(names)
        return []

    def connect(self, source_name):
        self.connected.append(source_name)
        return self._next_stream

    def close(self):
        self.closed = True


def _planar(freq, seconds, channels=2, rate=48000, amp=0.5, silent_channels=()):
    mono = _sine(freq, seconds, rate, amp)
    data = np.stack([np.zeros_like(mono) if c in silent_channels else mono for c in range(channels)])
    return data


def _collect(cap, want_frames, timeout=5.0):
    """Drain up to `want_frames` from the capture queue (bounded by `timeout`)."""
    import queue as _queue

    frames = []
    deadline = time.monotonic() + timeout
    while len(frames) < want_frames and time.monotonic() < deadline:
        try:
            frames.append(cap._queue.get(timeout=0.05))
        except _queue.Empty:
            pass
    return frames


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _make(backend, **kw):
    kw.setdefault("lost_after_s", 0.3)
    kw.setdefault("read_timeout_ms", 20)
    return NdiAudioCapture("MIXER (Program)", backend_factory=lambda: backend, **kw)


def test_ndi_frames_are_16k_mono_micCapture_format():
    data = _planar(440, 0.5, channels=2, silent_channels=(1,))
    chunks = [data[:, i:i + 480] for i in range(0, data.shape[1], 480)]
    backend = FakeBackend([(["MIXER (Program)"], FakeStreamSource(chunks))])
    cap = _make(backend, channels=[1, 2], frame_ms=30)
    cap.start()
    try:
        frames = _collect(cap, 10)
    finally:
        cap.stop()
    assert len(frames) >= 10
    assert all(f.shape == (480,) and f.dtype == np.float32 for f in frames)  # 30ms @ 16k
    # ch1 carries the tone, ch2 is silent: [1, 2] averages them -> amplitude 0.25
    body = np.concatenate(frames[2:])
    assert np.sqrt(np.mean(body**2)) == pytest.approx(0.25 / np.sqrt(2), rel=0.1)


def test_ndi_channel_selection_picks_requested_channel():
    data = _planar(440, 0.4, channels=2, silent_channels=(0,))  # tone only on ch2
    chunks = [data[:, i:i + 480] for i in range(0, data.shape[1], 480)]

    def run(channels):
        backend = FakeBackend([(["MIXER (Program)"], FakeStreamSource(list(chunks)))])
        cap = _make(backend, channels=channels)
        cap.start()
        try:
            frames = _collect(cap, 8)
        finally:
            cap.stop()
        return np.sqrt(np.mean(np.concatenate(frames[2:]) ** 2))

    assert run([1]) < 1e-4
    assert run([2]) == pytest.approx(0.5 / np.sqrt(2), rel=0.1)


def test_ndi_level_tracks_mix_and_drops_to_zero_when_source_stops():
    data = _planar(440, 0.3, channels=1)
    chunks = [data[:, i:i + 480] for i in range(0, data.shape[1], 480)]
    backend = FakeBackend([(["MIXER (Program)"], FakeStreamSource(chunks))])
    cap = _make(backend, lost_after_s=5.0)  # stay connected, just starve
    cap.start()
    try:
        assert _wait(lambda: cap.level > 0.1)
        assert cap.connected
        assert _wait(lambda: cap.level == 0.0, timeout=3.0)  # stalled meter returns to 0
    finally:
        cap.stop()
    assert cap.level == 0.0 and not cap.connected


def test_ndi_reconnects_after_source_disappears():
    data = _planar(440, 0.2, channels=1)
    chunks = lambda: [data[:, i:i + 480] for i in range(0, data.shape[1], 480)]
    first, second = FakeStreamSource(chunks()), FakeStreamSource(chunks())
    backend = FakeBackend([
        (["MIXER (Program)"], first),
        ([], None),                      # source gone
        ([], None),
        (["MIXER (Program)"], second),   # back
    ])
    cap = _make(backend)
    cap.start()
    try:
        assert _wait(lambda: len(backend.connected) == 2, timeout=8.0)
        assert first.closed  # the dead receiver was released
        _wait(lambda: second.chunks == [], timeout=3.0)
        frames = _collect(cap, 12, timeout=5.0)
        assert len(frames) >= 12  # frames from both sessions kept flowing
    finally:
        cap.stop()
    assert backend.closed


def test_ndi_missing_source_does_not_crash_and_waits():
    backend = FakeBackend([([], None)])
    cap = _make(backend)
    cap.start()
    try:
        assert _wait(lambda: backend.rounds >= 3)
        assert cap.level == 0.0 and not cap.connected
        assert cap._thread.is_alive()
    finally:
        cap.stop()


def test_ndi_too_few_channels_logs_error_and_stays_silent(caplog):
    data = _planar(440, 0.3, channels=1)  # mono source, but channel 2 requested
    chunks = [data[:, i:i + 480] for i in range(0, data.shape[1], 480)]
    backend = FakeBackend([(["MIXER (Program)"], FakeStreamSource(chunks))])
    cap = _make(backend, channels=[1, 2])
    with caplog.at_level("ERROR"):
        cap.start()
        try:
            assert _wait(lambda: any("needs 2 input channel" in r.message for r in caplog.records))
        finally:
            cap.stop()
    assert cap._queue.empty() and cap.level == 0.0


def test_ndi_backend_unavailable_is_logged_not_raised():
    def broken():
        raise RuntimeError("NDI runtime not installed")

    cap = NdiAudioCapture("X", backend_factory=broken)
    cap.start()
    time.sleep(0.1)
    assert cap._thread.is_alive() and cap.level == 0.0
    cap.stop()


def test_ndi_stop_unblocks_frames_generator():
    cap = _make(FakeBackend([([], None)]))
    cap.start()
    stop = threading.Event()
    done = threading.Event()

    def consume():
        for _ in cap.frames(timeout=0.05, stop_event=stop):
            pass
        done.set()

    threading.Thread(target=consume, daemon=True).start()
    stop.set()
    cap.stop()
    assert done.wait(2.0)


# --- own output exclusion / source matching ---------------------------------

def test_own_ndi_output_is_recognized_and_excluded():
    assert is_own_source("HOST (LiveCaption)", "LiveCaption")
    assert is_own_source("host (livecaption)", "LiveCaption")
    assert not is_own_source("HOST (LiveCaption 2)", "LiveCaption")
    assert not is_own_source("HOST (Mixer)", None)
    names = ["HOST (LiveCaption)", "MIXER (Program)"]
    assert find_source(names, "HOST (LiveCaption)", "LiveCaption") is None
    assert find_source(names, "mixer (program)", "LiveCaption") == "MIXER (Program)"
    assert find_source(names, "nope", "LiveCaption") is None


def test_ndi_capture_refuses_to_start_on_own_output():
    cap = NdiAudioCapture("HOST (LiveCaption)", own_stream_name="LiveCaption", backend_factory=lambda: None)
    with pytest.raises(ValueError, match="own caption output"):
        cap.start()


def test_list_ndi_sources_never_raises_without_runtime(monkeypatch):
    from narrator.audio import ndi_capture

    def boom():
        raise ImportError("no NDI")

    monkeypatch.setattr(ndi_capture, "_CyndiBackend", boom)
    assert ndi_capture.list_ndi_sources(0.01) == []


def test_list_ndi_sources_filters_own_output(monkeypatch):
    from narrator.audio import ndi_capture

    class B:
        def source_names(self, t):
            return ["HOST (LiveCaption)", "B (Cam)", "A (Mixer)"]

        def close(self):
            pass

    monkeypatch.setattr(ndi_capture, "_CyndiBackend", B)
    assert ndi_capture.list_ndi_sources(0.01, "LiveCaption") == ["A (Mixer)", "B (Cam)"]


# --- capture class selection -----------------------------------------------

def test_ndi_device_spec_parsing():
    assert ndi_source_from_spec("ndi:MIXER (Program)") == "MIXER (Program)"
    assert ndi_source_from_spec("  NDI: Cam 1 ") == "Cam 1"
    assert ndi_source_from_spec("default") is None
    assert ndi_source_from_spec("USB Audio") is None


def test_capture_class_for_device_spec():
    assert capture_class_for("ndi:MIXER (Program)") is NdiAudioCapture
    assert capture_class_for("default") is MicCapture
    assert capture_class_for("HDA Intel") is MicCapture


def test_create_capture_wires_channels_and_own_name():
    cfg = AudioConfig(device="ndi:MIXER (Program)", channels=[1, 2], frame_ms=20)
    cap = create_capture(cfg, own_ndi_name="LiveCaption")
    assert isinstance(cap, NdiAudioCapture)
    assert cap._channels == [1, 2] and cap._own == "LiveCaption" and cap._source_name == "MIXER (Program)"
    assert cap._frame_samples == 320

    mic = create_capture(AudioConfig(device="default", channels=[2]))
    assert isinstance(mic, MicCapture) and mic._channels == [2]


def test_controller_starts_capture_chosen_by_device(monkeypatch):
    from narrator import app_controller
    from narrator.app_controller import AppController

    created = []

    class FakeCapture:
        def __init__(self, kind):
            self.kind = kind
            self.level = 0.0
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            pass

        def frames(self, timeout=1.0, stop_event=None):
            return iter(())

    def fake_create(audio, own_ndi_name=None):
        cap = FakeCapture("ndi" if audio.device.startswith("ndi:") else "mic")
        created.append((cap, audio, own_ndi_name))
        return cap

    monkeypatch.setattr(app_controller, "create_capture", fake_create)

    controller = AppController.__new__(AppController)
    controller._audio_queue = None
    for device, expected in [("ndi:MIXER (Program)", "ndi"), ("default", "mic")]:
        cfg = RootConfig(audio={"device": device, "channels": [1, 2]}, ndi={"source_name": "MyCaptions"})
        controller._config = cfg
        controller._start_audio_capture()
        controller._audio_stop_event.set()
        controller._audio_thread.join(timeout=2)
        cap, audio, own = created[-1]
        assert cap.kind == expected and cap.started and controller._mic is cap
        assert audio.channels == [1, 2] and own == "MyCaptions"
