"""NDI audio as a capture source (`audio.device: "ndi:<source name>"`).

Same surface as `MicCapture` (`start()`, `stop()`, `level`, `frames()`) so the
feed thread, the GUI level meter and `AppController._restart_audio_capture`
need no special cases. NDI delivers planar float audio (channels x samples,
usually 48 kHz); each chunk goes through the shared `audio.channels` mix,
a streaming resample to 16 kHz, and is re-sliced into `frame_ms` frames, i.e.
exactly what MicCapture puts on its queue.

A background thread owns discovery and receiving. A source that vanishes (no
audio for `lost_after_s`) or never existed yet is not an error: the level
drops to 0, the situation is logged once, and discovery runs again until the
source (re)appears. narrator's own NDI output is never offered or connected.

All cyndilib access sits behind `_CyndiBackend` so tests drive the capture
with a fake backend (no NDI runtime needed).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Protocol

import numpy as np

from narrator.audio.mixing import (
    FrameChunker,
    StreamResampler,
    check_channels_available,
    mix_channels,
    required_channels,
)

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def is_own_source(source_name: str, own_stream_name: str | None) -> bool:
    """NDI source names are `HOST (stream name)`; narrator's output is the
    one whose stream name equals `ndi.source_name`."""
    if not own_stream_name:
        return False
    own = own_stream_name.strip().lower()
    name = source_name.strip().lower()
    return name == own or name.endswith(f"({own})")


def find_source(names: Sequence[str], wanted: str, own_stream_name: str | None = None) -> str | None:
    """Pick the discovered source matching `wanted` (exact, then
    case-insensitive), never narrator's own output."""
    candidates = [n for n in names if not is_own_source(n, own_stream_name)]
    if wanted in candidates:
        return wanted
    low = wanted.strip().lower()
    for n in candidates:
        if n.strip().lower() == low:
            return n
    return None


class NdiAudioStream(Protocol):
    def read(self, timeout_ms: int) -> tuple[np.ndarray, int] | None:
        """One chunk as (planar float32 array [channels, samples], sample rate),
        or None when nothing arrived within the timeout."""

    def close(self) -> None: ...


class NdiBackend(Protocol):
    def source_names(self, timeout_s: float) -> list[str]: ...

    def connect(self, source_name: str) -> NdiAudioStream: ...

    def close(self) -> None: ...


class _CyndiStream:
    def __init__(self, receiver, audio_frame, receive_audio_flag) -> None:
        self._receiver = receiver
        self._frame = audio_frame
        self._flag = receive_audio_flag

    def read(self, timeout_ms: int) -> tuple[np.ndarray, int] | None:
        self._receiver.receive(self._flag, timeout_ms)
        if not self._frame.get_buffer_depth():
            return None
        result = self._frame.get_all_read_data()
        if result is None:
            return None
        data = result[0]
        if data.size == 0:
            return None
        return data, int(self._frame.sample_rate)

    def close(self) -> None:
        try:
            self._receiver.disconnect()
        except Exception:
            logger.debug("NDI receiver disconnect failed", exc_info=True)


class _CyndiBackend:
    """Finder + Receiver from cyndilib. Imported lazily so a missing NDI
    runtime only fails NDI sources, never plain microphone use."""

    def __init__(self) -> None:
        from cyndilib import Finder

        self._finder = Finder()
        self._finder.open()

    def source_names(self, timeout_s: float) -> list[str]:
        self._finder.wait_for_sources(timeout_s)
        return list(self._finder.get_source_names())

    def connect(self, source_name: str) -> NdiAudioStream:
        from cyndilib import AudioRecvFrame, Receiver
        from cyndilib.receiver import ReceiveFrameType
        from cyndilib.wrapper import RecvBandwidth

        source = self._finder.get_source(source_name)
        receiver = Receiver(bandwidth=RecvBandwidth.audio_only, recv_name="narrator-audio")
        frame = AudioRecvFrame()
        receiver.set_audio_frame(frame)
        receiver.set_source(source)
        return _CyndiStream(receiver, frame, ReceiveFrameType.recv_audio)

    def close(self) -> None:
        try:
            self._finder.close()
        except Exception:
            logger.debug("NDI finder close failed", exc_info=True)


def list_ndi_sources(timeout_s: float = 1.5, own_stream_name: str | None = None) -> list[str]:
    """Discover NDI sources for a device picker / `--list-devices`. Returns
    [] (never raises) when cyndilib or the NDI runtime is unavailable."""
    try:
        backend = _CyndiBackend()
    except Exception as exc:
        logger.debug("NDI discovery unavailable: %s", exc)
        return []
    try:
        names = backend.source_names(timeout_s)
    except Exception as exc:
        logger.debug("NDI discovery failed: %s", exc)
        names = []
    finally:
        backend.close()
    return sorted(n for n in names if not is_own_source(n, own_stream_name))


class NdiAudioCapture:
    def __init__(
        self,
        source_name: str,
        frame_ms: int = 30,
        channels: Sequence[int] = (1,),
        own_stream_name: str | None = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        max_queued_frames: int = 200,
        backend_factory=_CyndiBackend,
        lost_after_s: float = 2.0,
        discovery_timeout_s: float = 1.0,
        read_timeout_ms: int = 100,
    ) -> None:
        if not source_name:
            raise ValueError('NDI audio device needs a source name, e.g. "ndi:MIXER (Program)"')
        self._source_name = source_name
        self._channels = list(channels)
        self._own = own_stream_name
        self._target_rate = target_sample_rate
        self._frame_samples = int(target_sample_rate * frame_ms / 1000)
        self._backend_factory = backend_factory
        self._lost_after_s = lost_after_s
        self._discovery_timeout_s = discovery_timeout_s
        self._read_timeout_ms = read_timeout_ms
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queued_frames)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backend: NdiBackend | None = None
        self.level: float = 0.0  # see MicCapture.level
        self.connected: bool = False
        # For the setup checks (narrator.health): "starting",
        # "runtime_unavailable", "searching", "receiving" or "channel_error",
        # since when (time.monotonic) and a human detail. Plain attribute
        # writes from the worker thread, read lock-free.
        self.status: str = "starting"
        self.status_since: float = time.monotonic()
        self.status_detail: str | None = None

    @property
    def source_name(self) -> str:
        return self._source_name

    def _set_status(self, status: str, detail: str | None = None) -> None:
        if status != self.status:
            self.status_since = time.monotonic()
        self.status_detail = detail
        self.status = status

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if is_own_source(self._source_name, self._own):
            raise ValueError(
                f"NDI source {self._source_name!r} is narrator's own caption output; "
                "choose a different source"
            )
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ndi-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.level = 0.0
        self.connected = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def frames(self, timeout: float = 1.0, stop_event: threading.Event | None = None) -> Iterator[np.ndarray]:
        """Same contract as MicCapture.frames."""
        while stop_event is None or not stop_event.is_set():
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                continue

    # --- worker thread ---------------------------------------------------

    def _run(self) -> None:
        backend = None
        try:
            while not self._stop.is_set():
                if backend is None:
                    try:
                        backend = self._backend_factory()
                    except Exception as exc:
                        logger.error("NDI runtime unavailable (%s); retrying in 5s", exc)
                        self._set_status("runtime_unavailable", str(exc))
                        self._stop.wait(5.0)
                        continue
                    self._backend = backend
                try:
                    self._connect_and_receive(backend)
                except Exception:
                    logger.exception("NDI audio error; reconnecting")
                    self._reset_level()
                    self._stop.wait(1.0)
        finally:
            self._reset_level()
            if backend is not None:
                backend.close()
            self._backend = None

    def _reset_level(self) -> None:
        self.level = 0.0
        self.connected = False

    def _connect_and_receive(self, backend: NdiBackend) -> None:
        """Wait for the source, receive until it goes quiet, return (the
        caller loops back to discovery)."""
        name = None
        warned = False
        if self.status != "searching":
            self._set_status("searching")
        while name is None and not self._stop.is_set():
            names = backend.source_names(self._discovery_timeout_s)
            name = find_source(names, self._source_name, self._own)
            if name is None:
                if not warned:
                    logger.warning("NDI audio source %r not found; waiting for it to appear", self._source_name)
                    warned = True
                self._stop.wait(0.2)
        if name is None:
            return
        logger.info("NDI audio: connecting to %r", name)
        stream = backend.connect(name)
        try:
            self._receive_loop(stream, name)
        finally:
            stream.close()
            self._reset_level()

    def _receive_loop(self, stream: NdiAudioStream, name: str) -> None:
        resampler: StreamResampler | None = None
        chunker = FrameChunker(self._frame_samples)
        last_audio = time.monotonic()
        channel_error_logged = False
        while not self._stop.is_set():
            got = stream.read(self._read_timeout_ms)
            now = time.monotonic()
            if got is None:
                if now - last_audio > 0.5:
                    self.level = 0.0  # stalled: the meter must not freeze at the last value
                if now - last_audio > self._lost_after_s:
                    logger.warning("NDI audio source %r stopped delivering audio; waiting for it to return", name)
                    return
                continue
            planar, rate = got
            last_audio = now
            try:
                check_channels_available(self._channels, planar.shape[0], f"NDI source {name!r}")
            except ValueError as exc:
                if not channel_error_logged:
                    logger.error("%s", exc)
                    channel_error_logged = True
                    self._set_status("channel_error", str(exc))
                self.level = 0.0
                continue
            if not self.connected:
                logger.info("NDI audio: receiving from %r (%d ch @ %d Hz)", name, planar.shape[0], rate)
                self.connected = True
                self._set_status("receiving")
            if resampler is None or resampler.source_rate != rate:
                resampler = StreamResampler(rate, self._target_rate)
                chunker.reset()
            mono = mix_channels(planar.T, self._channels)
            mono16 = resampler.process(mono)
            if mono16.size:
                self.level = float(np.sqrt(np.mean(mono16**2)))
            for frame in chunker.push(mono16):
                self._put(frame)

    def _put(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest, like MicCapture
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass
