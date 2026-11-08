"""Picks the capture class for `audio.device`: `ndi:<source>` -> NdiAudioCapture,
anything else -> MicCapture (PortAudio). Both expose the same surface
(`start()`, `stop()`, `level`, `frames()`), which is all AppController uses."""
from __future__ import annotations

from narrator.audio.capture import MicCapture
from narrator.audio.devices import ndi_source_from_spec
from narrator.audio.ndi_capture import NdiAudioCapture
from narrator.config.schema import AudioConfig


def capture_class_for(device_spec: str) -> type:
    return NdiAudioCapture if ndi_source_from_spec(device_spec) is not None else MicCapture


def create_capture(audio: AudioConfig, own_ndi_name: str | None = None) -> MicCapture | NdiAudioCapture:
    """Build (not start) the capture for `audio`. `own_ndi_name` is narrator's
    own NDI output name, excluded from NDI discovery."""
    ndi_source = ndi_source_from_spec(audio.device)
    if ndi_source is not None:
        return NdiAudioCapture(
            ndi_source,
            frame_ms=audio.frame_ms,
            channels=audio.channels,
            own_stream_name=own_ndi_name,
        )
    return MicCapture(audio.device, audio.frame_ms, channels=audio.channels)
