"""Cross-platform input-device enumeration (Windows/Ubuntu via PortAudio)."""
from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd

from narrator.audio.mixing import NDI_DEVICE_PREFIX


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float


def list_input_devices() -> list[InputDevice]:
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0:
            devices.append(
                InputDevice(
                    index=index,
                    name=info["name"],
                    max_input_channels=info["max_input_channels"],
                    default_sample_rate=info["default_samplerate"],
                )
            )
    return devices


def resolve_device(device_spec: str) -> int | None:
    """Resolve a config `audio.device` value to a sounddevice device index.

    "default" -> None (let PortAudio pick the system default input).
    Otherwise: case-insensitive substring match against device names; the
    first match wins. Raises ValueError if nothing matches, so a typo'd
    device name in config fails loudly at startup rather than silently
    falling back.
    """
    if device_spec.strip().lower() == "default":
        return None
    needle = device_spec.strip().lower()
    for device in list_input_devices():
        if needle in device.name.lower():
            return device.index
    available = ", ".join(d.name for d in list_input_devices())
    raise ValueError(f"No input device matching {device_spec!r}. Available: {available}")


def ndi_source_from_spec(device_spec: str) -> str | None:
    """`"ndi:MIXER (Program)"` -> `"MIXER (Program)"`; None for a microphone
    spec. The prefix match is case-insensitive."""
    spec = device_spec.strip()
    if spec[: len(NDI_DEVICE_PREFIX)].lower() == NDI_DEVICE_PREFIX:
        return spec[len(NDI_DEVICE_PREFIX):].strip()
    return None


def list_device_choices(own_ndi_name: str | None = None, ndi_timeout: float = 1.5) -> list[str]:
    """Device values a picker can offer: "default", each microphone name, and
    each discovered NDI source as `ndi:<name>`. Never raises on NDI trouble."""
    from narrator.audio.ndi_capture import list_ndi_sources

    names = ["default"]
    try:
        names += [d.name for d in list_input_devices()]
    except Exception:
        pass
    names += [f"{NDI_DEVICE_PREFIX}{n}" for n in list_ndi_sources(ndi_timeout, own_ndi_name)]
    return names


def rescan_devices() -> None:
    """Make PortAudio enumerate devices again (it lists them once, at
    initialization, so a microphone plugged in later stays invisible).
    Only safe while no stream is open: the audio "Retry" fix calls it after
    the old capture is stopped. sounddevice exposes no public call for this."""
    sd._terminate()
    sd._initialize()
