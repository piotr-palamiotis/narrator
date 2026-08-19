"""Cross-platform input-device enumeration (Windows/Ubuntu via PortAudio)."""
from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd


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
