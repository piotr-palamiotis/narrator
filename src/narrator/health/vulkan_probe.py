"""`python -m narrator.health.vulkan_probe`: print the Vulkan GPUs as JSON
(index, name, type, PCI bus id, VRAM: the unit list, narrator.asr.units).

The setup checks run this in a subprocess instead of enumerating in the app
process: creating a Vulkan instance loads the GPU driver's ICD into the
caller, and a driver (or ctypes) fault there would take the whole app down
-- exactly what the failure policy forbids. A crashed or hung probe only
makes the check report "could not check".
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    from narrator.asr.vulkan_devices import vulkan_devices

    devices = [
        {
            "index": d.index,
            "name": d.name,
            "discrete": d.is_discrete,
            "type": d.type_name,
            "pci_bus_id": d.pci_bus_id,
            "vram_mb": d.vram_mb,
        }
        for d in vulkan_devices()
    ]
    json.dump(devices, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
