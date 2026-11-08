"""Vulkan GPU enumeration through `libvulkan.so.1` via ctypes (moved out of
backend.py for the processing-unit list, todo C6; backend.py re-exports it).

Device detection calls into the Vulkan loader directly rather than shelling
out to the `vulkaninfo` CLI. `vulkaninfo` ships in the *vulkan-tools*
package, which is a developer/diagnostic package and is routinely absent on
machines that have a perfectly working Vulkan runtime (the ICD +
`libvulkan.so.1` come from the driver package instead). The old CLI-based
probe therefore reported "no Vulkan devices" on a machine with a working GPU.

Creating a Vulkan instance loads the GPU driver's ICD into the caller. The
app process never does that: the setup checks and the unit list enumerate in
a subprocess (narrator.health.vulkan_probe); the ASR worker, which loads the
driver anyway, enumerates in-process.
"""
from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Vulkan constants (vulkan_core.h). Only the handful we need.
_VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES = 1000071004
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT = 1000212000
_VK_SUCCESS = 0
_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256
_VK_MAX_EXTENSION_NAME_SIZE = 256
_VK_PHYSICAL_DEVICE_TYPE_CPU = 4
_VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
_VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 1
_VK_API_VERSION_1_1 = (1 << 22) | (1 << 12)
_PCI_BUS_INFO_EXTENSION = b"VK_EXT_pci_bus_info"

_DEVICE_TYPE_NAMES = {
    0: "other",
    1: "integrated",
    2: "discrete",
    3: "virtual",
    4: "cpu",
}

# VkPhysicalDeviceProperties is ~824 bytes and ends with two large nested
# structs (limits, sparseProperties) we never read. Rather than model those
# field-for-field, hand Vulkan an over-sized buffer and slice the leading
# scalar fields out by offset: apiVersion/driverVersion/vendorID/deviceID are
# 4 bytes each (16 total), then deviceType (4), then deviceName.
_PROPERTIES_BUFFER_SIZE = 2048
_DEVICE_TYPE_OFFSET = 16
_DEVICE_NAME_OFFSET = 20
# VkPhysicalDeviceProperties2: sType (4) + padding (4) + pNext (8), then the
# properties above (8-aligned: `limits` holds VkDeviceSize fields).
_PROPERTIES2_HEADER = 16
# VkPhysicalDeviceIDProperties: header (16), deviceUUID[16].
_ID_PROPERTIES_SIZE = 64
_UUID_OFFSET = 16
# VkPhysicalDevicePCIBusInfoPropertiesEXT: header (16), domain/bus/device/function (u32 each).
_PCI_PROPERTIES_SIZE = 32
_PCI_OFFSET = 16
# VkPhysicalDeviceMemoryProperties: memoryTypeCount (4), memoryTypes[32] (8
# each), memoryHeapCount (4) at 260, memoryHeaps[16] at 264 (size u64 +
# flags u32 + padding = 16 each).
_MEMORY_PROPERTIES_SIZE = 1024
_HEAP_COUNT_OFFSET = 260
_HEAPS_OFFSET = 264
_HEAP_SIZE = 16


@dataclass(frozen=True)
class VulkanDevice:
    """One GPU as whisper.cpp/ggml will index it (see `vulkan_devices`).
    `pci_bus_id` ("0000:01:00.0"), `vram_mb` (the largest device-local heap)
    and `uuid` are None when the driver/loader cannot report them."""

    index: int
    name: str
    device_type: int
    pci_bus_id: str | None = None
    vram_mb: int | None = None
    uuid: str | None = None

    @property
    def is_discrete(self) -> bool:
        return self.device_type == _VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU

    @property
    def type_name(self) -> str:
        return _DEVICE_TYPE_NAMES.get(self.device_type, str(self.device_type))

    def __str__(self) -> str:
        return f"{self.name} [{self.type_name}]"


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def format_pci_bus_id(domain: int, bus: int, device: int, function: int) -> str:
    """The Linux sysfs spelling ("0000:01:00.0"), which nvidia-smi's
    "00000000:01:00.0" is normalized to (see normalize_pci_bus_id)."""
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def normalize_pci_bus_id(value: str | None) -> str | None:
    """Lower-case, 4-digit domain: nvidia-smi prints an 8-digit domain
    ("00000000:01:00.0"), sysfs and Vulkan a 4-digit one."""
    if not value:
        return None
    text = value.strip().lower()
    parts = text.split(":")
    if len(parts) == 3:
        domain, bus, rest = parts
        try:
            domain = f"{int(domain, 16):04x}"
        except ValueError:
            return text
        return f"{domain}:{bus}:{rest}"
    if len(parts) == 2:  # no domain
        return f"0000:{text}"
    return text


def _load_loader():
    for name in ("libvulkan.so.1", "vulkan-1.dll"):  # Linux, Windows loader names
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _declare(lib) -> None:
    # Explicit argtypes: a bare Python int is passed as a 32-bit C int,
    # which truncates a handle above 4 GB -- always the case off the main
    # thread (glibc's per-thread arenas are mmapped high) and a SIGSEGV
    # there (seen when the setup checks probed from their own thread).
    vp = ctypes.c_void_p
    lib.vkCreateInstance.argtypes = [vp, vp, vp]
    lib.vkCreateInstance.restype = ctypes.c_int32
    lib.vkDestroyInstance.argtypes = [vp, vp]
    lib.vkDestroyInstance.restype = None
    lib.vkEnumeratePhysicalDevices.argtypes = [vp, vp, vp]
    lib.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    lib.vkGetPhysicalDeviceProperties.argtypes = [vp, vp]
    lib.vkGetPhysicalDeviceProperties.restype = None
    lib.vkGetPhysicalDeviceMemoryProperties.argtypes = [vp, vp]
    lib.vkGetPhysicalDeviceMemoryProperties.restype = None
    lib.vkEnumerateDeviceExtensionProperties.argtypes = [vp, vp, vp, vp]
    lib.vkEnumerateDeviceExtensionProperties.restype = ctypes.c_int32
    try:
        lib.vkGetPhysicalDeviceProperties2.argtypes = [vp, vp]
        lib.vkGetPhysicalDeviceProperties2.restype = None
    except AttributeError:  # a Vulkan 1.0 loader
        pass


def _create_instance(lib) -> tuple[ctypes.c_void_p | None, bool]:
    """(instance, is_1_1). A Vulkan 1.1 instance is asked for first (the
    PCI bus id and UUID need vkGetPhysicalDeviceProperties2); a 1.0-only
    loader refuses that, and a plain instance is created instead."""
    app = _VkApplicationInfo(sType=_VK_STRUCTURE_TYPE_APPLICATION_INFO, apiVersion=_VK_API_VERSION_1_1)
    for use_app_info in (True, False):
        instance = ctypes.c_void_p()
        create_info = _VkInstanceCreateInfo(
            sType=_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=ctypes.cast(ctypes.pointer(app), ctypes.c_void_p) if use_app_info else None,
        )
        try:
            result = lib.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
        except (AttributeError, OSError):
            return None, False
        if result == _VK_SUCCESS:
            return instance, use_app_info
    return None, False


def _has_extension(lib, handle: ctypes.c_void_p, wanted: bytes) -> bool:
    count = ctypes.c_uint32(0)
    if lib.vkEnumerateDeviceExtensionProperties(handle, None, ctypes.byref(count), None) != _VK_SUCCESS:
        return False
    if count.value == 0:
        return False
    stride = _VK_MAX_EXTENSION_NAME_SIZE + 4
    buffer = ctypes.create_string_buffer(stride * count.value)
    if lib.vkEnumerateDeviceExtensionProperties(handle, None, ctypes.byref(count), buffer) != _VK_SUCCESS:
        return False
    raw = buffer.raw
    for i in range(count.value):
        name = raw[i * stride : i * stride + _VK_MAX_EXTENSION_NAME_SIZE].split(b"\0")[0]
        if name == wanted:
            return True
    return False


def _identity_properties(lib, handle: ctypes.c_void_p) -> tuple[str | None, str | None]:
    """(pci bus id, device uuid) through vkGetPhysicalDeviceProperties2; the
    PCI struct is chained only when the device has VK_EXT_pci_bus_info."""
    if not hasattr(lib, "vkGetPhysicalDeviceProperties2"):
        return None, None
    id_props = ctypes.create_string_buffer(_ID_PROPERTIES_SIZE)
    ctypes.memmove(id_props, _VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES.to_bytes(4, "little"), 4)
    pci_props = None
    if _has_extension(lib, handle, _PCI_BUS_INFO_EXTENSION):
        pci_props = ctypes.create_string_buffer(_PCI_PROPERTIES_SIZE)
        ctypes.memmove(pci_props, _VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT.to_bytes(4, "little"), 4)
        # id_props.pNext -> pci_props
        ctypes.c_void_p.from_buffer(id_props, 8).value = ctypes.addressof(pci_props)
    props2 = ctypes.create_string_buffer(_PROPERTIES2_HEADER + _PROPERTIES_BUFFER_SIZE)
    ctypes.memmove(props2, _VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2.to_bytes(4, "little"), 4)
    ctypes.c_void_p.from_buffer(props2, 8).value = ctypes.addressof(id_props)
    lib.vkGetPhysicalDeviceProperties2(handle, props2)
    uuid_bytes = id_props.raw[_UUID_OFFSET : _UUID_OFFSET + 16]
    uuid = uuid_bytes.hex() if any(uuid_bytes) else None
    pci = None
    if pci_props is not None:
        raw = pci_props.raw
        domain, bus, device, function = (
            int.from_bytes(raw[_PCI_OFFSET + 4 * i : _PCI_OFFSET + 4 * i + 4], "little") for i in range(4)
        )
        pci = format_pci_bus_id(domain, bus, device, function)
    return pci, uuid


def _vram_mb(lib, handle: ctypes.c_void_p) -> int | None:
    buffer = ctypes.create_string_buffer(_MEMORY_PROPERTIES_SIZE)
    lib.vkGetPhysicalDeviceMemoryProperties(handle, buffer)
    raw = buffer.raw
    heaps = int.from_bytes(raw[_HEAP_COUNT_OFFSET : _HEAP_COUNT_OFFSET + 4], "little")
    largest = 0
    for i in range(min(heaps, 16)):
        offset = _HEAPS_OFFSET + i * _HEAP_SIZE
        size = int.from_bytes(raw[offset : offset + 8], "little")
        flags = int.from_bytes(raw[offset + 8 : offset + 12], "little")
        if flags & _VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
            largest = max(largest, size)
    return largest // (1024 * 1024) if largest else None


def vulkan_devices() -> list[VulkanDevice]:
    """Enumerate Vulkan GPUs, in ggml's own indexing order (which is what
    `context_params["gpu_device"]` selects against in a Vulkan-only build).

    ggml-vulkan lists discrete and integrated GPUs in the loader's order and
    skips `VK_PHYSICAL_DEVICE_TYPE_CPU` adapters (software rasterizers such
    as llvmpipe); so does this, otherwise a machine with llvmpipe installed
    would produce indices that don't line up with whisper.cpp's. ggml also
    drops a second driver for the same GPU (same deviceUUID, e.g. RADV +
    AMDVLK); this keeps the first one, so the count matches -- ggml may keep
    the other driver's entry (by its driver priority) and list it last.

    Any failure (no loader, no ICD, driver error) returns an empty list:
    callers treat that as "couldn't confirm a GPU", not a hard "no GPU",
    since whisper.cpp's own init remains the source of truth.
    """
    lib = _load_loader()
    if lib is None:
        logger.debug("Vulkan loader not found; cannot enumerate GPUs")
        return []
    try:
        _declare(lib)
    except AttributeError:
        logger.debug("Vulkan loader lacks core entry points")
        return []
    instance, is_1_1 = _create_instance(lib)
    if instance is None:
        logger.debug("vkCreateInstance failed; treating as no Vulkan devices")
        return []

    try:
        count = ctypes.c_uint32(0)
        if lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None) != _VK_SUCCESS:
            return []
        if count.value == 0:
            return []
        handles = (ctypes.c_void_p * count.value)()
        if lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles) != _VK_SUCCESS:
            return []

        devices: list[VulkanDevice] = []
        seen_uuids: set[str] = set()
        for raw_handle in handles:
            handle = ctypes.c_void_p(raw_handle)
            buffer = ctypes.create_string_buffer(_PROPERTIES_BUFFER_SIZE)
            lib.vkGetPhysicalDeviceProperties(handle, buffer)
            raw = buffer.raw
            device_type = int.from_bytes(raw[_DEVICE_TYPE_OFFSET : _DEVICE_TYPE_OFFSET + 4], "little")
            name = (
                raw[_DEVICE_NAME_OFFSET : _DEVICE_NAME_OFFSET + _VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]
                .split(b"\0")[0]
                .decode("utf-8", "replace")
            )
            if device_type == _VK_PHYSICAL_DEVICE_TYPE_CPU:
                continue  # ggml skips software rasterizers; keep indices aligned
            pci = uuid = None
            if is_1_1:
                try:
                    pci, uuid = _identity_properties(lib, handle)
                except (AttributeError, OSError, ValueError):
                    logger.debug("could not read the PCI bus id of %s", name, exc_info=True)
            if uuid is not None:
                if uuid in seen_uuids:
                    continue  # the same GPU through a second driver (see above)
                seen_uuids.add(uuid)
            try:
                vram = _vram_mb(lib, handle)
            except (AttributeError, OSError):
                vram = None
            devices.append(
                VulkanDevice(
                    index=len(devices), name=name, device_type=device_type, pci_bus_id=pci, vram_mb=vram, uuid=uuid
                )
            )
        return devices
    finally:
        try:
            lib.vkDestroyInstance(instance, None)
        except (AttributeError, OSError):
            pass


class GpuSelectionError(RuntimeError):
    """The configured GPU unit (asr.unit vulkan:<n>) is not present, or is
    present at that index but is a different card than the one recorded
    (asr.unit_identity) -- the indices shifted."""


def select_gpu_index(devices: list[VulkanDevice], configured: int | str) -> int:
    """Resolve a Vulkan device index against the enumerated devices.

    "auto" prefers the first discrete GPU, because the common failure mode is
    a workstation whose integrated adapter enumerates first: whisper.cpp
    would then default to `gpu_device=0` and run the model on the iGPU, which
    is dramatically slower but fails silently (it still "uses the GPU").
    """
    if configured != "auto":
        index = int(configured)
        # whisper.cpp silently loads the model on the CPU for an index it
        # doesn't have, which would break the "never switch units by itself"
        # policy, so refuse it. With nothing enumerated (no loader in this
        # process) whisper.cpp's own detection stays authoritative.
        if devices and not any(device.index == index for device in devices):
            raise GpuSelectionError(
                f"asr.unit vulkan:{index} is not among the {len(devices)} "
                f"enumerated Vulkan device(s): "
                + ", ".join(f"vulkan:{d.index}={d}" for d in devices)
            )
        return index

    for device in devices:
        if device.is_discrete:
            return device.index
    return 0
