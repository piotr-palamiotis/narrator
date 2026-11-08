#!/usr/bin/env python3
"""List Vulkan physical devices with ctypes -- no vulkan-tools, no Vulkan SDK.

Only the loader (libvulkan.so.1) and a driver ICD are needed, which is exactly
the runtime requirement of narrator's ggml-vulkan backend, so this answers
"would whisper.cpp find a GPU here?". Prints one line per device:

    <type>\t<name>\tapi <major.minor.patch>

Exit codes: 0 = at least one device, 1 = loader present but no device,
2 = loader missing / instance creation failed. Standard library only.
"""
import ctypes
import ctypes.util
import sys

TYPES = {0: "other", 1: "integrated-gpu", 2: "discrete-gpu", 3: "virtual-gpu", 4: "cpu"}


def main() -> int:
    name = ctypes.util.find_library("vulkan") or "libvulkan.so.1"
    try:
        vk = ctypes.CDLL(name)
    except OSError as exc:
        print(f"Vulkan loader not found ({exc}); install libvulkan1", file=sys.stderr)
        return 2

    # VkInstanceCreateInfo with every pointer/count zero: sType = 1.
    create_info = (ctypes.c_uint8 * 64)()
    ctypes.c_uint32.from_buffer(create_info, 0).value = 1
    instance = ctypes.c_void_p()
    rc = vk.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
    if rc != 0:
        print(f"vkCreateInstance failed (VkResult {rc}): no usable driver/ICD", file=sys.stderr)
        return 2

    count = ctypes.c_uint32(0)
    vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
    handles = (ctypes.c_void_p * max(count.value, 1))()
    vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles)
    for i in range(count.value):
        # VkPhysicalDeviceProperties: u32 apiVersion, driverVersion, vendorID,
        # deviceID, deviceType, then char deviceName[256]; 1024 bytes covers it.
        props = (ctypes.c_uint8 * 1024)()
        vk.vkGetPhysicalDeviceProperties(ctypes.c_void_p(handles[i]), ctypes.byref(props))
        api = ctypes.c_uint32.from_buffer(props, 0).value
        dtype = ctypes.c_uint32.from_buffer(props, 16).value
        dname = bytes(props[20:276]).split(b"\0", 1)[0].decode("utf-8", "replace")
        print(f"{TYPES.get(dtype, dtype)}\t{dname}\tapi {api >> 22}.{(api >> 12) & 0x3FF}.{api & 0xFFF}")
    vk.vkDestroyInstance(instance, None)
    return 0 if count.value else 1


if __name__ == "__main__":
    sys.exit(main())
