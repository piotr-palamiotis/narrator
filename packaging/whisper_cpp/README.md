# Building a Vulkan-accelerated whisper.cpp for narrator

By default, `pip install pywhispercpp` installs a prebuilt **CPU-only**
wheel from PyPI. To get real GPU acceleration (NVIDIA or AMD, via Vulkan)
you must build `pywhispercpp` from source with the Vulkan ggml backend
enabled. This is a one-time step on each build machine/target platform
before running `packaging/pyinstaller/build_linux.sh` or
`build_windows.ps1` -- confirmed against the pywhispercpp project's own
documented build flags (github.com/absadiki/pywhispercpp).

## Prerequisites (build machine only -- not required at runtime for the CPU fallback path)

- CMake >= 3.21 and a C/C++ compiler (build-essential on Ubuntu, MSVC Build
  Tools on Windows).
- The **Vulkan SDK** (https://vulkan.lunarg.com/) installed, so the Vulkan
  headers/loader are available to CMake at build time.
- Python matching the venv narrator will run in.

## Ubuntu

```bash
source .venv/bin/activate
pip uninstall -y pywhispercpp
GGML_VULKAN=1 pip install --no-cache-dir --force-reinstall -v \
    git+https://github.com/absadiki/pywhispercpp
```

## Windows (PowerShell, with the Vulkan SDK's env vars already loaded)

```powershell
.venv\Scripts\Activate.ps1
pip uninstall -y pywhispercpp
$env:GGML_VULKAN = "1"
pip install --no-cache-dir --force-reinstall -v `
    git+https://github.com/absadiki/pywhispercpp
```

## Verifying the build picked up Vulkan

```bash
python -c "
from narrator.asr.backend import WhisperBackend
from narrator.config.schema import AsrConfig
b = WhisperBackend(AsrConfig(model='tiny', device='vulkan'))
print(b.resolved_device_description)
"
```

This should print `Vulkan (<your GPU name>)` rather than `CPU`. If it falls
back to CPU, check that:
- The Vulkan SDK was installed and discoverable by CMake during the pip build.
- Your GPU has a working Vulkan driver (`vulkaninfo --summary` should list it).
- The pip install log (`-v`) actually shows a Vulkan CMake configure step --
  if it silently built without it, `GGML_VULKAN=1` wasn't picked up (check
  the environment variable is set in the same shell as the pip install).

## Why this instead of vendoring whisper.cpp directly

The original plan considered vendoring whisper.cpp as a git submodule driven
by a custom ctypes wrapper, for maximum control over build flags. In
practice, pywhispercpp's own `GGML_VULKAN=1` source-build path gives the
same Vulkan-enabled shared library with far less custom glue code to
maintain, so `narrator.asr.backend.WhisperBackend` targets pywhispercpp
directly rather than a hand-written ctypes binding. Revisit the vendored
approach only if a future pywhispercpp release drops or breaks this build
flag.

## CPU-only builds

If you don't need GPU acceleration on a given build machine, the plain
`pip install pywhispercpp` (CPU-only wheel) is fine -- `asr.device: auto`
will detect the lack of a Vulkan-enabled backend at runtime (GPU init will
simply fail) and fall back to CPU automatically.
