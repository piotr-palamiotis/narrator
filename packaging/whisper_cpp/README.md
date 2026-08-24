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
  headers/loader are available to CMake at build time. whisper.cpp's Vulkan
  backend also needs `find_package(SPIRV-Headers)` and `find_package(SPIRV-Tools)`
  to succeed -- the official LunarG SDK tarball bundles both, but a
  minimal apt-based setup needs them installed explicitly too (Ubuntu:
  `spirv-headers` and `spirv-tools` packages), or CMake's configure step
  fails with "Could not find a package configuration file provided by
  SPIRV-Headers" even though the Vulkan loader itself was found fine.
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

## Fixing the shared-library RPATH (required after every source build)

Confirmed empirically, not theoretical: a `pip install git+...` source build
of pywhispercpp reports `Successfully installed pywhispercpp` but is **not
actually usable yet**. The built extension (`_pywhispercpp*.so`) and its
sibling libraries (`libwhisper.so.1`, `libggml*.so`, `libggml-vulkan.so`,
...) land permanently in site-packages, but pywhispercpp's build does not
set a `$ORIGIN`-relative RPATH on them -- it bakes in the *absolute path of
pip's temporary build directory* instead. That directory is deleted the
moment the install finishes, so the very next `import pywhispercpp.model`
fails with `ImportError: libwhisper.so.1: cannot open shared object file: No
such file or directory` (verify with `readelf -d <file> | grep RUNPATH` --
it'll show `/tmp/pip-req-build-.../build/lib...`, a path that no longer
exists). This is an upstream packaging gap in every source build, not
something specific to one machine or one Vulkan setup.

Fix it once per build, right after the `pip install` command above:

```bash
pip install patchelf   # no root needed, prebuilt wheel
SITE=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
for f in "$SITE"/lib{whisper,ggml,ggml-base,ggml-cpu,ggml-vulkan}.so* \
         "$SITE"/_pywhispercpp*.so; do
    [ -f "$f" ] && patchelf --set-rpath '$ORIGIN' "$f"
done
```

After this, the built libraries resolve their sibling dependencies relative
to their own directory and need no `LD_LIBRARY_PATH` at runtime -- confirmed
by importing in a completely clean environment.

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
