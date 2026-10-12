# GPU acceleration for narrator (and when you'd rebuild whisper.cpp)

> **Update (2026-10-03):** the situation flipped back. The current PyPI
> `pywhispercpp` 1.5.1 wheel ships **no `libggml-vulkan.so`** (CPU ggml
> only), and its CPU path SIGILLs at `whisper_init` on the pre-AVX Pentium
> G4560 test box (core dump right after the `whisper_init_with_params_no_state`
> param logs, reproduced 2026-10-03). The source build below is required
> again. `make install` runs it automatically when the Vulkan backend is
> missing (`packaging/whisper_cpp/build_vulkan_linux.sh`: pinned to tag
> `v1.5.1`, `GGML_VULKAN=1`, `SETUPTOOLS_SCM_PRETEND_VERSION` because a
> shallow tag clone otherwise builds as version 0.0.0, plus the mandatory
> `$ORIGIN` RPATH fix). Measured after that build on the RX 6700 XT: bench
> 2s→0.24s, 10s→1.05s, 20s→1.96s, language detection 0.30s. The rest of
> this section is kept for history.

## Start here: you probably do not need to build anything

An earlier version of this document said the PyPI `pywhispercpp` wheel was
**CPU-only** and that a from-source rebuild with `GGML_VULKAN=1` was required
for GPU acceleration. That is **not true of current wheels**, and following
those instructions unnecessarily was costing a slow, fragile source build on
every deployment.

Verified against `pywhispercpp` 1.5.1 on a Radeon RX 6700 XT (RADV,
Mesa 25.2):

- The wheel installs `libggml-vulkan.so` into site-packages alongside
  `libggml-cpu.so`. Vulkan is compiled in.
- `use_gpu=True` really does run inference on the discrete GPU. whisper.cpp
  logs `ggml_vulkan: Found 2 Vulkan devices` and
  `whisper_backend_init_gpu: using Vulkan0 backend`.
- Measured decode latency on that GPU, large-v3: ~0.3s for a 2s window,
  ~1.4s for a 20s window. Comfortably real-time for live captioning.

So the normal deployment is just:

```bash
pip install -e .
```

### Model choice: do not use `large-v3-turbo` on pywhispercpp 1.5.1

The wheel's bundled whisper.cpp predates the turbo tensor layout: loading
`large-v3-turbo` (or `large-v3-turbo-q5_0` etc.) logs
`ERROR not all tensors loaded from model file - expected 587, got 170` and
then **segfaults the process** (measured on the RX 6700 XT / Mesa 25.2
setup, and it would do the same on CPU). In narrator that is worse than it
sounds: the ASR worker dies at spawn, the supervisor's quick-crash fallback
kicks in after 3 attempts, and the app silently lands on the CPU path --
which on a pre-AVX CPU means ~96 seconds per 2s decode window (measured on
the same Pentium G4560 test box). Stick with `large-v3` until the wheel
ships a newer whisper.cpp; it decodes a 20s window in ~1.0s on the RX 6700
XT, which is comfortably real-time.

## Runtime requirements on the target machine

You need the Vulkan **loader and driver**, which come from the graphics
driver packages -- not the SDK, and *not* the `vulkan-tools` diagnostic
package:

```bash
# Ubuntu/Debian, AMD (Mesa RADV):
sudo apt install libvulkan1 mesa-vulkan-drivers
```

That gives you `libvulkan.so.1` plus an ICD JSON in
`/usr/share/vulkan/icd.d/` (e.g. `radeon_icd.json`), which is all
whisper.cpp and narrator's own device probe need.

`vulkaninfo` (from `vulkan-tools`) is a *diagnostic* tool and is **not**
required. narrator deliberately does not depend on it: it enumerates GPUs by
calling `libvulkan.so.1` through ctypes (see
`narrator.asr.backend._vulkan_devices`). An older revision shelled out to
`vulkaninfo`, which made narrator report "no Vulkan devices" on machines
that had a perfectly working GPU but no diagnostic package installed.

## CPU architecture: no special build needed

The same wheels are built for **baseline x86-64**. Disassembling
`libggml-cpu.so`, `libggml-base.so`, `libwhisper.so` and `libggml-vulkan.so`
finds **zero** AVX/AVX2/FMA/VEX-encoded instructions, so they load and run on
pre-AVX CPUs. Verified on an Intel Pentium G4560, which tops out at SSE4.2:
both `use_gpu=True` and the `use_gpu=False` CPU fallback load and transcribe
without an illegal-instruction crash.

In other words, an old CPU is not a reason to rebuild whisper.cpp, and the
CPU fallback path is safe to rely on everywhere.

## Verifying which backend you actually got

```bash
python -c "
from narrator.asr.backend import WhisperBackend
from narrator.config.schema import AsrConfig
b = WhisperBackend(AsrConfig(model='tiny', device='vulkan'))
print(b.resolved_device_description)
"
```

This prints the selected adapter, e.g.:

```
Vulkan (AMD Radeon RX 6700 XT (RADV NAVI22) [discrete])
```

If it prints `CPU`, check that `libvulkan1` and your driver's ICD are
installed (see above) and that the user running narrator can open
`/dev/dri/renderD*` (membership in the `render` group on most distros).

### Choosing between multiple GPUs

Machines with a discrete card *and* an integrated one expose both to Vulkan.
whisper.cpp on its own always takes device 0, so on a box whose iGPU
enumerates first it would silently run on the slow adapter. narrator defaults
`asr.gpu_device: auto`, which prefers the first **discrete** GPU. Override it
with an explicit index if you need a specific card:

```yaml
asr:
  device: vulkan
  gpu_device: 1
```

The indices are the ones narrator logs at startup
(`Vulkan devices: 0=..., 1=...`), which match whisper.cpp's own ordering.

---

# Building pywhispercpp from source

Only needed if a future wheel drops the Vulkan backend, or you want a
different ggml backend (CUDA/HIP) than the wheel ships. Everything below is
still accurate for that case.

## Prerequisites (build machine only)

- CMake >= 3.21 and a C/C++ compiler (build-essential on Ubuntu, MSVC Build
  Tools on Windows).
- The **Vulkan SDK** (https://vulkan.lunarg.com/), so the Vulkan
  headers/loader are available to CMake at build time. whisper.cpp's Vulkan
  backend also needs `find_package(SPIRV-Headers)` and
  `find_package(SPIRV-Tools)` to succeed -- the official LunarG SDK tarball
  bundles both, but a minimal apt-based setup needs them installed
  explicitly too (Ubuntu: `spirv-headers` and `spirv-tools`), or CMake's
  configure step fails with "Could not find a package configuration file
  provided by SPIRV-Headers" even though the Vulkan loader itself was found
  fine.
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

## Why pywhispercpp instead of vendoring whisper.cpp directly

The original plan considered vendoring whisper.cpp as a git submodule driven
by a custom ctypes wrapper, for maximum control over build flags. In
practice the published wheel already ships the Vulkan backend, so
`narrator.asr.backend.WhisperBackend` targets pywhispercpp directly rather
than a hand-written ctypes binding. Revisit the vendored approach only if a
future pywhispercpp release drops the Vulkan backend from its wheels.
