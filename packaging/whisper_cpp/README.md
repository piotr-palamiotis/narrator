# GPU acceleration for narrator (and when you'd rebuild whisper.cpp)

## Current state (2026-10-17): source build, whisper.cpp v1.9.5

The PyPI `pywhispercpp` 1.5.1 wheel ships **no `libggml-vulkan.so`** (CPU
ggml only), and its CPU path SIGILLs at `whisper_init` on the pre-AVX
Pentium G4560 test box (reproduced 2026-10-03). narrator therefore always
uses a source build, made by `packaging/whisper_cpp/build_vulkan_linux.sh`
(`make install` / `make asr-backend` run it when the expected version is not
installed):

- pywhispercpp **v1.5.1** bindings (its newest release; `main` == the tag)
  with the whisper.cpp submodule moved from the vendored v1.8.5 to
  **v1.9.5** (`PYWHISPERCPP_TAG` / `WHISPER_CPP_TAG` in the script). The C
  API diff is additions only, so the 1.5.1 bindings compile unchanged.
  Installed version: `1.5.1+whispercpp.1.9.5`.
- `GGML_VULKAN=1` (exported by the script; pywhispercpp's setup.py turns
  every environment variable into a CMake `-D`). The CPU backend is
  `-march=native`, i.e. SSE4.2-only on the Pentium (no ymm/VEX instructions
  in the built libs) and not portable to CPUs with fewer extensions.
- whisper.cpp >= 1.9 forces `CMAKE_LIBRARY_OUTPUT_DIRECTORY` to
  `<build>/bin`; the script guards that line with `if (NOT ...)`, otherwise
  libwhisper/libggml* never reach the wheel and the import fails.
- `SETUPTOOLS_SCM_PRETEND_VERSION` (a shallow tag clone builds as 0.0.0
  otherwise), the venv's bin on `PATH` (ggml-vulkan builds its shader
  generator as a `-G Ninja` sub-project), and the `$ORIGIN` RPATH fix below
  (now including `libparakeet.so`, new in 1.9).
- Build time on the Pentium: ~4 min; build tree ~260 MB.

## CUDA build (NVIDIA) -- `ASR_GPU=cuda`, not yet run on a machine

The same script builds the CUDA backend instead of Vulkan:

```bash
make asr-backend ASR_GPU=cuda          # or ASR_GPU=auto (the default):
                                       # cuda when nvcc + nvidia-smi work
```

- Needs the NVIDIA driver and the **CUDA toolkit** (`nvcc`; on PATH, or
  under `/usr/local/cuda` / `$CUDA_HOME`, which the script adds to PATH),
  plus cmake and git. Vulkan packages are not needed.
- `GGML_CUDA=1` replaces `GGML_VULKAN=1`. Only one GPU backend per build: a
  build with both would number the NVIDIA card twice in whisper.cpp's
  `gpu_device`. A CUDA build therefore has no Vulkan units (vulkan:<n> is
  refused by the backend and reported by the setup checks), and the other
  way round.
- GPU architectures: ggml's default with `GGML_NATIVE` and CUDA >= 11.6 is
  `native` (the cards present at build time); set e.g.
  `CMAKE_CUDA_ARCHITECTURES=86` in the environment to build for another card.
- RUNPATH: `$ORIGIN`, plus the toolkit's `lib64` when it lives under
  `/usr/local/cuda` (libcudart/libcublas), so the clean-environment import
  check works without `LD_LIBRARY_PATH`.
- Runtime: `asr.unit: cuda:<n>` (nvidia-smi's index; the worker sets
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` so CUDA's numbering matches) or `auto`
  (prefers CUDA). The shipped default (`preset: quality` = large-v3,
  `unit: auto`) runs large-v3 on the first CUDA card.
- Switching back: `make asr-backend ASR_GPU=vulkan` (the version string is
  the same; the script decides by the backend library in site-packages).

Going back to whisper.cpp 1.8.5 (verified):

```bash
WHISPER_CPP_TAG=v1.8.5 make asr-backend        # rebuilds, ~6 min
# or install the kept wheel, then run the patchelf loop from "Fixing the
# shared-library RPATH" below:
.venv/bin/pip install --force-reinstall --no-deps third_party/wheels/pywhispercpp-1.5.1+whispercpp.1.8.5-*.whl
```

Measured on the RTX 3070 (driver 615, Vulkan device 0), `narrator.asr.bench`,
large-v3, 2/5/10/20 s windows:

| build | decode (s) | detect (s) |
|---|---|---|
| 1.8.5, ggml defaults (before) | 0.19 / 0.46 / 0.90 / 1.73 | 0.15 |
| 1.9.5, ggml defaults | 0.25 / 0.63 / 1.23 / 2.25 | 0.14 |
| 1.8.5, graph optimize off | 0.17 / 0.41 / 0.80 / 1.50 | 0.13 |
| **1.9.5, graph optimize off (now)** | **0.17 / 0.41 / 0.80 / 1.52** | **0.13** |

The 1.9.5 slowdown is entirely ggml-vulkan's graph optimizer, which runs on
the CPU for every decoded token (12.6 vs 8.5 ms/token; encode is unchanged
at ~85 ms per 20 s window). `WhisperBackend` therefore sets
`GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` for GPU loads unless the environment
already sets it (`NARRATOR_VK_GRAPH_OPTIMIZE=1` keeps ggml's default). The
text is identical either way. A faster host CPU may not need this; measure
before changing it.

Re-verified on 1.9.5 (details in CLAUDE.md): `detect_language=True` still
returns zero segments; beam search runs on the GPU (about 2x greedy cost);
the token API backend.py uses is unchanged; `flash_attn` works on the 3070
and is ~17% faster on 20 s windows but changes the text, so it stays opt-in
(whisper.cpp's own default is ON, so it is always passed explicitly); DTW
token timestamps work (alignment-head presets per model; disabled by
whisper.cpp when flash attention is on); the built-in ggml Silero VAD is
not usable from pywhispercpp 1.5.1 as a streaming VAD. `large-v3-turbo` on
1.9.5 was not tested (no model file here); the segfault described below
was on the 1.5.1 wheel's older whisper.cpp.

---

History (2026-10-03 and earlier; RX 6700 XT, kept for reference). Parts of
it no longer apply: the PyPI wheel is not usable (above).

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
sounds: the ASR worker dies at spawn, and (at the time) the supervisor's
quick-crash fallback kicked in after 3 attempts and silently landed the app
on the CPU path -- which on a pre-AVX CPU means ~96 seconds per 2s decode
window (measured on the same Pentium G4560 test box). Today a crash loop
stops ASR and shows a problem in the web UI instead (no CPU fallback). Stick with `large-v3` until the wheel
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

On NVIDIA the proprietary driver ships its own Vulkan ICD
(`nvidia_icd.json`); `libvulkan1` is the only extra package. No CUDA
toolkit is needed for the Vulkan backend (verified on an RTX 3070 with
driver 615 and no `nvcc`).

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
`asr.unit: auto`, which prefers the first **discrete** GPU. Pick a specific
card in the web UI's ASR tab (it lists every unit with name, PCI slot and
VRAM, and records the card's identity), or set it by hand:

```yaml
asr:
  unit: vulkan:1
  unit_identity: ""   # "<name> @ <PCI bus id>", filled in by the web UI
```

The indices are the ones narrator logs at startup
(`Vulkan devices: vulkan:0=..., vulkan:1=...`), which match whisper.cpp's own
ordering. With `unit_identity` recorded, a card added or removed that shifts
the indices is reported in the problems panel instead of silently running on
another card. (Before this, the setting was `asr.device: vulkan` +
`asr.gpu_device: 1`; such configs are converted on load.)

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
for f in "$SITE"/lib{whisper,parakeet,ggml,ggml-base,ggml-cpu,ggml-vulkan}.so* \
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
