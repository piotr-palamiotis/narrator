#!/usr/bin/env bash
# Build and install pywhispercpp with a GPU ggml backend (Vulkan or CUDA)
# into the active virtual environment (or the one given via PYTHON=...).
#
# ASR_GPU picks the backend (despite this script's name):
#   vulkan  GGML_VULKAN=1 (AMD/Intel/NVIDIA via Vulkan)
#   cuda    GGML_CUDA=1   (NVIDIA; needs the CUDA toolkit, i.e. nvcc)
#   auto    (default) cuda when nvcc and a working nvidia-smi are found,
#           else vulkan
# Only one GPU backend is built: with both, whisper.cpp's gpu_device index
# would count the NVIDIA card twice (once per backend).
#
# Why a source build: the published PyPI wheel ships a CPU-only ggml (no
# libggml-vulkan.so), and its CPU path SIGILLs at whisper_init on pre-AVX
# CPUs. See packaging/whisper_cpp/README.md.
#
# What is built: the pywhispercpp bindings at PYWHISPERCPP_TAG with their
# whisper.cpp submodule moved to WHISPER_CPP_TAG (pywhispercpp v1.5.1 -- the
# newest release -- vendors whisper.cpp v1.8.5; v1.9.5 is API-compatible
# with those bindings, only additions). The installed version is
# "<pywhispercpp>+whispercpp.<whisper.cpp>", e.g. 1.5.1+whispercpp.1.9.5.
#
# Idempotent: exits early when exactly that version, with the Vulkan
# backend, is already installed. FORCE=1 rebuilds anyway.
#
# Going back to the previous build (pywhispercpp v1.5.1's own whisper.cpp):
#   WHISPER_CPP_TAG=v1.8.5 PYTHON=.venv/bin/python bash packaging/whisper_cpp/build_vulkan_linux.sh
#
# System prerequisites (Ubuntu/Debian, once):
#   vulkan: sudo apt install -y cmake libvulkan-dev glslc spirv-headers spirv-tools
#   cuda:   cmake, git, the NVIDIA driver and the CUDA toolkit (nvcc), e.g.
#           NVIDIA's cuda-toolkit-12-x/13-x packages (/usr/local/cuda) or
#           Ubuntu's nvidia-cuda-toolkit
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
PYWHISPERCPP_TAG="${PYWHISPERCPP_TAG:-v1.5.1}"
WHISPER_CPP_TAG="${WHISPER_CPP_TAG:-v1.9.5}"
PYWHISPERCPP_VER="${PYWHISPERCPP_TAG#v}+whispercpp.${WHISPER_CPP_TAG#v}"
# gitignored, inside the repo (cwd is the repo root); overridable for side-by-side builds
SRC="${PYWHISPERCPP_SRC:-third_party/pywhispercpp-src}"

# The CUDA toolkit's usual location is not on PATH by default.
for d in "${CUDA_HOME:-}" /usr/local/cuda; do
    if [ -n "$d" ] && [ -x "$d/bin/nvcc" ]; then
        export PATH="$d/bin:$PATH"
        CUDA_ROOT="$d"
        break
    fi
done
CUDA_ROOT="${CUDA_ROOT:-}"

ASR_GPU="${ASR_GPU:-auto}"
if [ "$ASR_GPU" = "auto" ]; then
    if command -v nvcc > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
        ASR_GPU=cuda
    else
        ASR_GPU=vulkan
    fi
    echo "ASR_GPU=auto -> $ASR_GPU"
fi
case "$ASR_GPU" in
    vulkan) GPU_LIB=libggml-vulkan.so; GPU_LABEL=Vulkan ;;
    cuda)   GPU_LIB=libggml-cuda.so;   GPU_LABEL=CUDA ;;
    *) echo "ERROR: ASR_GPU must be vulkan, cuda or auto (got '$ASR_GPU')." >&2; exit 2 ;;
esac

# Must run inside a virtual environment.
"$PYTHON" -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" || {
    echo "ERROR: not inside a virtual environment." >&2
    echo "  source .venv/bin/activate   (or set PYTHON=/path/to/venv/bin/python)" >&2
    exit 1
}

PIP="$PYTHON -m pip"
SITE=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
VBIN=$("$PYTHON" -c "import sys, os; print(os.path.join(sys.prefix, 'bin'))")
# The venv's ninja/patchelf must be on PATH even when the venv is not
# activated (make asr-backend passes only PYTHON=): ggml-vulkan builds its
# shader generator as a separate CMake project with -G Ninja.
export PATH="$VBIN:$PATH"

INSTALLED=$("$PYTHON" -c "import importlib.metadata as m; print(m.version('pywhispercpp'))" 2>/dev/null || true)
if [ "${FORCE:-0}" != "1" ] && [ "$INSTALLED" = "$PYWHISPERCPP_VER" ] \
        && compgen -G "$SITE/$GPU_LIB*" > /dev/null; then
    echo "pywhispercpp $INSTALLED with the $GPU_LABEL backend already installed -- nothing to do."
    exit 0
fi

if [ "$ASR_GPU" = vulkan ]; then
    for tool in cmake glslc git; do
        if ! command -v "$tool" > /dev/null; then
            echo "ERROR: '$tool' not found. The Vulkan source build needs system packages:" >&2
            echo "  sudo apt install -y cmake libvulkan-dev glslc spirv-headers spirv-tools git" >&2
            exit 1
        fi
    done
else
    for tool in cmake git nvcc; do
        if ! command -v "$tool" > /dev/null; then
            echo "ERROR: '$tool' not found. The CUDA source build needs cmake, git and the CUDA toolkit" >&2
            echo "  (nvcc on PATH, or installed under /usr/local/cuda or \$CUDA_HOME)." >&2
            exit 1
        fi
    done
    echo "CUDA toolkit: $(nvcc --version | tail -n 1)"
fi

echo "Installing Python build tools into the venv..."
$PIP install -q --upgrade ninja wheel setuptools-scm patchelf

echo "Cloning pywhispercpp ${PYWHISPERCPP_TAG} + whisper.cpp ${WHISPER_CPP_TAG} (shallow)..."
rm -rf "$SRC"
mkdir -p "$(dirname "$SRC")"
git clone --depth 1 --recursive --shallow-submodules \
    -b "$PYWHISPERCPP_TAG" https://github.com/absadiki/pywhispercpp "$SRC"
git -C "$SRC/whisper.cpp" fetch -q --depth 1 origin tag "$WHISPER_CPP_TAG"
git -C "$SRC/whisper.cpp" -c advice.detachedHead=false checkout -q "$WHISPER_CPP_TAG"

# whisper.cpp >= 1.9 forces CMAKE_LIBRARY_OUTPUT_DIRECTORY to <build>/bin,
# overriding the directory pywhispercpp's setup.py passes on the command
# line: libwhisper/libggml* then never reach the wheel (only the extension
# module does) and the import fails. Respect a value that is already set.
sed -i 's|^set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)$|if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)\n    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)\nendif()|' \
    "$SRC/whisper.cpp/CMakeLists.txt"

echo "Building the wheel with the $GPU_LABEL backend (this compiles whisper.cpp + GPU kernels; CUDA takes a while)..."
# pywhispercpp's setup.py turns every environment variable into a -D CMake
# define, so GGML_VULKAN=1 / GGML_CUDA=1 here is what enables the backend.
# CUDA: with GGML_NATIVE (default) and CUDA >= 11.6, ggml compiles for the
# GPUs present at build time (CMAKE_CUDA_ARCHITECTURES=native); set
# CMAKE_CUDA_ARCHITECTURES (e.g. 86) to build for another card. ggml's CPU
# backend is compiled with -march=native (GGML_NATIVE default): on a pre-AVX
# host that yields a pre-AVX build, but the wheel is then not portable to
# CPUs with fewer extensions than the build machine.
# SETUPTOOLS_SCM_PRETEND_VERSION: a shallow tag clone otherwise builds as
# version 0.0.0.
(
    cd "$SRC"
    if [ "$ASR_GPU" = vulkan ]; then
        GPU_ENV=(GGML_VULKAN=1)
    else
        GPU_ENV=(GGML_CUDA=1)
    fi
    rm -rf dist
    env -u GGML_VULKAN -u GGML_CUDA "${GPU_ENV[@]}" SETUPTOOLS_SCM_PRETEND_VERSION="$PYWHISPERCPP_VER" \
        "$PYTHON" -m pip wheel --no-build-isolation --no-deps -w dist .
)
WHEEL=$(ls "$SRC"/dist/pywhispercpp-"$PYWHISPERCPP_VER"-*.whl)

# Refuse to install a wheel that silently lacks the backend (e.g. glslc was
# missing so CMake skipped the Vulkan compile) or the native libraries.
# (Listing captured first: `unzip | grep -q` under pipefail fails at random
# when grep exits early and unzip gets SIGPIPE.)
LISTING=$(unzip -l "$WHEEL")
for lib in "$GPU_LIB" libwhisper.so libggml-cpu.so; do
    if ! grep -q "$lib" <<< "$LISTING"; then
        echo "ERROR: built wheel does not contain $lib." >&2
        exit 1
    fi
done

echo "Installing $(basename "$WHEEL")..."
$PIP install --force-reinstall --no-deps "$WHEEL"

# pywhispercpp's source build bakes the *pip temp build dir* into RUNPATH;
# that directory disappears right after install, leaving imports broken.
# Re-point every shipped library at its own directory ($ORIGIN). The CUDA
# backend also needs the toolkit's libcudart/libcublas: its lib dir is
# appended when the toolkit is not in the system's default library path.
RPATH='$ORIGIN'
if [ "$ASR_GPU" = cuda ] && [ -n "$CUDA_ROOT" ]; then
    for d in "$CUDA_ROOT/lib64" "$CUDA_ROOT/targets/x86_64-linux/lib"; do
        [ -d "$d" ] && RPATH="$RPATH:$d"
    done
fi
echo "Fixing RUNPATH to $RPATH..."
for f in "$SITE"/lib{whisper,parakeet,ggml,ggml-base,ggml-cpu,ggml-vulkan,ggml-cuda}.so* \
         "$SITE"/_pywhispercpp*.so; do
    [ -f "$f" ] && "$VBIN/patchelf" --set-rpath "$RPATH" "$f"
done
# One GPU backend only (see the top): pip's reinstall already removes the
# previous build's files; this catches a library copied in by hand.
if [ "$ASR_GPU" = cuda ]; then
    rm -f "$SITE"/libggml-vulkan.so*
else
    rm -f "$SITE"/libggml-cuda.so*
fi

# Verify the import works with no inherited environment (no LD_LIBRARY_PATH,
# no PYTHONPATH), which is how the ASR worker's spawn ends up running.
env -i HOME="$HOME" PATH="/usr/bin:/bin" "$PYTHON" -c "import pywhispercpp.model" \
    || { echo "ERROR: clean-environment import of pywhispercpp failed." >&2; exit 1; }

echo "OK: pywhispercpp $PYWHISPERCPP_VER with $GPU_LABEL backend installed."
echo "    Wheel kept at: $WHEEL"
