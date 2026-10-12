#!/usr/bin/env bash
# Build and install pywhispercpp with the Vulkan ggml backend into the active
# virtual environment (or the one given via PYTHON=...).
#
# Why a source build: the published PyPI wheel ships a CPU-only ggml (no
# libggml-vulkan.so), and its CPU path SIGILLs at whisper_init on pre-AVX
# CPUs. See packaging/whisper_cpp/README.md. Idempotent: exits early when a
# Vulkan-enabled pywhispercpp is already installed.
#
# System prerequisites (Ubuntu/Debian, once):
#   sudo apt install -y cmake libvulkan-dev glslc spirv-headers spirv-tools
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
PYWHISPERCPP_TAG="v1.5.1"
PYWHISPERCPP_VER="${PYWHISPERCPP_TAG#v}"
SRC="build/pywhispercpp-src"

# Must run inside a virtual environment.
"$PYTHON" -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" || {
    echo "ERROR: not inside a virtual environment." >&2
    echo "  source .venv/bin/activate   (or set PYTHON=/path/to/venv/bin/python)" >&2
    exit 1
}

PIP="$PYTHON -m pip"
SITE=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
VBIN=$("$PYTHON" -c "import sys, os; print(os.path.join(sys.prefix, 'bin'))")

if compgen -G "$SITE/libggml-vulkan.so*" > /dev/null; then
    echo "pywhispercpp Vulkan backend already installed -- nothing to do."
    exit 0
fi

for tool in cmake glslc; do
    if ! command -v "$tool" > /dev/null; then
        echo "ERROR: '$tool' not found. The Vulkan source build needs system packages:" >&2
        echo "  sudo apt install -y cmake libvulkan-dev glslc spirv-headers spirv-tools" >&2
        exit 1
    fi
done

echo "Installing Python build tools into the venv..."
$PIP install -q --upgrade ninja wheel setuptools-scm patchelf

echo "Cloning pywhispercpp ${PYWHISPERCPP_TAG} (shallow, shallow submodules)..."
rm -rf "$SRC"
mkdir -p "$(dirname "$SRC")"
git clone --depth 1 --recursive --shallow-submodules \
    -b "$PYWHISPERCPP_TAG" https://github.com/absadiki/pywhispercpp "$SRC"

echo "Building the wheel with GGML_VULKAN=1 (this compiles whisper.cpp + Vulkan shaders)..."
(
    cd "$SRC"
    SETUPTOOLS_SCM_PRETEND_VERSION="$PYWHISPERCPP_VER" \
        "$PYTHON" -m pip wheel --no-build-isolation --no-deps -w dist .
)
WHEEL=$(ls "$SRC"/dist/pywhispercpp-"$PYWHISPERCPP_VER"-*.whl)

# Refuse to install a wheel that silently lacks the backend (e.g. glslc was
# missing so CMake skipped the Vulkan compile).
if ! unzip -l "$WHEEL" | grep -q "libggml-vulkan.so"; then
    echo "ERROR: built wheel does not contain libggml-vulkan.so -- Vulkan was not compiled in." >&2
    exit 1
fi

echo "Installing $(basename "$WHEEL")..."
$PIP install --force-reinstall --no-deps "$WHEEL"

# pywhispercpp's source build bakes the *pip temp build dir* into RUNPATH;
# that directory disappears right after install, leaving imports broken.
# Re-point every shipped library at its own directory ($ORIGIN).
echo "Fixing RUNPATH to \$ORIGIN..."
for f in "$SITE"/lib{whisper,ggml,ggml-base,ggml-cpu,ggml-vulkan}.so* \
         "$SITE"/_pywhispercpp*.so; do
    [ -f "$f" ] && "$VBIN/patchelf" --set-rpath '$ORIGIN' "$f"
done

# Verify the import works with no inherited environment (no LD_LIBRARY_PATH,
# no PYTHONPATH), which is how the ASR worker's spawn ends up running.
env -i HOME="$HOME" PATH="/usr/bin:/bin" "$PYTHON" -c "import pywhispercpp.model" \
    || { echo "ERROR: clean-environment import of pywhispercpp failed." >&2; exit 1; }

echo "OK: pywhispercpp $PYWHISPERCPP_VER with Vulkan backend installed."
echo "    Wheel kept at: $WHEEL"
