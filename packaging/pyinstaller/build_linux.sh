#!/usr/bin/env bash
# Build a single-file Ubuntu binary. Run from the repo root:
#   ./packaging/pyinstaller/build_linux.sh
#
# Prerequisites:
#   - A whisper.cpp build with Vulkan enabled has been installed such that
#     `import pywhispercpp` loads the Vulkan-capable libwhisper.so (see
#     packaging/whisper_cpp/build_vulkan_linux.sh). The plain PyPI wheel is
#     CPU-only and will NOT provide real GPU acceleration. The source build's
#     CPU backend is -march=native: the binary only runs on CPUs with at
#     least the build machine's instruction set.
#   - System packages: libportaudio2 (mic capture), libvulkan1 or a vendor
#     Vulkan driver package if you want GPU acceleration (CPU fallback works
#     without it).
#   - The NDI SDK/Runtime redistributable is a separate operator install;
#     see README.md for the current licensing/bundling status.
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m pip install --upgrade pyinstaller
pyinstaller packaging/pyinstaller/narrator.spec --distpath dist --workpath build --noconfirm

echo "Built: dist/narrator"
