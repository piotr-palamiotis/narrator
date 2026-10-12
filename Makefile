# Makefile for the narrator project.
#
# Common targets:
#   make install   Create .venv, install narrator + dev deps, and build the
#                  Vulkan-enabled pywhispercpp the pip wheel no longer provides
#   make test      Run the test suite
#   make build     Build the standalone binary with PyInstaller (dist/narrator)
#   make run       Run narrator from the venv (pass args: make run ARGS="--no-gui")
#   make clean     Remove build artifacts (keeps .venv)
#   make distclean Also remove .venv
#
# One-time system prerequisites for the pywhispercpp source build:
#   sudo apt install -y cmake libvulkan-dev glslc spirv-headers spirv-tools

VENV   := .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

.PHONY: install asr-backend test build run clean distclean help

help:
	@echo "Targets: install, test, build, run, clean, distclean"

install: $(VENV)/.installed asr-backend

$(VENV)/.installed: pyproject.toml
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	touch $@

# The PyPI pywhispercpp wheel ships a CPU-only ggml (no libggml-vulkan.so)
# whose CPU path SIGILLs on pre-AVX CPUs, so ASR comes from a source build
# with GGML_VULKAN=1 instead. The script is idempotent: it exits immediately
# when the Vulkan backend is already installed in the venv.
asr-backend:
	@PYTHON="$(abspath $(VENV)/bin/python)" bash packaging/whisper_cpp/build_vulkan_linux.sh

test: install
	$(PYTHON) -m pytest tests/

build: install
	$(PIP) install --upgrade pyinstaller
	$(PYTHON) -m PyInstaller packaging/pyinstaller/narrator.spec \
		--distpath dist --workpath build --noconfirm
	@echo "Built: dist/narrator"

run: install
	$(PYTHON) -m narrator.main $(ARGS)

clean:
	rm -rf build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV)
