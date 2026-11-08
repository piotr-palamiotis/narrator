# Makefile for the narrator project.
#
# Common targets:
#   make setup     One-shot, idempotent machine bootstrap (packaging/setup/setup.sh):
#                  apt packages (sudo, shown first) -> render/video groups -> NDI
#                  runtime check -> venv + Vulkan build -> performance preset +
#                  model -> OBS + DistroAV -> make install -> smoke check.
#                  OBS=skip, OBS_ROUTE=ppa|flatpak, SERVICE=skip, PRESET=skip,
#                  TESTS=skip, YES=1 (no prompts), DRY_RUN=1 (print commands only)
#   make setup-obs OBS + DistroAV only (packaging/setup/obs.sh); ARGS=--check only
#                  reports what it found and would do
#   make install   Create .venv, install narrator + dev deps, build the
#                  GPU-enabled pywhispercpp the pip wheel does not provide
#                  (ASR_GPU=auto|cuda|vulkan, below), then install + start the
#                  systemd user service (below).
#                  SERVICE=skip does only the first two (no systemd changes).
#   make unit      Render packaging/systemd/narrator.service.in only (no systemctl)
#   make uninstall Stop + disable the service and remove its unit file (the
#                  repo, .venv, config, models and var/ stay; linger stays on)
#   make status    systemctl --user status narrator
#   make logs      Follow the service's journal (journalctl --user -u narrator)
#   make test      Run the test suite
#   make configure Install the [tune] extra and run `narrator-tune configure`:
#                  search this machine for the model + decode settings that
#                  best match scripts/test_audio/ and write them into config
#                  (pass flags: make configure ARGS="--quick --dry-run")
#   make pick-preset   Benchmark this machine and set asr.preset in the
#                  service's config/config.yaml (`narrator --pick-preset`;
#                  ARGS="--dry-run" / "--download")
#   make tune-presets  Maintainer mode: `narrator-tune presets` (rewrites
#                  src/narrator/config/presets.yaml; hours per tier)
#   make build     Build the standalone binary with PyInstaller (dist/narrator)
#   make run       Run narrator from the venv (pass args: make run ARGS="--headless")
#   make clean     Remove build artifacts (keeps .venv)
#   make distclean Also remove .venv
#
# One-time system prerequisites for the pywhispercpp source build:
#   Vulkan: sudo apt install -y cmake libvulkan-dev glslc spirv-headers spirv-tools
#   CUDA:   cmake + the NVIDIA driver + the CUDA toolkit (nvcc)
#
# ASR_GPU picks the GPU backend of that build: auto (default: cuda when nvcc
# and nvidia-smi work, else vulkan), cuda or vulkan. Then set asr.unit to
# cuda:<n> / vulkan:<n> (or leave it on auto, which prefers CUDA).

VENV   := .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

# systemd user service. Rendered from packaging/systemd/narrator.service.in
# with this checkout's absolute path. UNIT_DIR can be overridden to render
# into a scratch dir (e.g. for `systemd-analyze --user verify`).
ASR_GPU  ?= auto
SERVICE  ?= install
UNIT_DIR ?= $(HOME)/.config/systemd/user
UNIT     := $(UNIT_DIR)/narrator.service

.PHONY: setup setup-obs install deps asr-backend service unit uninstall status logs test tune-deps configure pick-preset tune-presets build run clean distclean help

help:
	@echo "Targets: setup, setup-obs, install, unit, uninstall, status, logs, test, configure, pick-preset, tune-presets, build, run, clean, distclean"
	@echo "         (make install SERVICE=skip: no systemd service)"

# Bootstrap scripts, not Makefile recipes: they print each sudo command first and
# honour DRY_RUN / YES / OBS / OBS_ROUTE / SERVICE from the command line or the
# environment (command-line variables reach the recipe's environment).
setup:
	@bash packaging/setup/setup.sh

setup-obs:
	@bash packaging/setup/obs.sh $(ARGS)

# `deps` = everything but the service; test/build/run/configure depend on it
# so they never touch systemd.
deps: $(VENV)/.installed asr-backend

install: deps service

$(VENV)/.installed: pyproject.toml
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	touch $@

# The PyPI pywhispercpp wheel ships a CPU-only ggml (no GPU backend) whose
# CPU path SIGILLs on pre-AVX CPUs, so ASR comes from a source build with
# GGML_CUDA=1 or GGML_VULKAN=1 instead (ASR_GPU). The script is idempotent:
# it exits immediately when that backend is already installed in the venv.
asr-backend:
	@ASR_GPU="$(ASR_GPU)" PYTHON="$(abspath $(VENV)/bin/python)" bash packaging/whisper_cpp/build_vulkan_linux.sh

# Render the unit with the repo's absolute path.
unit:
	@case "$(CURDIR)" in *[[:space:]]*) \
		echo "ERROR: repo path contains whitespace ($(CURDIR)); the unit template does not quote paths." >&2; exit 1;; esac
	mkdir -p "$(UNIT_DIR)"
	sed 's|@REPO@|$(CURDIR)|g' packaging/systemd/narrator.service.in > "$(UNIT)"
	@echo "Rendered $(UNIT)"

# The service part of `make install` (skip with SERVICE=skip). Needs a
# systemd user session; changes the system only by: writing the unit under
# ~/.config/systemd/user, enabling + starting it, and enabling linger for
# $(USER) so it also starts at boot without a login.
service:
ifeq ($(SERVICE),skip)
	@echo "SERVICE=skip: not installing the systemd user service."
else
	@echo "Installing the narrator systemd user service:"
	@echo "  1. render packaging/systemd/narrator.service.in -> $(UNIT)"
	@echo "  2. systemctl --user daemon-reload && systemctl --user enable --now narrator"
	@echo "  3. loginctl enable-linger $(USER)   (start at boot without a login)"
	@echo "Config: $(CURDIR)/config/config.yaml (created from the defaults on first start)."
	@if [ -f "$(HOME)/.config/narrator/config.yaml" ] && [ ! -f config/config.yaml ]; then \
		echo "Note: an older config exists at ~/.config/narrator/config.yaml; the service does NOT use it."; \
		echo "      To keep those settings: cp ~/.config/narrator/config.yaml config/config.yaml"; fi
	@$(MAKE) --no-print-directory unit
	systemctl --user daemon-reload
	systemctl --user enable --now narrator
	loginctl enable-linger "$(USER)"
	@echo "Done. 'make status' / 'make logs'; 'make uninstall' to remove the service."
endif

uninstall:
	-systemctl --user disable --now narrator
	rm -f "$(UNIT)"
	-systemctl --user daemon-reload
	@echo "Service removed. The repo, .venv, config/, models/ and var/ are untouched; linger is left as is (loginctl disable-linger $(USER) to undo)."

status:
	systemctl --user status narrator --no-pager

logs:
	journalctl --user -u narrator -f -n 100

test: deps
	$(PYTHON) -m pytest tests/

# The tuning tool is not part of the app: its extra is installed here, not
# by `make install`.
tune-deps: deps
	$(PIP) install -e ".[tune]"

configure: tune-deps
	$(VENV)/bin/narrator-tune configure $(ARGS)

pick-preset: deps
	$(VENV)/bin/narrator --config config/config.yaml --pick-preset $(ARGS)

tune-presets: tune-deps
	$(VENV)/bin/narrator-tune presets $(ARGS)

build: deps
	$(PIP) install --upgrade pyinstaller
	$(PYTHON) -m PyInstaller packaging/pyinstaller/narrator.spec \
		--distpath dist --workpath build --noconfirm
	@echo "Built: dist/narrator"

run: deps
	$(PYTHON) -m narrator.main $(ARGS)

clean:
	rm -rf build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV)
