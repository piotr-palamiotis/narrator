#!/usr/bin/env bash
# make setup: one-shot, idempotent machine bootstrap (Ubuntu/Debian first;
# other distros get the package list to install by hand).
#
# Steps, in order. Only steps that need root use sudo; each prints the exact
# sudo command and asks first (YES=1 answers for you). Re-running skips what
# is already in place.
#   1. system packages (apt; skipped per package via dpkg -s)
#   2. render/video group membership (re-login needed after a change)
#   3. NDI runtime for narrator's own sender (cyndilib bundles libndi)
#   4. venv + GPU pywhispercpp build (CUDA/Vulkan)   (make deps; ASR_GPU=auto|cuda|vulkan)
#   5. performance preset + its model                (narrator --pick-preset --download)
#   6. OBS + DistroAV                                (packaging/setup/obs.sh; OBS=skip)
#   7. narrator + systemd user service               (make install; SERVICE=skip)
#   8. smoke check: devices, Vulkan, NDI, tests, service
#
# Switches (environment or flags): DRY_RUN=1/--dry-run prints every
# state-changing command instead of running it; YES=1/--yes confirms prompts
# (never the NDI licence: NDI_ACCEPT_LICENSE=1); OBS=skip; OBS_ROUTE=ppa|flatpak;
# SERVICE=skip; PRESET=skip (don't benchmark); REPICK=1 (benchmark again even
# though a preset was already picked); TESTS=skip (no pytest in the smoke check).
# With no TTY on stdin every prompt is answered "no".
set -uo pipefail

SETUP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SETUP_DIR/../.." && pwd)
cd "$REPO_ROOT" || exit 1
# shellcheck source=packaging/setup/lib.sh
. "$SETUP_DIR/lib.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --yes|-y) YES=1 ;;
        -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "unknown argument: $1"; exit 2 ;;
    esac
    shift
done
export YES DRY_RUN

MAKE_CMD=${MAKE:-make}
VENV_PY="$REPO_ROOT/.venv/bin/python"
NARRATOR="$REPO_ROOT/.venv/bin/narrator"
ME=${USER:-$(id -un)}

# What `make asr-backend` + audio + tests need (README "What cannot live in the
# repo"), plus the toolchain/venv basics the build takes for granted.
REQUIRED_PKGS=(cmake libvulkan-dev glslc spirv-headers spirv-tools libvulkan1 mesa-vulkan-drivers
               libportaudio2 gstreamer1.0-plugins-base gstreamer1.0-plugins-good)
BASIC_PKGS=(build-essential git curl)   # + python3-venv, see step_apt

finish() {   # finish EXIT_CODE
    print_ledger
    exit "$1"
}
abort() { err "$1"; record "$2" FAIL "$1"; say "Setup stopped: later steps need this one."; finish 1; }

if [ "$(id -u)" = 0 ]; then
    err "Run 'make setup' as your normal user, not root: the venv and config belong to you, and sudo is used for the steps that need it."
    exit 1
fi
dry && info "DRY RUN: nothing below is changed; state-changing commands are only printed."
say "Repository: $REPO_ROOT"

# ---- 1. system packages ------------------------------------------------------
step_apt() {
    step "1/8 System packages"
    local missing=() p
    if ! have_apt; then
        warn "Not an apt system. Install the equivalents of these yourself:"
        say "  ${REQUIRED_PKGS[*]} ${BASIC_PKGS[*]}"
        record "system packages" WARN "not apt: install manually"
        return
    fi
    for p in "${REQUIRED_PKGS[@]}" "${BASIC_PKGS[@]}"; do
        pkg_installed "$p" || missing+=("$p")
    done
    # python3-venv is a metapackage (python3.12-venv may be there without it):
    # judge by whether venv + ensurepip actually work.
    python3 -c 'import venv, ensurepip' 2>/dev/null || missing+=(python3-venv)
    if [ ${#missing[@]} -eq 0 ]; then
        ok "all ${#REQUIRED_PKGS[@]}+${#BASIC_PKGS[@]} packages already installed"
        record "system packages" SKIP "already installed"
        return
    fi
    say "Missing: ${missing[*]}"
    say "This step needs root. It will run exactly:"
    say "  sudo apt-get update"
    say "  sudo apt-get install -y ${missing[*]}"
    if ! ask "Run it?"; then
        warn "skipped; the Vulkan build will fail until these are installed"
        record "system packages" WARN "declined: ${missing[*]}"
        return
    fi
    if run sudo apt-get update && run sudo apt-get install -y "${missing[@]}"; then
        record "system packages" RAN "install ${#missing[@]}: ${missing[*]}"
    else
        abort "apt failed" "system packages"
    fi
}

# ---- 2. groups ----------------------------------------------------------------
step_groups() {
    step "2/8 render/video groups (GPU access without root)"
    local g need=() relog=() members
    for g in render video; do
        getent group "$g" >/dev/null || continue
        members=$(getent group "$g" | cut -d: -f4)
        if ! printf ',%s,' "$members" | grep -q ",$ME,"; then
            need+=("$g")
        elif ! id -nG | tr ' ' '\n' | grep -qx "$g"; then
            relog+=("$g")      # in /etc/group, but this login session predates it
        fi
    done
    if [ ${#need[@]} -gt 0 ]; then
        local csv; csv=$(IFS=,; echo "${need[*]}")
        say "$ME is not in: ${need[*]}. This step needs root and will run exactly:"
        say "  sudo usermod -aG $csv $ME"
        if ask "Run it?"; then
            if run sudo usermod -aG "$csv" "$ME"; then
                warn "Log out and back in (or reboot) for the new group(s) to apply; until then the GPU may not be usable by new processes."
                record "groups" RAN "add to $csv (re-login needed)"
            else
                record "groups" WARN "usermod failed"
            fi
        else
            record "groups" WARN "declined: not in ${need[*]}"
        fi
    elif [ ${#relog[@]} -gt 0 ]; then
        warn "$ME is in ${relog[*]} but this session predates it: log out and back in."
        record "groups" WARN "RE-LOGIN NEEDED for ${relog[*]}"
    else
        ok "$ME is in render and video (where they exist)"
        record "groups" SKIP "already a member"
    fi
}

# ---- 3. NDI runtime for narrator's own sender -----------------------------------
# narrator sends through cyndilib. The PyPI manylinux wheel bundles libndi
# (site-packages/cyndilib.libs/libndi-*.so.6.x, verified on cyndilib 0.1.1), so
# no system NDI runtime is needed for narrator itself, and nothing is
# downloaded from ndi.video here. A system libndi v6 matters only for OBS
# installs that are not Flatpak (obs.sh installs it via DistroAV's
# libndi-get.sh, after the NDI licence is accepted). If the installed wheel
# turns out not to carry a loadable libndi, the same route is offered here.
step_ndi() {
    step "3/8 NDI runtime for narrator's sender"
    if [ ! -x "$VENV_PY" ]; then
        ok "cyndilib is installed in step 4; its PyPI Linux wheel bundles libndi (verified again in the smoke check)"
        note "no download from ndi.video is needed for narrator; NDI redistribution terms: $NDI_LICENSE_URL"
        record "NDI runtime (sender)" SKIP "deferred until the venv exists"
        return
    fi
    local out rc
    out=$("$VENV_PY" "$SETUP_DIR/ndi_check.py" runtime 2>&1); rc=$?
    say "$out" | sed 's/^/  /'
    case $rc in
        0) ok "narrator's NDI sender loads its bundled runtime"
           record "NDI runtime (sender)" OK "bundled in the cyndilib wheel" ;;
        2) record "NDI runtime (sender)" SKIP "cyndilib not installed yet (venv step)" ;;
        *) warn "cyndilib cannot open its NDI runtime."
           if [ -n "$(system_libndi_path)" ]; then
               note "a system libndi exists at $(system_libndi_path) but cyndilib could not use it either; see the message above (unsupported CPU?)"
               record "NDI runtime (sender)" WARN "Finder failed"
           elif install_system_libndi; then
               record "NDI runtime (sender)" OK "system libndi installed"
           else
               record "NDI runtime (sender)" WARN "no runtime: $NDI_SDK_PAGE"
           fi ;;
    esac
}

# ---- 4. venv + GPU ASR build (ASR_GPU: auto = CUDA when nvcc works, else Vulkan) ----
step_venv() {
    step "4/8 Python venv + GPU pywhispercpp build (ASR_GPU=${ASR_GPU:-auto})"
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        abort "python3 >= 3.11 is required ($(python3 --version 2>&1))" "venv + ASR build"
    fi
    say "(first run compiles whisper.cpp with the GPU kernels: several minutes, longer for CUDA)"
    if run "$MAKE_CMD" --no-print-directory deps; then
        record "venv + ASR build" RAN "make deps"
    else
        abort "'make deps' failed (see above; common cause: missing system packages from step 1)" "venv + ASR build"
    fi
}

# ---- 5. performance preset -----------------------------------------------------------
step_preset() {
    step "5/8 Performance preset + model (benchmark on this machine)"
    local marker="$REPO_ROOT/var/setup/preset-picked"
    if [ "${PRESET:-}" = skip ]; then
        record "preset" SKIP "PRESET=skip"; return
    fi
    if [ -f "$marker" ] && ! is_true "${REPICK:-}"; then
        ok "a preset was already picked on $(cat "$marker" 2>/dev/null) (REPICK=1 to benchmark again)"
        record "preset" SKIP "already picked"; return
    fi
    say "Benchmarks candidate models on this machine and downloads the winner's model into models/."
    say "This can take many minutes (and a multi-GB download). A failure here does not stop the setup."
    if run "$NARRATOR" --config config/config.yaml --pick-preset --download; then
        if ! dry; then mkdir -p "$(dirname "$marker")" && date +%F >"$marker"; fi
        record "preset" RAN "narrator --pick-preset --download"
    else
        warn "preset picking failed (setup continues). Retry with: make pick-preset ARGS=--download"
        record "preset" WARN "failed; run: make pick-preset ARGS=--download"
    fi
}

# ---- 6. OBS ------------------------------------------------------------------------------
step_obs() {
    step "6/8 OBS + DistroAV (so OBS can receive narrator's NDI stream)"
    if [ "${OBS:-}" = skip ]; then
        say "OBS=skip: not touching OBS. Later: make setup-obs (ARGS=--check to only report)."
        record "OBS + DistroAV" SKIP "OBS=skip"; return
    fi
    local rc=0
    bash "$SETUP_DIR/obs.sh" || rc=$?
    case $rc in
        0) record "OBS + DistroAV" RAN "see above" ;;
        3) warn "OBS is running; close it and run: make setup-obs"
           record "OBS + DistroAV" WARN "OBS running: close it, then make setup-obs" ;;
        *) warn "OBS step reported problems (exit $rc); re-run: make setup-obs"
           record "OBS + DistroAV" WARN "exit $rc; re-run make setup-obs" ;;
    esac
}

# ---- 7. install ------------------------------------------------------------------------------
step_install() {
    step "7/8 make install (narrator + systemd user service)"
    if [ "${SERVICE:-install}" = skip ]; then say "SERVICE=skip: no systemd changes."; fi
    if run "$MAKE_CMD" --no-print-directory install; then
        record "make install" RAN "make install ${SERVICE:+SERVICE=$SERVICE}"
    else
        record "make install" FAIL "see above"
    fi
}

# ---- 8. smoke check ---------------------------------------------------------------------------
smoke() {
    step "8/8 Smoke check"
    local out rc py=$VENV_PY
    [ -x "$py" ] || py=python3

    # Microphones + NDI sources
    if [ -x "$NARRATOR" ]; then
        if dry; then run "$NARRATOR" --list-devices; record "smoke: --list-devices" SKIP "dry run"
        else
            out=$("$NARRATOR" --list-devices 2>&1); rc=$?
            say "$out" | sed 's/^/  /'
            if [ $rc -eq 0 ]; then record "smoke: --list-devices" OK "exit 0"
            else record "smoke: --list-devices" WARN "exit $rc"; fi
        fi
    else
        record "smoke: --list-devices" WARN "no $NARRATOR (venv missing)"
    fi

    # Vulkan: loader + ICD only (no vulkan-tools): vulkan_devices.py uses ctypes.
    out=$(python3 "$SETUP_DIR/vulkan_devices.py" 2>&1); rc=$?
    say "$out" | sed 's/^/  vulkan: /'
    if [ $rc -eq 0 ]; then
        if printf '%s\n' "$out" | grep -qE '^(discrete|integrated|virtual)-gpu'; then
            record "smoke: Vulkan" OK "$(printf '%s\n' "$out" | grep -cE '^(discrete|integrated|virtual)-gpu') GPU device(s)"
        else
            record "smoke: Vulkan" WARN "only a software device (llvmpipe): use asr.unit: cpu or install a GPU driver"
        fi
    else
        record "smoke: Vulkan" WARN "no Vulkan device (install libvulkan1 + mesa-vulkan-drivers; check render/video groups)"
    fi

    # NDI runtime for the sender, now that the venv exists.
    if [ -x "$VENV_PY" ]; then
        out=$("$VENV_PY" "$SETUP_DIR/ndi_check.py" runtime 2>&1); rc=$?
        say "$out" | sed 's/^/  ndi: /'
        if [ $rc -eq 0 ]; then record "smoke: NDI sender runtime" OK "Finder opens"
        else record "smoke: NDI sender runtime" WARN "exit $rc"; fi
    fi

    # Tests
    if [ "${TESTS:-}" = skip ]; then record "smoke: pytest" SKIP "TESTS=skip"
    elif [ -x "$VENV_PY" ]; then
        if run "$VENV_PY" -m pytest tests/ -q; then record "smoke: pytest" RAN "tests/ passed"
        else record "smoke: pytest" WARN "failures (see above)"; fi
    else
        record "smoke: pytest" WARN "no venv"
    fi

    # Service
    if [ "${SERVICE:-install}" = skip ]; then record "smoke: service" SKIP "SERVICE=skip"
    elif run systemctl --user status narrator --no-pager; then record "smoke: service" RAN "service is up"
    else record "smoke: service" WARN "not running: make logs"; fi
}

step_apt
step_groups
step_ndi
step_venv
step_preset
step_obs
step_install
smoke

say ""
dry && say "Dry run complete: nothing was changed."
rc=0
for row in "${LEDGER[@]}"; do case "$row" in *"|FAIL|"*) rc=1 ;; esac; done
say "Next: 'make status' / 'make logs'; the web UI: .venv/bin/narrator --open-ui. Re-login if the groups line says so."
finish $rc
