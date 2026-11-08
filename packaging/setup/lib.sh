# shellcheck shell=bash
# Shared helpers for packaging/setup/setup.sh and packaging/setup/obs.sh.
# Source it; it expects the caller to have set REPO_ROOT.
#
# Operator switches (environment, or the matching command-line flags):
#   DRY_RUN=1   print every state-changing command instead of running it
#               (read-only detection still runs); prompts auto-answer "yes"
#               so the whole flow is shown
#   YES=1       answer "yes" to every confirmation prompt (non-interactive).
#               It does NOT accept the NDI SDK licence: set NDI_ACCEPT_LICENSE=1
#               for that, deliberately a separate switch.
# Without a TTY on stdin (cron, CI, `make setup < /dev/null`) every prompt
# defaults to "no": nothing that needs consent happens unless YES=1.

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_B=$'\033[1;34m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_D=$'\033[2m'; C_0=$'\033[0m'
else
    C_B=''; C_G=''; C_Y=''; C_R=''; C_D=''; C_0=''
fi

say()  { printf '%s\n' "$*"; }
step() { printf '\n%s== %s ==%s\n' "$C_B" "$*" "$C_0"; }
info() { printf '%s==>%s %s\n' "$C_B" "$C_0" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
err()  { printf '%s[FAIL]%s %s\n' "$C_R" "$C_0" "$*" >&2; }
note() { printf '%s       %s%s\n' "$C_D" "$*" "$C_0"; }

is_true() { case "${1:-}" in 1|y|Y|yes|YES|Yes|true|TRUE|on|ON) return 0 ;; *) return 1 ;; esac; }
dry() { is_true "${DRY_RUN:-}"; }

quote_cmd() { local out='' a; for a in "$@"; do out+="$(printf '%q' "$a") "; done; printf '%s' "${out% }"; }

# run CMD...: print it, execute it unless DRY_RUN. No pipes/redirections;
# wrap those in `bash -c`.
run() {
    printf '  %s$ %s%s\n' "$C_D" "$(quote_cmd "$@")" "$C_0"
    dry && return 0
    "$@"
}

# ask "question": 0 = yes. Default no; YES=1 -> yes; DRY_RUN -> yes (so the
# whole plan prints); no TTY -> no.
ask() {
    if dry; then printf '%s [y/N] y (dry run)\n' "$1"; return 0; fi
    if is_true "${YES:-}"; then printf '%s [y/N] y (YES=1)\n' "$1"; return 0; fi
    if [ ! -t 0 ]; then
        printf '%s [y/N] n (no terminal; set YES=1 to auto-confirm)\n' "$1"
        return 1
    fi
    local a
    read -r -p "$1 [y/N] " a || return 1
    case "$a" in y|Y|yes|YES|Yes) return 0 ;; *) return 1 ;; esac
}

# A confirmation that must be an explicit human "yes" (or the named env var):
# YES=1 alone is not enough. ask_explicit "question" ENV_VAR_NAME
ask_explicit() {
    local q="$1" var="$2"
    if dry; then printf '%s [yes/N] yes (dry run)\n' "$q"; return 0; fi
    if is_true "${!var:-}"; then printf '%s [yes/N] yes (%s=1)\n' "$q" "$var"; return 0; fi
    if [ ! -t 0 ]; then
        printf '%s [yes/N] no (no terminal; set %s=1 to accept)\n' "$q" "$var"
        return 1
    fi
    local a
    read -r -p "$q Type 'yes' to accept: " a || return 1
    [ "$a" = yes ]
}

have() { command -v "$1" >/dev/null 2>&1; }

# ---- status ledger (setup.sh summary) ------------------------------------
LEDGER=()
# name|OK/SKIP/WARN/FAIL|detail. RAN = "OK if it really ran": a dry run shows
# it as SKIP so the summary never claims a command that was only printed.
record() {
    if [ "$2" = RAN ]; then
        if dry; then LEDGER+=("$1|SKIP|dry run, would: $3"); else LEDGER+=("$1|OK|$3"); fi
    else
        LEDGER+=("$1|$2|$3")
    fi
}
print_ledger() {
    local row name st detail col
    printf '\n%sSummary%s\n' "$C_B" "$C_0"
    for row in "${LEDGER[@]}"; do
        IFS='|' read -r name st detail <<<"$row"
        case "$st" in OK) col=$C_G ;; SKIP) col=$C_D ;; WARN) col=$C_Y ;; *) col=$C_R ;; esac
        printf '  %s%-5s%s %-28s %s\n' "$col" "$st" "$C_0" "$name" "$detail"
    done
}

# ---- platform ------------------------------------------------------------
os_field() { ( . /etc/os-release 2>/dev/null && eval "printf '%s' \"\${$1:-}\"" ); }
have_apt() { have apt-get && have dpkg-query; }
# Multi-arch packages print one status per architecture, hence grep -x.
pkg_installed() { dpkg-query -W -f='${db:Status-Status}\n' "$1" 2>/dev/null | grep -qx installed; }
host_arch() { uname -m; }

# ---- NDI runtime (system-wide libndi) ------------------------------------
# DistroAV (deb / usr-local OBS) loads libndi v6 from the system; narrator's
# own sender does not need it when the cyndilib wheel bundles libndi (checked
# by setup.sh). The one documented route is DistroAV's CI/libndi-get.sh, which
# downloads the NDI SDK from ndi.video and installs libndi into /usr/local/lib:
#   https://github.com/DistroAV/DistroAV/wiki/1.-Installation  (NDI Runtime)
LIBNDI_GET_URL="https://raw.githubusercontent.com/DistroAV/DistroAV/refs/heads/master/CI/libndi-get.sh"
# sha256 of the script revision reviewed on 2026-10-15 (it runs as root, so a
# changed script is never run unattended: it is shown/confirmed first).
LIBNDI_GET_REVIEWED_SHA256="cb6fca70e6ea6c9493a22f4f7f40e89fe83f9bcebb2231b23635bf386cd78683"
NDI_LICENSE_URL="http://ndi.link/ndisdk_license"
NDI_SDK_PAGE="https://ndi.video/for-developers/ndi-sdk/"

# Where an installed system libndi v6 is, empty if none.
system_libndi_path() {
    local d p
    if [ -n "${NDI_RUNTIME_DIR_V6:-}" ]; then
        for p in "$NDI_RUNTIME_DIR_V6"/libndi.so.6*; do [ -e "$p" ] && { printf '%s' "$p"; return 0; }; done
    fi
    for d in /usr/local/lib /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu; do
        for p in "$d"/libndi.so.6*; do [ -e "$p" ] && { printf '%s' "$p"; return 0; }; done
    done
    if have ldconfig; then
        p=$(ldconfig -p 2>/dev/null | awk '/libndi\.so\.6/ {print $NF; exit}')
        [ -n "$p" ] && { printf '%s' "$p"; return 0; }
    fi
    return 1
}

sha256_of() { sha256sum "$1" | awk '{print $1}'; }

# install_system_libndi: the documented libndi-get.sh route, with the NDI
# licence shown and explicitly accepted first. 0 = libndi is present after.
install_system_libndi() {
    if [ "$(host_arch)" != x86_64 ]; then
        warn "libndi-get.sh only handles x86_64. Install the NDI runtime for $(host_arch) from $NDI_SDK_PAGE"
        return 1
    fi
    say "  The documented NDI runtime route (DistroAV wiki) downloads the NDI SDK for"
    say "  Linux from downloads.ndi.tv and installs libndi into /usr/local/lib (sudo)."
    say "  Its installer's licence is accepted automatically by libndi-get.sh, so you"
    say "  must read it first: $NDI_LICENSE_URL"
    ask_explicit "  Accept the NDI SDK licence and install the NDI runtime?" NDI_ACCEPT_LICENSE || {
        warn "NDI runtime not installed. Manual route: $NDI_SDK_PAGE"
        return 1
    }
    local tmp script
    if dry; then
        run curl -fsSL -o "/tmp/narrator-XXXXXX/libndi-get.sh" "$LIBNDI_GET_URL"
        run sudo bash "/tmp/narrator-XXXXXX/libndi-get.sh" install
        return 0
    fi
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/narrator-libndi.XXXXXX") || return 1
    script="$tmp/libndi-get.sh"
    run curl -fsSL -o "$script" "$LIBNDI_GET_URL" || { err "download failed"; rm -rf "$tmp"; return 1; }
    local got; got=$(sha256_of "$script")
    if [ "$got" != "$LIBNDI_GET_REVIEWED_SHA256" ]; then
        warn "libndi-get.sh differs from the revision reviewed for this repo."
        note "reviewed: $LIBNDI_GET_REVIEWED_SHA256"
        note "got:      $got"
        note "read it before continuing: $script"
        if is_true "${YES:-}" || [ ! -t 0 ]; then
            err "refusing to run a changed root script unattended"
            rm -rf "$tmp"; return 1
        fi
        ask "  Run the changed script anyway?" || { rm -rf "$tmp"; return 1; }
    fi
    # The script is run exactly as the wiki shows (sudo ./libndi-get.sh install);
    # it ends with `ln -s libndi.so.6 libndi.so.5`, which fails on a re-run, so
    # success is judged by libndi being present afterwards.
    run sudo bash "$script" install || warn "libndi-get.sh exited non-zero; checking whether libndi is installed anyway"
    rm -rf "$tmp"
    system_libndi_path >/dev/null
}
