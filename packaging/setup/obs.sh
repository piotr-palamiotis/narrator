#!/usr/bin/env bash
# make setup-obs: make sure OBS can receive narrator's NDI stream, i.e. the
# DistroAV plugin (plus the NDI runtime that route needs) in EVERY OBS install
# found, each by the route its own developers document. Routes are never
# mixed (no .deb plugin into a Flatpak OBS), an OBS is never uninstalled or
# replaced, and nothing is changed without the printed plan being confirmed.
#
# Usage: packaging/setup/obs.sh [--check] [--yes] [--dry-run] [--route ppa|flatpak]
#   --check     report only: detected installs, what would be done; changes nothing
#   --yes       answer yes to every prompt (env YES=1). The NDI SDK licence is a
#               separate explicit acceptance (env NDI_ACCEPT_LICENSE=1).
#   --dry-run   print every state-changing command instead of running it (DRY_RUN=1)
#   --route     how to install OBS when none is present (env OBS_ROUTE): flatpak
#               (default, Flathub) or ppa (Ubuntu only, ppa:obsproject/obs-studio)
# Without a TTY on stdin every prompt defaults to "no".
#
# Sources (re-checked 2026-10-15; the script follows them, so update it when
# they change):
#  * DistroAV wiki "1. Installation"
#    https://github.com/DistroAV/DistroAV/wiki/1.-Installation
#      - Flatpak: `flatpak install com.obsproject.Studio com.obsproject.Studio.Plugin.DistroAV`;
#        NDI runtime already bundled; for OBS 32+ (Avahi change) also
#        `sudo flatpak override com.obsproject.Studio --system-talk-name=org.freedesktop.Avahi`;
#        remove the legacy `com.obsproject.Studio.Plugin.NDI` first.
#      - Debian/Ubuntu: latest .deb from the GitHub Releases, NDI runtime v6+
#        installed first (`sudo ./libndi-get.sh install`, DistroAV's
#        CI/libndi-get.sh), then `sudo dpkg -i <deb>` and, if dependencies are
#        missing, `sudo apt-get -f install`. Known-issue workaround: symlinks
#        /usr/lib/x86_64-linux-gnu/obs-plugins/distroav.so ->
#        /usr/local/lib/obs-plugins/ and /usr/share/obs/obs-plugins/distroav/ ->
#        /usr/local/share/obs/obs-plugins/ (the wiki's only "manual" route; it
#        gives no source-build steps).
#      - Networking: `sudo apt install avahi-daemon ffmpeg`, enable + start
#        avahi-daemon; UFW: 5353/udp, 5959:5969 tcp+udp, 6960:6970 tcp+udp,
#        7960:7970 tcp+udp, 5960/tcp. FFmpeg >= 7 matters only for NDI HX
#        (narrator sends full NDI), so ffmpeg is only warned about, not installed.
#  * OBS KB https://obsproject.com/kb/linux-installation: Flathub
#    (com.obsproject.Studio) is the recommended route on non-Ubuntu distros;
#    Ubuntu: `sudo add-apt-repository ppa:obsproject/obs-studio` +
#    `sudo apt install obs-studio` (we add -y: add-apt-repository otherwise
#    waits for Enter); other distros: the OBS wiki install-instructions
#    (https://github.com/obsproject/obs-studio/wiki/install-instructions#linux).
#    The KB documents no Snap route and neither does DistroAV.
#  * Flathub setup for Ubuntu https://flathub.org/en/setup/Ubuntu:
#    `sudo apt install flatpak`, then
#    `flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo`.
#  * Release API https://api.github.com/repos/DistroAV/DistroAV/releases/latest:
#    tag_name, assets[].{name,browser_download_url,digest:"sha256:<hex>"}; the
#    body also lists "### Checksums". The Linux package is
#    `distroav-<tag>-x86_64-linux-gnu.deb` (x86_64 only; a `-dbgsym.ddeb` sits
#    next to it and is ignored). The deb's only OBS dependency is an unversioned
#    `obs-studio`; a versioned one would be checked against the installed OBS.
#
# Flatpak OBS in the *user* installation: the wiki's Avahi command is the
# system-wide one (sudo); for a user-installed OBS the equivalent
# `flatpak override --user ...` is used instead (no root) -- an adaptation of
# the documented command, not a different route.
set -uo pipefail

SETUP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SETUP_DIR/../.." && pwd)
# shellcheck source=packaging/setup/lib.sh
. "$SETUP_DIR/lib.sh"

CHECK=${CHECK:-}
OBS_ROUTE=${OBS_ROUTE:-flatpak}
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK=1 ;;
        --yes|-y) YES=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --route) shift; OBS_ROUTE=${1:-} ;;
        --route=*) OBS_ROUTE=${1#--route=} ;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "unknown argument: $1"; exit 2 ;;
    esac
    shift
done
case "$OBS_ROUTE" in flatpak|ppa) ;; *) err "OBS_ROUTE must be 'flatpak' or 'ppa' (got '$OBS_ROUTE')"; exit 2 ;; esac
export YES DRY_RUN

OBS_FLATPAK_ID=com.obsproject.Studio
DISTROAV_FLATPAK_ID=com.obsproject.Studio.Plugin.DistroAV
LEGACY_FLATPAK_ID=com.obsproject.Studio.Plugin.NDI
FLATHUB_URL=https://dl.flathub.org/repo/flathub.flatpakrepo
RELEASES_API=https://api.github.com/repos/DistroAV/DistroAV/releases/latest
DEB_PLUGIN=/usr/lib/x86_64-linux-gnu/obs-plugins/distroav.so
DEB_DATA=/usr/share/obs/obs-plugins/distroav
UFW_RULES=(5353/udp 5959:5969/tcp 5959:5969/udp 6960:6970/tcp 6960:6970/udp 7960:7970/tcp 7960:7970/udp 5960/tcp)

# ---- detection ------------------------------------------------------------
declare -A I_VER I_DESC I_PLUGIN I_ORIGIN I_LEGACY I_NOTE
INSTALLS=()              # ids: flatpak:system flatpak:user deb snap other
OBS_RUNNING=''
ACT_FN=(); ACT_DESC=(); ACT_OBS=()   # the plan: function+args, text, touches-OBS flag

dpkg_version() { dpkg-query -W -f='${Version}' "$1" 2>/dev/null; }

detect_flatpak() {
    have flatpak || return 0
    local inst id
    for inst in system user; do
        flatpak info "--$inst" "$OBS_FLATPAK_ID" >/dev/null 2>&1 || continue
        id="flatpak:$inst"; INSTALLS+=("$id")
        I_VER[$id]=$(flatpak info "--$inst" "$OBS_FLATPAK_ID" 2>/dev/null | awk -F': *' '/^[[:space:]]*Version:/ {print $2; exit}')
        I_ORIGIN[$id]=$(flatpak info "--$inst" --show-origin "$OBS_FLATPAK_ID" 2>/dev/null | head -1)
        I_DESC[$id]="Flatpak ($inst installation, remote ${I_ORIGIN[$id]:-?})"
        if flatpak info "--$inst" "$DISTROAV_FLATPAK_ID" >/dev/null 2>&1; then
            I_PLUGIN[$id]="present $(flatpak info "--$inst" "$DISTROAV_FLATPAK_ID" 2>/dev/null | awk -F': *' '/^[[:space:]]*Version:/ {print $2; exit}')"
        else
            I_PLUGIN[$id]=missing
        fi
        if flatpak info "--$inst" "$LEGACY_FLATPAK_ID" >/dev/null 2>&1; then I_LEGACY[$id]=1; fi
        # Per-user plugin dir of the sandboxed OBS.
        if plugin_dir_has_ndi "$HOME/.var/app/$OBS_FLATPAK_ID/config/obs-studio/plugins"; then
            I_NOTE[$id]="a per-user DistroAV/obs-ndi plugin dir exists in the sandbox config (left alone)"
        fi
    done
}

# Which package source the installed deb comes from, via apt-cache policy.
deb_origin() {
    local ver; ver=$(dpkg_version obs-studio)
    local pol src
    pol=$(apt-cache policy obs-studio 2>/dev/null)
    src=$(printf '%s\n' "$pol" | awk '/\*\*\*/ {f=1; next} f && /^        / {print; next} f {exit}')
    if printf '%s\n%s\n' "$ver" "$src" | grep -qiE 'obsproject'; then
        echo ppa
    elif printf '%s\n' "$src" | grep -qE 'https?://|ftp'; then
        echo distro
    else
        echo local
    fi
}

# Does a per-user OBS plugins dir hold a DistroAV / obs-ndi plugin?
plugin_dir_has_ndi() {
    local d
    for d in "$1"/*; do
        case "$(basename "$d")" in *[Dd]istro[Aa][Vv]*|*obs-ndi*) [ -e "$d" ] && return 0 ;; esac
    done
    return 1
}

detect_deb() {
    have_apt || return 0
    pkg_installed obs-studio || return 0
    INSTALLS+=(deb)
    I_VER[deb]=$(dpkg_version obs-studio)
    local origin; origin=$(deb_origin)
    case "$origin" in
        ppa) I_DESC[deb]="deb, ppa:obsproject/obs-studio build" ;;
        distro) I_DESC[deb]="deb, distribution build" ;;
        *) I_DESC[deb]="deb, origin unknown (installed from a local .deb?)" ;;
    esac
    I_ORIGIN[deb]=$origin
    local dv=''
    pkg_installed distroav && dv=$(dpkg_version distroav)
    if [ -n "$dv" ]; then
        I_PLUGIN[deb]="present $dv (deb package)"
    elif [ -e "$DEB_PLUGIN" ] || [ -e /usr/local/lib/obs-plugins/distroav.so ]; then
        I_PLUGIN[deb]="present (files only, no distroav package)"
    elif plugin_dir_has_ndi "$HOME/.config/obs-studio/plugins"; then
        I_PLUGIN[deb]="present (per-user, $HOME/.config/obs-studio/plugins)"
    else
        I_PLUGIN[deb]=missing
    fi
    if pkg_installed obs-ndi || [ -e /usr/lib/x86_64-linux-gnu/obs-plugins/obs-ndi.so ]; then
        I_NOTE[deb]="the old 'obs-ndi' plugin is installed too (DistroAV's predecessor); not touched"
    fi
}

detect_snap() {
    have snap || return 0
    snap list obs-studio >/dev/null 2>&1 || return 0
    INSTALLS+=(snap)
    I_VER[snap]=$(snap list obs-studio 2>/dev/null | awk 'NR==2 {print $2}')
    I_DESC[snap]="Snap"
    I_PLUGIN[snap]="n/a (DistroAV documents no Snap route)"
}

detect_other() {
    have obs || return 0
    local path; path=$(readlink -f "$(command -v obs)")
    case "$path" in /snap/*|/var/lib/flatpak/*|"$HOME"/.local/share/flatpak/*) return 0 ;; esac
    if have dpkg && dpkg -S "$path" >/dev/null 2>&1; then return 0; fi   # the deb, already covered
    INSTALLS+=(other)
    I_VER[other]=$(QT_QPA_PLATFORM=offscreen timeout 15 "$path" --version 2>/dev/null | head -1 | sed 's/^OBS Studio - //')
    I_DESC[other]="not packaged: $path"
    local prefix; prefix=$(dirname "$(dirname "$path")")
    I_ORIGIN[other]=$prefix
    if [ -e "$prefix/lib/obs-plugins/distroav.so" ] || [ -L "$prefix/lib/obs-plugins/distroav.so" ]; then
        if [ -e "$prefix/lib/obs-plugins/distroav.so" ]; then
            I_PLUGIN[other]="present ($prefix/lib/obs-plugins/distroav.so)"
        else
            I_PLUGIN[other]="broken symlink ($prefix/lib/obs-plugins/distroav.so)"
        fi
    elif plugin_dir_has_ndi "$HOME/.config/obs-studio/plugins"; then
        I_PLUGIN[other]="present (per-user, $HOME/.config/obs-studio/plugins)"
    else
        I_PLUGIN[other]=missing
    fi
}

detect_running() {
    if pgrep -x obs >/dev/null 2>&1 || pgrep -x obs-studio >/dev/null 2>&1; then OBS_RUNNING=1; fi
    if have flatpak && flatpak ps --columns=application 2>/dev/null | grep -q "^$OBS_FLATPAK_ID\$"; then OBS_RUNNING=1; fi
    return 0
}

detect_all() {
    INSTALLS=(); I_VER=(); I_DESC=(); I_PLUGIN=(); I_ORIGIN=(); I_LEGACY=(); I_NOTE=()
    detect_flatpak; detect_deb; detect_snap; detect_other; detect_running
}

# ---- network prerequisites ------------------------------------------------
avahi_state() {   # prints: missing | stopped | disabled | ok
    if ! pkg_installed avahi-daemon && ! have avahi-daemon; then echo missing; return; fi
    systemctl is-active --quiet avahi-daemon 2>/dev/null || { echo stopped; return; }
    systemctl is-enabled --quiet avahi-daemon 2>/dev/null || { echo disabled; return; }
    echo ok
}

ufw_active() {
    have ufw || return 1
    systemctl is-active --quiet ufw 2>/dev/null || return 1
    grep -qiE '^ENABLED=yes' /etc/ufw/ufw.conf 2>/dev/null
}

# Rules still missing, when they can be read without a password prompt;
# otherwise all of them (ufw skips an existing rule itself).
ufw_missing_rules() {
    local status='' r
    if ! is_true "$CHECK" && ! dry && sudo -n true 2>/dev/null; then
        status=$(sudo -n ufw status 2>/dev/null)
    fi
    for r in "${UFW_RULES[@]}"; do
        if [ -n "$status" ] && printf '%s\n' "$status" | grep -qE "^${r//./\\.}[[:space:]]+ALLOW"; then continue; fi
        printf '%s\n' "$r"
    done
}

ffmpeg_warning() {
    have ffmpeg || return 0
    local major; major=$(ffmpeg -version 2>/dev/null | sed -n '1s/^ffmpeg version [^0-9]*\([0-9]\+\).*/\1/p')
    if [ -n "$major" ] && [ "$major" -lt 7 ]; then
        warn "FFmpeg $major.x: NDI HX in OBS needs FFmpeg >= 7 (older versions can crash). Narrator sends full NDI, so this does not affect it."
    fi
}

# ---- DistroAV release -----------------------------------------------------
REL_STATE=''   # '' not fetched, ok, fail
REL_TAG=''; REL_URL=''; REL_NAME=''; REL_SHA=''
fetch_release() {
    [ -n "$REL_STATE" ] && { [ "$REL_STATE" = ok ]; return; }
    REL_STATE=fail
    if ! have curl || ! have python3; then warn "curl and python3 are needed to query the DistroAV releases"; return 1; fi
    local arch json
    case "$(host_arch)" in x86_64) arch=x86_64-linux-gnu ;; aarch64) arch=aarch64-linux-gnu ;; *) arch=$(host_arch)-linux-gnu ;; esac
    json=$(curl -fsSL --retry 2 -H 'Accept: application/vnd.github+json' "$RELEASES_API" 2>/dev/null) || {
        warn "could not query $RELEASES_API"; return 1; }
    local out
    out=$(printf '%s' "$json" | python3 -I -c '
import json, re, sys
arch = sys.argv[1]
d = json.load(sys.stdin)
tag = d.get("tag_name", "")
pat = re.compile(r"^distroav-.*-" + re.escape(arch) + r"\.deb$")
for a in d.get("assets", []):
    if pat.match(a["name"]):
        sha = (a.get("digest") or "")
        sha = sha.split(":", 1)[1] if sha.startswith("sha256:") else ""
        if not sha:
            m = re.search(r"^\s*" + re.escape(a["name"]) + r":\s*([0-9a-f]{64})\s*$", d.get("body") or "", re.M)
            sha = m.group(1) if m else ""
        print("\t".join([tag, a["browser_download_url"], a["name"], sha]))
        break
else:
    print(tag)
' "$arch") || { warn "unexpected release API response"; return 1; }
    IFS=$'\t' read -r REL_TAG REL_URL REL_NAME REL_SHA <<<"$out"
    [ -n "$REL_TAG" ] || { warn "release API returned no tag"; return 1; }
    REL_STATE=ok
    [ -n "$REL_URL" ] || { warn "DistroAV $REL_TAG has no .deb for $arch (no documented route for this CPU)"; REL_STATE=fail; return 1; }
    return 0
}

# ---- actions (each is a function; arguments carry no spaces) ---------------
add_action() { ACT_FN+=("$1"); ACT_DESC+=("$2"); ACT_OBS+=("$3"); }

act_libndi() { install_system_libndi; }

act_distroav_deb() {
    fetch_release || { err "cannot determine the DistroAV release"; return 1; }
    local tmp deb
    say "  DistroAV $REL_TAG: $REL_NAME"
    if [ -z "$REL_SHA" ]; then
        warn "the release publishes no checksum for $REL_NAME; it cannot be verified"
        ask "  Install it anyway?" || return 1
    fi
    if dry; then
        run curl -fsSL -o "/tmp/narrator-XXXXXX/$REL_NAME" "$REL_URL"
        say "  (sha256 ${REL_SHA:-unavailable} would be verified, and the .deb's obs-studio dependency compared with the installed OBS)"
        run sudo dpkg -i "/tmp/narrator-XXXXXX/$REL_NAME"
        say "  if that reports missing dependencies:"
        run sudo apt-get -f install -y
        return 0
    fi
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/narrator-distroav.XXXXXX") || return 1
    deb="$tmp/$REL_NAME"
    run curl -fsSL -o "$deb" "$REL_URL" || { err "download failed"; rm -rf "$tmp"; return 1; }
    if [ -n "$REL_SHA" ] && [ "$(sha256_of "$deb")" != "$REL_SHA" ]; then
        err "checksum mismatch for $REL_NAME (expected $REL_SHA); not installing"
        rm -rf "$tmp"; return 1
    fi
    [ -n "$REL_SHA" ] && ok "sha256 verified ($REL_SHA)"
    local need have_v
    need=$(dpkg-deb -f "$deb" Depends 2>/dev/null | grep -oE 'obs-studio \(>= [^)]+\)' | head -1 | sed -E 's/.*>= ([^)]+)\).*/\1/')
    have_v=$(dpkg_version obs-studio)
    if [ -n "$need" ] && [ -n "$have_v" ] && ! dpkg --compare-versions "$have_v" ge "$need"; then
        err "$REL_NAME needs obs-studio >= $need, installed is $have_v; not installing"
        rm -rf "$tmp"; return 1
    fi
    if [ -n "$need" ]; then note "obs-studio $have_v satisfies the package's dependency (>= $need)"
    else note "the package declares no minimum OBS version (plain 'obs-studio' dependency)"; fi
    local rc=0
    if ! run sudo dpkg -i "$deb"; then
        say "  dpkg reported problems (usually missing dependencies); the wiki's next step:"
        run sudo apt-get -f install -y || rc=1
    fi
    rm -rf "$tmp"
    return $rc
}

# /usr/local OBS: the wiki's symlinks from the deb's files.
act_usrlocal_links() {
    local prefix=$1
    if ! dry && [ ! -e "$DEB_PLUGIN" ]; then
        err "$DEB_PLUGIN is missing (the deb route did not provide it); not linking"
        return 1
    fi
    run sudo mkdir -p "$prefix/lib/obs-plugins" "$prefix/share/obs/obs-plugins" || return 1
    run sudo ln -sfn "$DEB_PLUGIN" "$prefix/lib/obs-plugins/distroav.so" || return 1
    run sudo ln -sfn "$DEB_DATA" "$prefix/share/obs/obs-plugins/distroav"
}

act_flatpak_plugin() {
    local inst=$1 remote=${2:-flathub}
    if [ "$inst" = system ]; then
        run sudo flatpak install --system -y "$remote" "$DISTROAV_FLATPAK_ID"
    else
        run flatpak install --user -y "$remote" "$DISTROAV_FLATPAK_ID"
    fi
}

act_flatpak_legacy_remove() {
    local inst=$1
    say "  The legacy NDI plugin ($LEGACY_FLATPAK_ID) is installed in the $inst installation."
    say "  The DistroAV wiki says to remove it before installing DistroAV (it is the old, pre-6 plugin)."
    ask "  Remove $LEGACY_FLATPAK_ID?" || { warn "legacy plugin left in place"; return 0; }
    if [ "$inst" = system ]; then run sudo flatpak remove --system -y "$LEGACY_FLATPAK_ID"
    else run flatpak remove --user -y "$LEGACY_FLATPAK_ID"; fi
}

act_flatpak_avahi() {
    local inst=$1
    if [ "$inst" = system ]; then
        run sudo flatpak override "$OBS_FLATPAK_ID" --system-talk-name=org.freedesktop.Avahi
    else
        run flatpak override --user "$OBS_FLATPAK_ID" --system-talk-name=org.freedesktop.Avahi
    fi
}

act_avahi() {
    local state; state=$(avahi_state)
    if [ "$state" = missing ]; then
        have_apt || { err "avahi-daemon is missing and this is not an apt system; install it with your package manager"; return 1; }
        run sudo apt-get install -y avahi-daemon || return 1
    fi
    run sudo systemctl enable --now avahi-daemon
}

act_ufw() {
    local rules=() r
    mapfile -t rules < <(ufw_missing_rules)
    [ ${#rules[@]} -gt 0 ] || { ok "UFW already allows the NDI ports"; return 0; }
    say "  UFW is active. The DistroAV wiki's mDNS/NDI rules still to add:"
    for r in "${rules[@]}"; do say "    sudo ufw allow $r"; done
    ask "  Add these ${#rules[@]} UFW rules?" || { warn "UFW rules not added; NDI discovery/streams may be blocked"; return 0; }
    local rc=0
    for r in "${rules[@]}"; do run sudo ufw allow "$r" || rc=1; done
    return $rc
}

act_apt_flatpak() { run sudo apt-get install -y flatpak; }
act_flathub_remote() { run sudo flatpak remote-add --system --if-not-exists flathub "$FLATHUB_URL"; }
act_flatpak_obs() {
    run sudo flatpak install --system -y flathub "$OBS_FLATPAK_ID" "$DISTROAV_FLATPAK_ID"
}
act_ppa_obs() {
    if ! have add-apt-repository; then run sudo apt-get install -y software-properties-common || return 1; fi
    run sudo add-apt-repository -y ppa:obsproject/obs-studio || return 1
    run sudo apt-get install -y obs-studio
}

# ---- planning ---------------------------------------------------------------
need_libndi() { [ -z "$(system_libndi_path)" ]; }

plan_deb_plugin() {   # install or upgrade the deb DistroAV for the deb OBS
    local id=$1 state=${I_PLUGIN[$1]}
    if [[ "$state" == missing ]]; then
            fetch_release || { add_action : "DistroAV .deb: cannot be planned (release lookup failed or no package for this CPU)" 0; return 1; }
            need_libndi && add_action act_libndi "NDI runtime v6 (libndi) via DistroAV's libndi-get.sh -> /usr/local/lib (sudo, NDI licence)" 0
            add_action act_distroav_deb "DistroAV $REL_TAG: download the .deb from GitHub, verify sha256, sudo dpkg -i (+ apt-get -f install if needed)" 1
    elif [[ "$state" == "present "*"(deb package)" ]]; then
            local cur=${state#present }; cur=${cur%% *}
            if fetch_release && dpkg --compare-versions "$cur" lt "$REL_TAG"; then
                need_libndi && add_action act_libndi "NDI runtime v6 (libndi) via libndi-get.sh" 0
                add_action act_distroav_deb "DistroAV: upgrade $cur -> $REL_TAG (.deb, sha256-verified, sudo dpkg -i)" 1
            else
                need_libndi && add_action act_libndi "NDI runtime v6 (libndi) is missing: DistroAV cannot load without it; libndi-get.sh" 0
            fi
    elif [[ "$state" == present* ]]; then
            need_libndi && add_action act_libndi "NDI runtime v6 (libndi) is missing: libndi-get.sh" 0
    fi
    return 0
}

plan_installs() {
    local id inst remote major any_install='' any_acting=''
    for id in "${INSTALLS[@]}"; do
        case "$id" in
            flatpak:*)
                any_install=1; any_acting=1; inst=${id#flatpak:}; remote=${I_ORIGIN[$id]:-flathub}
                if [ -n "${I_LEGACY[$id]:-}" ]; then
                    add_action "act_flatpak_legacy_remove $inst" "[$id] remove the legacy $LEGACY_FLATPAK_ID (asks again before removing)" 1
                fi
                if [ "${I_PLUGIN[$id]}" = missing ]; then
                    add_action "act_flatpak_plugin $inst $remote" "[$id] flatpak install $DISTROAV_FLATPAK_ID (NDI runtime is bundled)" 1
                fi
                major=${I_VER[$id]%%.*}
                if [ -z "$major" ] || ! [ "$major" -eq "$major" ] 2>/dev/null || [ "$major" -ge 32 ]; then
                    if ! flatpak override --system --show "$OBS_FLATPAK_ID" 2>/dev/null | grep -q 'org.freedesktop.Avahi=talk' && \
                       ! flatpak override --user --show "$OBS_FLATPAK_ID" 2>/dev/null | grep -q 'org.freedesktop.Avahi=talk'; then
                        add_action "act_flatpak_avahi $inst" "[$id] OBS 32+ Avahi fix: flatpak override $OBS_FLATPAK_ID --system-talk-name=org.freedesktop.Avahi" 1
                    fi
                fi
                ;;
            deb)
                any_install=1; any_acting=1
                plan_deb_plugin deb
                ;;
            snap)
                any_install=1   # reported only: nothing is planned for a Snap
                ;;
            other)
                any_install=1; any_acting=1
                if [ "${I_PLUGIN[other]}" = missing ] || [[ "${I_PLUGIN[other]}" == broken* ]]; then
                    if [ "${I_ORIGIN[other]}" != /usr/local ]; then
                        I_NOTE[other]="prefix ${I_ORIGIN[other]} is not /usr/local: DistroAV documents no route for it; install manually (see the wiki)"
                    elif [ -z "${I_PLUGIN[deb]:-}" ] && ! pkg_installed obs-studio; then
                        I_NOTE[other]="the documented route links the DistroAV .deb's files, but no deb obs-studio is installed and 'dpkg -i' would pull one in (apt-get -f install) -- not done automatically; see the wiki's symlink steps"
                    else
                        # the deb OBS's own plan (above) provides the plugin files these link to
                        add_action "act_usrlocal_links /usr/local" "[other] link the deb's plugin into /usr/local/lib/obs-plugins and /usr/local/share/obs/obs-plugins (wiki symlinks)" 1
                    fi
                fi
                ;;
        esac
    done
    if [ -z "$any_install" ]; then plan_no_obs; fi
    # Network prerequisites, once, whenever a route we act on is (or will be) in the picture.
    if [ -n "$any_acting" ] || [ ${#ACT_FN[@]} -gt 0 ]; then
        case "$(avahi_state)" in
            missing) add_action act_avahi "avahi-daemon: install + enable + start (mDNS discovery of NDI sources)" 0 ;;
            stopped|disabled) add_action act_avahi "avahi-daemon: enable + start (installed but $(avahi_state))" 0 ;;
        esac
        if ufw_active; then
            if [ -n "$(ufw_missing_rules)" ]; then
                add_action act_ufw "UFW is active: show the wiki's NDI/mDNS rules and ask before adding" 0
            fi
        fi
    fi
}

plan_no_obs() {
    info "No OBS install found."
    case "$OBS_ROUTE" in
        ppa)
            if [ "$(os_field ID)" != ubuntu ] || ! have_apt; then
                warn "OBS_ROUTE=ppa is the OBS KB's Ubuntu route; this is not Ubuntu. Use the Flatpak route or the OBS wiki: https://github.com/obsproject/obs-studio/wiki/install-instructions#linux"
                return
            fi
            add_action act_ppa_obs "install OBS from ppa:obsproject/obs-studio (OBS KB's Ubuntu route; sudo)" 1
            if fetch_release; then
                need_libndi && add_action act_libndi "NDI runtime v6 (libndi) via DistroAV's libndi-get.sh" 0
                add_action act_distroav_deb "DistroAV $REL_TAG .deb (sha256-verified, sudo dpkg -i)" 1
            fi
            ;;
        flatpak)
            if ! have flatpak; then
                if have_apt; then
                    add_action act_apt_flatpak "install Flatpak (sudo apt install flatpak, per flathub.org/en/setup/Ubuntu; a re-login may be needed afterwards)" 1
                else
                    warn "No Flatpak and no apt: set Flatpak up per https://flathub.org/en/setup, or follow the OBS wiki https://github.com/obsproject/obs-studio/wiki/install-instructions#linux. Nothing changed."
                    return
                fi
            fi
            if ! have flatpak || ! flatpak remotes --system --columns=name 2>/dev/null | grep -qx flathub; then
                add_action act_flathub_remote "add the Flathub remote (system installation)" 1
            fi
            add_action act_flatpak_obs "flatpak install flathub $OBS_FLATPAK_ID $DISTROAV_FLATPAK_ID (system; NDI runtime bundled)" 1
            add_action "act_flatpak_avahi system" "OBS 32+ Avahi fix: sudo flatpak override $OBS_FLATPAK_ID --system-talk-name=org.freedesktop.Avahi" 1
            ;;
    esac
}

# ---- reporting --------------------------------------------------------------
print_report() {
    step "OBS installs"
    if [ ${#INSTALLS[@]} -eq 0 ]; then say "  none detected (checked: flatpak system+user, deb, snap, 'obs' on PATH)"; fi
    local id
    for id in "${INSTALLS[@]}"; do
        printf '  %-16s OBS %-14s %s\n' "[$id]" "${I_VER[$id]:-?}" "${I_DESC[$id]}"
        printf '  %-16s DistroAV: %s\n' '' "${I_PLUGIN[$id]}"
        [ -n "${I_LEGACY[$id]:-}" ] && printf '  %-16s legacy %s installed\n' '' "$LEGACY_FLATPAK_ID"
        [ -n "${I_NOTE[$id]:-}" ] && printf '  %-16s note: %s\n' '' "${I_NOTE[$id]}"
    done
    if [ -n "$OBS_RUNNING" ]; then warn "OBS is running (it must be closed before anything is changed)"; fi
    for id in "${INSTALLS[@]}"; do
        case "$id" in
            snap) warn "[snap] DistroAV documents no Snap route and a Snap sandbox cannot load the plugin. Recommended: the Flatpak (flatpak install flathub $OBS_FLATPAK_ID $DISTROAV_FLATPAK_ID). Nothing will be changed." ;;
        esac
    done
    say ""
    say "  NDI runtime (system libndi v6, for non-Flatpak OBS): ${C_D}$(system_libndi_path || echo 'not found')${C_0}"
    say "  Avahi (mDNS): $(avahi_state)"
    if ufw_active; then say "  UFW: active"; else say "  UFW: not active (or not installed)"; fi
    ffmpeg_warning
}

print_plan() {
    step "Plan"
    if [ ${#ACT_FN[@]} -eq 0 ]; then ok "nothing to do: every OBS found can already receive NDI (or has no documented route)"; return; fi
    local i
    for i in "${!ACT_FN[@]}"; do
        [ "${ACT_FN[$i]}" = : ] && { warn "${ACT_DESC[$i]}"; continue; }
        printf '  %d. %s\n' "$((i + 1))" "${ACT_DESC[$i]}"
    done
    say "  Never done: uninstalling or replacing any OBS, mixing routes, touching installs not listed above."
}

# ---- verification -----------------------------------------------------------
verify() {
    step "Verification"
    detect_all
    local id bad=0
    for id in "${INSTALLS[@]}"; do
        case "${I_PLUGIN[$id]}" in
            present*) ok "[$id] DistroAV: ${I_PLUGIN[$id]}" ;;
            n/a*) note "[$id] ${I_PLUGIN[$id]}" ;;
            *) err "[$id] DistroAV: ${I_PLUGIN[$id]}"; bad=1 ;;
        esac
    done
    [ ${#INSTALLS[@]} -eq 0 ] && { warn "no OBS install detected after the run (a fresh Flatpak may need a re-login before 'obs' is on PATH)"; }
    note "the plugin is loaded when OBS starts (needs a display), so that part is not checked here"
    if pgrep -x narrator >/dev/null 2>&1 || systemctl --user is-active --quiet narrator 2>/dev/null; then
        local py="$REPO_ROOT/.venv/bin/python"
        if [ -x "$py" ]; then
            info "narrator is running: discovering its NDI source over mDNS (cyndilib Finder)"
            "$py" "$SETUP_DIR/ndi_check.py" find --timeout 5 || warn "narrator's source was not discovered: check avahi-daemon, the firewall and that NDI is enabled in narrator"
        else
            warn "narrator is running but $py is missing; skipping mDNS discovery"
        fi
    else
        note "narrator is not running: start it ('make install' / 'make run') and re-run 'make setup-obs ARGS=--check' to verify NDI discovery"
    fi
    return $bad
}

# ---- main -------------------------------------------------------------------
main() {
    is_true "$CHECK" && info "check mode: reporting only, nothing is changed"
    dry && info "dry run: state-changing commands are printed, not run"
    detect_all
    print_report
    plan_installs
    print_plan
    if is_true "$CHECK"; then
        say ""; say "${#ACT_FN[@]} action(s) would be taken. (--check: nothing was changed)"
        if pgrep -x narrator >/dev/null 2>&1 || systemctl --user is-active --quiet narrator 2>/dev/null; then
            [ -x "$REPO_ROOT/.venv/bin/python" ] && "$REPO_ROOT/.venv/bin/python" "$SETUP_DIR/ndi_check.py" find --timeout 4
        fi
        return 0
    fi
    [ ${#ACT_FN[@]} -gt 0 ] || { verify; return $?; }

    local i touches=''
    for i in "${!ACT_OBS[@]}"; do [ "${ACT_OBS[$i]}" = 1 ] && touches=1; done
    if [ -n "$touches" ] && [ -n "$OBS_RUNNING" ] && ! dry; then
        err "OBS is running. Close it, then run 'make setup-obs' again. Nothing was changed."
        return 3
    fi
    say ""
    ask "Carry out this plan?" || { say "Nothing changed."; return 0; }

    local rc=0 libndi_failed=''
    for i in "${!ACT_FN[@]}"; do
        [ "${ACT_FN[$i]}" = : ] && continue
        step "$((i + 1)). ${ACT_DESC[$i]}"
        if [ "${ACT_FN[$i]}" = act_distroav_deb ] && [ -n "$libndi_failed" ]; then
            warn "skipped: the plugin cannot load without the NDI runtime (the previous step failed)"
            rc=1; continue
        fi
        # shellcheck disable=SC2086  # intentional word splitting: "fn arg arg"
        if ! ${ACT_FN[$i]}; then
            err "step failed: ${ACT_DESC[$i]}"
            rc=1
            [ "${ACT_FN[$i]}" = act_libndi ] && libndi_failed=1
        fi
    done
    if dry; then
        say ""; say "Dry run complete: nothing was changed."
        return $rc
    fi
    verify || rc=1
    return $rc
}

main
