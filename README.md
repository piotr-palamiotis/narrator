# narrator

Live speech captioning, streamed as an NDI RGBA video source (transparent
background) so any NDI-capable app (OBS, vMix, NDI Studio Monitor, ...) can
receive and display it as a lower-third overlay. Speech-to-text runs locally
via [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (through
[pywhispercpp](https://github.com/absadiki/pywhispercpp)), on CPU, NVIDIA
GPU, or AMD GPU (via Vulkan) with automatic fallback. Default configuration
targets Polish transcription with the `large-v3` model.

See `/home/xxx/.claude/plans/i-want-you-to-shimmering-duckling.md` for the
full design rationale.

## Prerequisites

- **Python 3.11+**
- **NDI runtime**: narrator's sender uses the `libndi` bundled in the
  `cyndilib` wheel, so nothing is installed for narrator itself (confirm
  current NDI SDK redistribution terms before shipping a build that bundles
  it). A receiving OBS needs the DistroAV plugin, which `make setup-obs`
  installs (see Quick start).
- **Ubuntu**: `libportaudio2` for microphone capture
  (`sudo apt install libportaudio2`). For GPU acceleration, a working
  Vulkan driver (`mesa-vulkan-drivers` for AMD/Intel, or the NVIDIA driver's
  bundled Vulkan support) -- CPU (`asr.unit: cpu`) works with no extra
  packages. The processing unit (`asr.unit`: `auto`, `cpu`, `vulkan:<n>`,
  `cuda:<n>`) is picked in the web UI's ASR tab, which lists this machine's
  CPU and GPUs with live load, VRAM, temperature and power; `auto` takes the
  recommended one (a CUDA card with the CUDA build, else a discrete
  Vulkan GPU, else the CPU, an integrated GPU last). Old configs with
  `asr.device`/`asr.gpu_device` are converted automatically. The configured
  unit is used as-is: if the GPU cannot be
  initialized, ASR stops and the web UI's problems panel says why (with a
  "Use CPU instead" button where the CPU backend runs); the app never
  switches to CPU by itself, and it keeps running (NDI output, manual
  captions, web UI).
- **Windows**: microphone capture and Vulkan (via GPU vendor drivers) work
  out of the box; no extra system packages needed.
- **pywhispercpp must be built from source with Vulkan.** The published
  `pywhispercpp` wheel no longer ships the Vulkan ggml backend, and its
  CPU-only ggml SIGILLs at `whisper_init` on pre-AVX CPUs (measured on a
  Pentium G4560, 2026-10-03). `make install` builds and installs the
  Vulkan-enabled wheel automatically (pywhispercpp's `v1.5.1` bindings over
  whisper.cpp `v1.9.5`, installed as `1.5.1+whispercpp.1.9.5`)
  (`packaging/whisper_cpp/build_vulkan_linux.sh`), or run that script
  manually inside an activated venv. The runtime still needs the Vulkan
  loader and driver ICD only -- `libvulkan1` plus `mesa-vulkan-drivers` on
  AMD/Intel, the proprietary driver's own ICD on NVIDIA (no CUDA toolkit
  needed) -- and *not* the `vulkan-tools` diagnostic package.
  **NVIDIA with CUDA**: install the CUDA toolkit (`nvcc`), then
  `make install ASR_GPU=cuda` (or `make asr-backend ASR_GPU=cuda` in an
  existing venv; the default `ASR_GPU=auto` already picks CUDA when `nvcc`
  and `nvidia-smi` work). With `asr.unit: auto` (the default) or
  `cuda:<n>`, large-v3 then runs on the CUDA card. One GPU backend per
  build: a CUDA build has no `vulkan:<n>` units.
  `packaging/whisper_cpp/README.md` has the details.

## Quick start: `make setup` (Ubuntu/Debian)

```bash
make setup                    # everything below, in order, asking before each sudo command
make setup DRY_RUN=1          # print every state-changing command instead of running it
make setup OBS=skip           # leave OBS alone
make setup OBS_ROUTE=ppa      # only when no OBS exists: install it from ppa:obsproject/obs-studio
                              #   instead of Flathub (the default; Ubuntu only)
make setup YES=1              # answer yes to every prompt (no TTY: prompts default to no)
make setup-obs                # just the OBS + DistroAV step; ARGS=--check only reports
```

`make setup` (`packaging/setup/setup.sh`) is idempotent; re-running skips what
is in place. Steps: (1) missing apt packages (Vulkan loader/driver, build
toolchain, PortAudio, GStreamer; the exact `sudo apt-get install` line is
printed first), (2) `render`/`video` group membership (log out and in
afterwards), (3) the NDI runtime check, (4) venv + Vulkan pywhispercpp build,
(5) `narrator --pick-preset --download` (a failure is reported, not fatal; skip
with `PRESET=skip`, redo with `REPICK=1`), (6) OBS + DistroAV, (7) `make
install` (`SERVICE=skip` for no systemd changes), (8) a smoke check
(`narrator --list-devices`, Vulkan device enumeration, `pytest tests/`,
service status). Only the steps that need root use sudo.

**NDI runtime.** narrator's own sender needs nothing from ndi.video: the
`cyndilib` Linux wheel bundles `libndi` (setup verifies it loads). A system
`libndi` v6 is needed only for a non-Flatpak OBS (deb/PPA, `/usr/local`); the
step uses DistroAV's documented `libndi-get.sh`, which downloads the NDI SDK
from ndi.video and accepts its licence for you, so it asks you to accept
http://ndi.link/ndisdk_license first (`NDI_ACCEPT_LICENSE=1` in
non-interactive runs; `YES=1` alone does not accept it).

**OBS + DistroAV** (`make setup-obs`, `packaging/setup/obs.sh`). Finds every
OBS install (Flatpak system/user, deb with PPA vs distro build, Snap, other),
prints what it found and what it plans, and refuses to change anything while
OBS is running. Each install gets the route its developers document, never
mixed: Flatpak OBS -> the `com.obsproject.Studio.Plugin.DistroAV` Flatpak
(NDI bundled; plus the OBS 32+ Avahi override; the legacy `...Plugin.NDI` is
removed only after asking); deb OBS -> the latest DistroAV `.deb` from GitHub
(sha256-verified) after the NDI runtime, via `dpkg -i`; `/usr/local` OBS -> the
wiki's symlinks; Snap OBS -> no documented route, reported only. It also
checks `avahi-daemon` and, when UFW is active, shows the NDI/mDNS port rules
and asks before adding them. It never uninstalls or replaces an OBS.
`make setup-obs ARGS=--check` only reports; `DRY_RUN=1` prints the commands.
Afterwards, with narrator running, it checks over mDNS that narrator's NDI
source is discoverable (the OBS side itself needs a display, so that part is
not checked headless). Sources: the DistroAV wiki "1. Installation" and the
OBS Linux install KB, as quoted in the script's header.

## Install as a service (Linux)

```bash
make install                  # venv + Vulkan ASR build + systemd user service
make install SERVICE=skip     # same, without touching systemd
make status | make logs       # systemctl --user status / journalctl --user -u narrator -f
make uninstall                # disable + remove the unit only; the repo stays
```

`make install` renders `packaging/systemd/narrator.service.in` with this
checkout's absolute path into `~/.config/systemd/user/narrator.service`,
runs `systemctl --user daemon-reload && systemctl --user enable --now
narrator`, and `loginctl enable-linger $USER` so the service starts at boot
without a login. It runs `<repo>/.venv/bin/narrator --headless --config
<repo>/config/config.yaml`, restarts on failure with a growing delay (at most
5 starts per 5 minutes), logs to journald, and on `systemctl stop` (SIGTERM)
shuts down cleanly: web UI, NDI stream, ASR worker, audio, transcript file.
Setup problems (no GPU, missing model or microphone, no NDI runtime, an ASR
worker that keeps crashing) never stop the service: it keeps running and the
web UI shows them (see "Problems panel"), so "failure" restarts happen only
when the process itself dies (e.g. an unreadable config.yaml at startup).
The service is operated from the browser (see "Web UI and remote control").
It is a *user* service because the microphone (PipeWire/PulseAudio) lives in
the user session and the Vulkan render node only needs the user in the
`render`/`video` groups. Add `--log-file` to the unit's `ExecStart` for a
rotating `var/log/narrator.log` (plus `narrator-asr.log` for the ASR worker)
in addition to the journal. A Windows service (NSSM or a service wrapper) is
out of scope for now and a possible follow-up.

## Where things live

Everything the app writes stays inside the repo folder (all gitignored), so
a checkout is the whole installation:

| Path | Contents |
|---|---|
| `config/config.yaml` | the app config; created from the defaults on first start when passed with `--config` (the service does) |
| `models/` | whisper ggml models; downloads land here, and a model already in the old `~/.local/share/pywhispercpp/models` is reused rather than downloaded again |
| `var/` | `log/` (optional file log), `transcripts/` (default transcript dir), `tune/` (narrator-tune cache and reports) |
| `third_party/` | the pywhispercpp/whisper.cpp source and build tree used by `make asr-backend` |

A bare `narrator` (no `--config`) still uses the platform config dir
(`~/.config/narrator/config.yaml`); a service started from a repo that was
using that file needs `cp ~/.config/narrator/config.yaml config/config.yaml`
once. A non-editable install or the PyInstaller binary is not in a checkout
and uses the platform user dirs for everything.

What **cannot** live in the repo, and must be installed on the machine:

- apt packages (Ubuntu): Vulkan loader and driver (`libvulkan1`,
  `mesa-vulkan-drivers`), the build toolchain for the ASR build (`cmake`,
  `libvulkan-dev`, `glslc`, `spirv-headers`, `spirv-tools`, a C++ compiler),
  PortAudio (`libportaudio2`) and GStreamer (`gstreamer1.0-plugins-base`,
  `-good`; used to decode audio for tuning/tests).
- a system NDI runtime (`libndi` v6, https://ndi.video) only for a non-Flatpak
  OBS with DistroAV; `make setup-obs` installs it through DistroAV's
  `libndi-get.sh` after you accept the NDI licence. It is never fetched
  silently or bundled in the repo.

## Setup (development)

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\Activate.ps1 on Windows
pip install -e .
narrator --list-devices           # sanity-check microphone enumeration
narrator                          # runs with default config; opens the web UI in a browser if there is a display
```

On first run, a default config is copied to the path given with `--config`
(`narrator --config config/config.yaml`), or, with no flag, to the platform's
user config directory (e.g. `~/.config/narrator/config.yaml` on Linux,
`%APPDATA%\narrator\config.yaml` on Windows). Edit it by hand or through the
web UI's Save button -- both paths go through the same validation and hot-reload
(the running app picks up changes without a restart, except for canvas
size/NDI source name and audio device, which need a restart).

## Web UI and remote control

narrator is operated from a browser (the Tkinter GUI is gone). The admin UI
listens on `http://127.0.0.1:8765` (`web.bind`, `web.port`) -- loopback
only; from another machine use an SSH tunnel (`ssh -L 8765:127.0.0.1:8765
host`) or Tailscale, never a LAN bind.

```bash
narrator --open-ui        # open the running instance's UI, already logged in
narrator --headless       # (the service) never opens a browser; the web UI still runs
narrator --no-web         # no web UI and no OSC input (--no-gui: old spelling)
```

Each start creates a random login token: it is logged ("login token: ...",
`journalctl --user -u narrator | grep "login token"`) and stored in
`var/web-token.json` (mode 0600). `--open-ui` uses it; the login page also
accepts it pasted. Requests with a foreign `Host` (DNS rebinding) or a
cross-site `Origin` (CSRF) are refused.

The page has everything the Tk GUI had: live preview (with unsaved display
settings previewed before they go on air), mic level, ASR Active/Inactive and
backend, auto/manual caption mode with a manual entry, Clear, pause/resume,
every config section (language whitelist, text filters, vocabulary and saved
vocabulary lists, model and preset, blocklist, NDI, transcript, shortcuts,
API tokens, OSC) and a status/log view. Save validates with the config schema
(errors appear next to the fields) and writes `config.yaml`, which the running
app picks up like a hand edit.

**Problems panel.** Above the page, a list of what is wrong with the setup,
most severe first, each with fix buttons: the configured processing unit is
missing or unusable (no Vulkan GPU, no Vulkan loader, a pre-AVX CPU with the
PyPI wheel), the model is not installed or not downloadable, the ASR worker
crashed and was stopped, the microphone or its channels are missing, the NDI
audio source is not on the network, the NDI runtime or caption output failed,
disk space for recordings is low. Checks run at startup, on every config
change and every 30 s, on their own thread. Nothing in the list stops the
app, and nothing is changed automatically: *Retry* restarts only the
affected part (ASR worker, audio input or NDI output), *Use CPU instead*
(offered only where the CPU backend runs) writes `asr.unit: cpu` to
`config.yaml` like a Save, *Use vulkan:0 (...)* switches to another unit
(e.g. the card you picked, after a card added in front of it shifted the
indices -- that is reported, never followed silently), and the settings
buttons open the matching tab.
The same items are in `GET /api/state` (`problems`) and the WebSocket state;
`POST /api/fix` (`{"action": "asr.retry"}`, admin session only -- not the
API tokens) applies a fix. An ASR worker that crashes 3 times within 15 s of
starting stays stopped until a fix or an ASR config change.

**Keyboard shortcuts** (configurable under `web.shortcuts`; `?` shows them):
Esc clear, F2 auto/manual, F4 focus the manual entry, Enter send, F8
pause/resume captions, F9 recording on/off. They work while
typing in a field. Ctrl+N/T/W and other keys the browser keeps for itself
are refused.

**Control API.** `GET /api/state`, `POST /api/clear`, `POST /api/mode`
(`{"mode": "auto"|"manual"|"toggle"}`), `POST /api/manual-text`
(`{"text": "..."}`), `POST /api/pause` (`{"paused": true|false|"toggle"}`),
`POST /api/recording` (`{"recording": true|false|"toggle"}`). Parameters may also be
given as query strings. Integrations authenticate with a named token from
`web.api_tokens` (add one on the *Web & remote* tab; delete it to revoke):
`Authorization: Bearer <token>`. Those tokens reach only these endpoints.

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/api/clear
curl -X POST -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:8765/api/mode?mode=toggle"
```

**Stream Deck via Bitfocus Companion** (the *Generic: HTTP Requests*
module, running on the same machine or reaching it through the tunnel). Base
URL `http://127.0.0.1:8765`, and on every action the header
`{"Authorization": "Bearer <token>"}` and content type `application/json`:

| Button | Action | URL | Body |
|---|---|---|---|
| CLEAR | POST | `/api/clear` | `{}` |
| AUTO/MANUAL | POST | `/api/mode` | `{"mode": "toggle"}` |
| AUTO | POST | `/api/mode` | `{"mode": "auto"}` |
| MANUAL | POST | `/api/mode` | `{"mode": "manual"}` |
| PAUSE | POST | `/api/pause` | `{"paused": "toggle"}` |
| "Prosimy o ciszę" | POST | `/api/manual-text` | `{"text": "Prosimy o ciszę"}` |

For button feedback (e.g. a lit MANUAL key), poll `GET /api/state` with the
same header (`caption_mode`, `captions_paused`); a dedicated Companion module
is only worth writing if that proves too limited.

**OSC** (vMix, QLab, TouchOSC): set `web.osc.enabled: true` (UDP
`127.0.0.1:9000` by default). Addresses: `/narrator/clear`,
`/narrator/mode [auto|manual|toggle]`, `/narrator/manual-text <text>`,
`/narrator/pause [1|0|T|F]`, `/narrator/recording [1|0]`; no argument
toggles. OSC has no authentication -- keep it on loopback. A lone `0`
argument on `clear`/`mode` is a button release and ignored; for pause use a
toggle button or send no argument.

## Session recording and privacy

Off by default. Switch it on from the web UI, the F9 shortcut,
`POST /api/recording` or OSC. Each session is written to
`var/recordings/<timestamp>/`: `audio.flac` (the exact 16 kHz mono signal the
speech recognizer received), `live.srt` (what went on air), and afterwards a
background `draft.srt` (offline large-v3 pass, low priority, waits for headroom
next to live ASR). The *Recordings* tab lists them (duration, size, status),
downloads SRTs, takes a corrected SRT back, deletes one or all recordings, and
copies a corrected one into `scripts/test_audio/` ("Use as reference") so
`make configure` learns from real venues. `recording.retention_days` purges old
sessions; recording stops below `recording.min_free_disk_mb`. A REC badge shows
in the UI, and `recording.on_air_dot` adds a small red dot to the NDI output.

**Consent / GDPR.** A recording is a recording of people speaking, i.e.
personal data (voices, and what they said). In the EU, tell speakers and the
audience before you record, have a lawful basis (usually consent), keep
recordings only as long as needed (use `retention_days`), and delete on
request. Recordings stay on this machine unless you download them.

## Configuration

All layout is percentage-based and font size is never set directly -- it's
derived from canvas height, margins, line count, and interline gap. See
`src/narrator/config/schema.py` for the full field list and validation
rules (e.g. `margin_top_pct + margin_bottom_pct` must be `< 100`), and
`src/narrator/config/defaults.yaml` for the shipped defaults.

Style overrides (`style.overrides` in the config) cover the requested
legibility effects: `background_band` (a solid box or rounded pill behind
each line -- set `shape: rectangle` and a bright color for a high-contrast
"yellow box" broadcast caption look), `shadow` (blurred drop shadow), and
`stroke` (text outline).

Transcript file: set `transcript.enabled: true` (or use the web UI's Transcript
tab) to save everything the ASR finalizes as an `.srt` (timed cues) or
`.txt` file, one per session, in `var/transcripts/` of the repo by
default (the platform user data dir outside a checkout).

Text cleanup: whisper's `[bracketed]` sound annotations ("[oklaski]") are
dropped by default (`asr.strip_bracketed_annotations`). `asr.text_filters`
takes find-and-replace rules (case-insensitive regular expressions) applied
before anything is captioned. Two optional quality gates,
`asr.min_mean_token_p` and `asr.max_timestamp_ratio`, discard low-confidence
or hallucinated decodes. They are off by default; `narrator-tune configure`
(`make configure`) measures them.

## Verifying without a microphone/NDI receiver

```bash
python scripts/render_preview.py
```

Renders a few layout/style scenarios to `scripts/output/*.png` (composited
over a checkerboard so transparency is visible) -- useful for checking
margins, wrapping, scrolling, and style overrides without any audio/GPU/NDI
dependency.

## Building a standalone binary

```bash
./packaging/pyinstaller/build_linux.sh      # Ubuntu
.\packaging\pyinstaller\build_windows.ps1   # Windows
```

See `packaging/whisper_cpp/README.md` first if you want GPU acceleration in
the packaged binary.

## Development status / what's been verified where

This was built and unit-tested in a sandboxed Linux container without a
microphone, GPU, display, or NDI receiver. What that means concretely:

- **Fully verified in that sandbox**: config validation, the font-size
  derivation and word-wrap algorithms, the rolling caption buffer, and the
  streaming local-agreement decode state machine (34 unit tests, all
  passing -- `pytest tests/`); the Pillow renderer end-to-end with the
  bundled Polish-capable font and all style overrides (see
  `scripts/output/*.png`); the Silero VAD ONNX wrapper against real
  silence; and the NDI sender end-to-end (renderer -> `cyndilib` ->
  `write_video`, no errors).
- **Not runnable in that sandbox, needs verification on your target
  machine(s)**: live microphone capture (`sounddevice`/PortAudio wasn't
  available there). The web UI (which replaced the Tkinter GUI) is
  exercised there with curl and headless Chromium against the real app.

  Note: an earlier revision of this list claimed prebuilt whisper.cpp
  requires AVX2. It does not -- the shipped libraries contain no AVX/AVX2/FMA
  instructions at all and run fine on a pre-AVX Pentium G4560. Real
  whisper.cpp inference and Vulkan GPU acceleration have since been verified
  end-to-end on that machine against `test.mp3`.
- The ASR backend's Python API usage (`pywhispercpp.model.Model`,
  `context_params={"use_gpu": ...}`, model auto-download) and the NDI
  bindings (`cyndilib.Sender`/`VideoSendFrame`, `FourCC.RGBA`) were both
  confirmed against the actual installed packages, not guessed.
