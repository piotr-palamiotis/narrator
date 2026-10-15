# TODO

## Status (audited 2026-10-15)

| Task | State |
|---|---|
| A. Web UI + system service + self-contained `make setup` + performance presets + autoconfigure split out as `make configure` | **Not started** (added 2026-10-15) |
| C. Features chosen from the 2026-10-15 review of similar software | **Not started** (added 2026-10-15) |
| B. obs-localvocal ASR architecture | **Mostly open.** Done: decode quality gates, bracket stripping, operator text filters, `flash_attn` (benched: no speedup on RADV), transcript file, and their GUI/autoconfigure wiring. Not started: Phase 0 baseline, the segmenter (§2–3), partials/pacer, the schema switch-over, sim/autoconfigure re-targeting, the whisper.cpp pin bump, cut-over. `streaming_decoder.py` (local agreement) is still the live decoder. |

---

# A. Web UI instead of Tkinter, narrator as a system service

Goal: narrator runs headless as a service that starts with the machine.
The operator controls it from a browser (admin UI on a local web server)
instead of the Tkinter window. Everything lives in the repo folder.
A public caption *viewer* for phones is **out of scope for now**. Design the
web layer so it can be added later as a separate, read-only listener (see
"Later" below).

## A1. Web admin UI (replaces `narrator.gui`)
- [ ] New `narrator/web/` package. Choose a small async framework (e.g.
  Starlette or aiohttp) running on its own thread/event loop inside the app
  process. It must never block the ASR, render or NDI threads: it reads
  controller state and sends caption events through a bounded queue that
  drops the oldest entries.
- [ ] **Bind to `127.0.0.1` only** by default (`web.bind`, `web.port`). Admin
  is never exposed on the LAN. Remote access goes through an SSH tunnel or
  Tailscale, not a LAN bind.
- [ ] Localhost is not automatically safe (CSRF and DNS rebinding from any
  browser tab on the machine):
  - reject requests whose `Host` is not `localhost`/`127.0.0.1:<port>`;
  - check `Origin` on every state-changing request;
  - require a random startup token: printed to the log and stored in a
    0600 file under the repo's `var/`, opened via `narrator --open-ui`;
    exchanged for an HttpOnly, SameSite=Strict cookie.
- [ ] **Config stays the single source of truth**: Save writes the YAML and goes
  through the same watchdog → `model_validate` → atomic swap path as hand
  edits. Validation errors from the Pydantic schema are shown in the form.
  No second validation layer in JS.
- [ ] Feature parity with the Tkinter GUI (see CLAUDE.md "GUI"):
  - all config tabs (Audio/ASR, canvas/layout/style, NDI, Transcript), including
    the language-whitelist editor, the text-filters editor, and the model
    dropdown (`AVAILABLE_MODELS` + cached models, still free-text editable);
  - live preview: render frames via `CaptionRenderer` → PNG/WebP over
    WebSocket or MJPEG, at a capped fps;
  - mic level meter (`MicCapture.level`, re-read from `controller._mic` every
    tick) and ASR Active/Inactive from `_asr_process.is_alive()`;
  - manual caption mode + text entry, Clear (Ctrl+N as a page-level shortcut);
  - status/log view (recent log lines, resolved ASR backend, worker restarts).
  Caption text in the page is rendered with `textContent`, never `innerHTML`.
- [ ] Static frontend bundled in the package (plain HTML/JS/CSS, no node build
  step), so the PyInstaller and offline installs keep working.
- [ ] Remove `gui/` (Tkinter, sv-ttk, pystray, `tray.py`) once parity is
  reached. Drop `sv-ttk`/`pystray` from `pyproject.toml` and the PyInstaller
  spec. `gui.enabled`/`--no-gui` become `web.enabled` with a
  `model_validator(mode="before")` migration for the old key.
  `render_preview.py` stays.
- [ ] Tests: API endpoints against a fake controller (no GPU/mic/NDI), Host/Origin/
  token rejection, config save → validation error round-trip.

## A2. Run as a service
- [ ] **systemd *user* service** (`~/.config/systemd/user/narrator.service`)
  plus `loginctl enable-linger <user>` so it starts at boot without a login.
  Not a system-wide service, because:
  - PipeWire/PulseAudio (the mic) runs per user session, and a root/system
    service cannot see it without hacks;
  - the Vulkan render node only needs the user in the `render`/`video` groups.
  Verify this on the reference box (headless boot → mic + GPU available).
- [ ] Unit runs `<repo>/.venv/bin/narrator --headless --config <repo>/config/config.yaml`.
  Settings: `Restart=on-failure` with a backoff, `WorkingDirectory=<repo>`,
  logs to journald (`journalctl --user -u narrator`). Optional rotating file
  log in `<repo>/var/log/`.
- [ ] Graceful shutdown on SIGTERM: the same path as GUI close (stop NDI, ASR
  worker, audio; close the transcript file).
- [ ] `make install` = venv + ASR backend (as today) + generate the unit from a
  template with the repo's absolute path + `systemctl --user daemon-reload &&
  enable --now`. Add `make uninstall` (disable + remove the unit only, leave
  the repo) and `make status` / `make logs`.
- [ ] Windows: out of scope for v1. Note it as a possible follow-up (NSSM or a
  Windows service wrapper).

## A3. Self-contained repo layout
Everything the app writes lives under the repo folder (gitignored):
- [ ] `config/config.yaml`: the service passes `--config` explicitly. Keep
  `platformdirs` only as the fallback for a bare `narrator` run. First start
  copies `defaults.yaml` there.
- [ ] `models/`: the whisper ggml models. pywhispercpp's `MODELS_DIR` points
  at the user cache today; redirect it (env var or explicit model path).
  Fix the GUI dropdown and the autoconfigure cache lookups, which read
  `MODELS_DIR` directly.
- [ ] `var/` (logs, token, transcripts default dir), `third_party/` (whisper.cpp
  source checkout and build tree for `build_vulkan_linux.sh`).
- [ ] What *cannot* live in the repo (document this in README): apt packages
  (Vulkan loader/driver, build toolchain, PortAudio, GStreamer) and the NDI
  runtime. NDI redistribution terms must be checked before downloading it
  automatically (see CLAUDE.md Notes).

## A4. `make setup` (one-shot machine bootstrap)
- [ ] `make setup` installs system prerequisites (detect apt; `sudo` only for
  this step, printed before running):
  `cmake libvulkan-dev glslc spirv-headers spirv-tools libvulkan1
  mesa-vulkan-drivers libportaudio2 gstreamer1.0-plugins-{base,good}`.
  It then runs the venv and the Vulkan pywhispercpp build, **picks a
  performance preset with a quick benchmark (A5)**, downloads that preset's
  model into `models/`, then runs `make install`.
  Idempotent: re-running skips what is present.
- [ ] Add the user to the `render`/`video` groups if missing (needs re-login;
  say so).
- [ ] NDI runtime: detect it; if missing, print the download link (or
  fetch it automatically if the license allows).
- [ ] **OBS + DistroAV** (step `make setup-obs`; `make setup` runs it unless
  `OBS=skip`). The minimum requirement: OBS can receive narrator's NDI
  stream, i.e. the DistroAV plugin plus whatever NDI runtime that route
  needs. Rule: **always use the install route the OBS/DistroAV developers
  document for the packaging found**; never mix routes (e.g. a .deb plugin
  into a Flatpak OBS). Sources, checked 2026-10-15:
  DistroAV wiki "1. Installation"
  (https://github.com/DistroAV/DistroAV/wiki/1.-Installation) and the OBS
  KB (https://obsproject.com/kb/linux-installation). Re-check them when
  implementing; follow the docs if they changed.
  - **Detect** every OBS install (there can be several) and report each one:
    - Flatpak: `flatpak info com.obsproject.Studio`, both `--system` and `--user`;
    - deb: `dpkg -s obs-studio`; use `apt-cache policy` to tell the
      `ppa:obsproject/obs-studio` build from the distro build;
    - Snap: `snap list obs-studio`;
    - other (source build in `/usr/local`, AppImage, ...): `command -v obs`
      plus `obs --version`.
    Record the OBS version and whether DistroAV is already present. Flatpak:
    `com.obsproject.Studio.Plugin.DistroAV`. deb:
    `/usr/lib/x86_64-linux-gnu/obs-plugins/distroav.so`, or
    `/usr/local/lib/obs-plugins/`. Also check for a per-user plugin in
    `~/.config/obs-studio/plugins/`.
  - **Flatpak OBS** →
    `flatpak install com.obsproject.Studio.Plugin.DistroAV`, in the same
    installation (system/user) as OBS. The NDI runtime is bundled, so no
    libndi is needed for OBS. Apply the documented OBS 32+ Avahi fix:
    `sudo flatpak override com.obsproject.Studio --system-talk-name=org.freedesktop.Avahi`.
    Remove the legacy `com.obsproject.Studio.Plugin.NDI` if it's present
    (ask first).
  - **deb OBS (PPA or distro)** → download the latest DistroAV `.deb` for
    this arch from the GitHub releases API (check the release's minimum
    OBS version against the installed one, and verify a checksum if the
    release publishes one). Install with `sudo dpkg -i …` then
    `sudo apt-get -f install`. NDI runtime v6+ via DistroAV's documented
    `libndi-get.sh install` (it downloads from ndi.video).
  - **Source build / `/usr/local` OBS** → same .deb, plus the wiki's documented
    symlinks into `/usr/local/lib/obs-plugins` and
    `/usr/local/share/obs/obs-plugins`. Non-Debian distro without Flatpak:
    no documented route → report it and stop, don't improvise.
  - **Snap OBS** → DistroAV documents no Snap route: report it, recommend
    the Flatpak, and change nothing.
  - **No OBS at all** → install per the OBS KB. Flathub `com.obsproject.Studio`
    is "recommended (all distros)", and DistroAV's Flatpak bundles NDI, so
    that's the default. `OBS_ROUTE=ppa` on Ubuntu uses the KB's
    `sudo add-apt-repository ppa:obsproject/obs-studio && sudo apt install obs-studio`.
    Other distros: the OBS GitHub wiki install instructions. Ensure Flatpak +
    the Flathub remote exist first (the distro's documented way).
  - **NDI networking prerequisites** from the DistroAV wiki:
    `avahi-daemon` installed, enabled and running (mDNS discovery). If UFW is
    active, show the documented mDNS/NDI port rules and ask before adding
    them. FFmpeg ≥ 7 only matters for NDI HX; narrator sends full NDI, so
    just warn.
  - Narrator's own sender (cyndilib) needs an NDI runtime too, outside any
    Flatpak. Check whether the cyndilib wheel bundles libndi; if not, the
    same `libndi-get.sh` route covers it. Handle this in the NDI step of
    `make setup`, not here.
  - Safety: print the plan (detected installs → actions) before doing
    anything; `sudo` only for the steps that need it. Refuse to modify OBS
    while it's running (ask the operator to close it). Idempotent: an
    already-installed, current plugin is a no-op. A `--check` mode only
    reports. Never uninstall or replace an existing OBS.
  - Verify after install: the plugin shows in the packaging's plugin
    listing / files. Then, with narrator running, discover narrator's NDI
    source over mDNS (e.g. via cyndilib's finder) to prove the network path
    works. Opening OBS itself needs a display, so it can't be checked headless.
  - Later (not now): add narrator's NDI source to an OBS scene through
    obs-websocket.
- [ ] Smoke check at the end: `narrator --list-devices`, Vulkan device
  enumeration, `pytest tests/`, service status.

## A5. Performance presets (what `make setup` picks)
Most settings are about accuracy and don't depend on the hardware. Tune them
once on the reference library and ship them as defaults. A preset covers only
the compute settings: `model`, `step_ms`, `agreement_passes` (or, after task B,
`partial_latency_ms` / `max_segment_ms`).
- [ ] `src/narrator/config/presets.yaml`, shipped with the package. Tiers, for
  example: `quality` (large-v3), `balanced` (large-v3-turbo/medium), `light`
  (small), `cpu` (base/small, larger step). Each row is produced by
  `make tune-presets` (A6), never hand-guessed. Until a tier has been searched,
  mark it `measured: false` in the file.
- [ ] Schema: `asr.preset: <name> | custom`. The preset supplies the values of its
  fields; an explicit value in config.yaml always wins (hand edits keep working).
  Resolved in `model_validator`, so the rest of the app sees plain values.
  `asr.device` and the language whitelist are never part of a preset.
- [ ] **Hardware check = a short real benchmark, not a GPU-name lookup**
  (`narrator --pick-preset`, run by `make setup`):
  1. about 1 min of decodes per candidate model on a bundled deterministic
     Polish clip (one of the Piper fixtures), reusing `bench.py` and
     `DecodeCostModel`;
  2. predict worker load and caption delay for each preset's timing;
  3. pick the most accurate preset that fits: load ≤ 0.7 (headroom below
     autoconfigure's 0.9) and delay p50 ≤ 3 s.
  It respects the configured device (no GPU→CPU downgrade) and writes only
  `asr.preset` (previous file → `config.yaml.bak`).
  `bench.py` and `stream_sim.DecodeCostModel` therefore **stay in narrator**:
  the runtime needs them; the tuning tool imports them.
- [ ] Web UI: preset dropdown, "Re-run benchmark" button, "Custom" shown as
  soon as someone edits a covered field.
- [ ] Tests: preset resolution / override precedence, the picker's choice from
  a fake cost model.

## A6. Autoconfigure becomes a separate tool (`make configure`)
Autoconfigure leaves the narrator app: it is not part of the runtime, the
service, the web UI or the PyInstaller build. It becomes a sibling package in
the same repo that **imports** narrator's real decode path, so what it
measures is still exactly what runs live.
- [ ] New package `src/narrator_tune/`, with its own console script
  `narrator-tune` and an optional dependency group `[tune]` in
  pyproject.toml (installed by `make configure`, not by `make install`).
- [ ] Move into it: `autoconfigure.py`, `library_eval.py`, `quality_probe.py`,
  `real_fixtures.py`, `report_library.py`, `ollama_analyzer.py`.
  Stay in `narrator.asr` and are imported: `backend`, `streaming_decoder`
  (or the segmenter), `vad`, `stream_sim`, `bench`, `metrics`.
  `scripts/stream_eval.py`, `eval_transcript.py` and `stress_full_pipeline.py`
  switch their imports.
- [ ] The cache code hash must keep covering the *narrator* modules it reuses
  (streaming_decoder/backend/vad/stream_sim/ring_buffer), so a change in the
  app still invalidates cached results.
- [ ] Config: the `autoconfigure:` section moves out of `RootConfig` into the
  tool's own `config/tune.yaml` (plus its own Pydantic model). narrator's schema
  gets a `model_validator(mode="before")` that drops a leftover
  `autoconfigure:` key with a log line, so old configs still load. Remove the
  GUI/web Autoconfigure tab.
- [ ] CLI: remove `narrator --autoconfigure` / `--analyze-reports`. Replace with:
  - `make configure` → `narrator-tune configure`: the current behavior. Search
    this machine against `scripts/test_audio/` and write the winning settings
    into the service's `config/config.yaml` (`asr.preset: custom`). Keep the
    existing flags (`--quick`, `--dry-run`); reruns resume from the cache.
    Restart the service at the end if it's running (or rely on hot reload).
  - `make tune-presets` → `narrator-tune presets`: the maintainer mode. Run the
    search once per preset tier, with that tier's load limit, and write
    `presets.yaml` (A5).
  - `narrator-tune analyze` → the Ollama report analysis.
- [ ] Cache and reports move to the self-contained layout:
  `var/tune/cache/`, `var/tune/reports/` (migrate or leave
  `scripts/autoconfigure_*` in place for one release).
- [ ] Tests: `tests/test_autoconfigure.py` and related → `tests/tune/`. Keep
  them GPU-free. Add a test that narrator imports and runs without
  `narrator_tune` installed.
- [ ] PyInstaller spec: exclude `narrator_tune`.

## A7. Docs
- [ ] CLAUDE.md: commands (`make setup`, `make configure`, `make tune-presets`, service targets), the autoconfigure section → `narrator_tune`, presets, the GUI section → web
  UI, the process/thread model (web thread), the sandbox limitations (the web UI
  *can* be tested here with curl/a headless browser; the mic still can't).
  README: install and operate as a service.

## Later (not now)
- Public read-only caption viewer for phones: separate listener/route set
  with no admin routes, bound to the audience interface. One-way SSE, a
  connection cap, slow clients dropped, `textContent` rendering, QR code.
  Venue networks: watch for Wi-Fi client isolation.

---

# C. Features chosen from the review of similar software (2026-10-15)

Turned down in the same review: live health dashboard, phone/browser
caption output (the viewer under A "Later" stays parked), correction buffer
before air, display styles (pop-on/fade/second aspect ratio), noise
suppression/AGC.

Suggested order: C1 with A1 → C2 → C3 → C4 → C5 → C6 → C7.

## C1. Keyboard shortcuts in the web UI + remote control (build with A1)
- [ ] **Browser keybinds** for every operator action: clear captions, auto/manual
  switch, focus the manual entry, submit, pause/resume captions, and
  recording on/off (C3).
  - Bound at document level, so they work while a text field has focus
    (the same rule as today's root-level Ctrl+N).
  - Configurable in the UI and stored in config (`web.shortcuts`).
  - **Chrome reserves Ctrl+N / Ctrl+T / Ctrl+W, and a page can't intercept
    them**, so the default clear key must be something else (e.g. Esc, or
    Ctrl+Shift+Space). Reject reserved combos when the operator binds keys.
  - A visible cheat-sheet (`?` overlay).
- [ ] **Control API** that the web UI itself uses: `POST /api/clear`, `/api/mode`,
  `/api/manual-text`, `/api/pause`, `/api/recording`, `GET /api/state`.
  Same localhost bind and token as A1 (a per-integration API token in
  config, revocable).
- [ ] **Stream Deck via Bitfocus Companion**: Companion's generic HTTP module
  can call the API directly, so document a ready button set first. Only
  write a dedicated Companion module (button feedback such as a "manual
  mode" light) if the generic one is too limited.
- [ ] OSC input (UDP, localhost by default) as a thin adapter onto the same
  API calls, for vMix/QLab/TouchOSC setups. Optional, off by default.
- [ ] Tests: API auth (token required, Host/Origin checks), each action against
  a fake controller, rejection of reserved keybinds.

## C2. Input channel selection + NDI audio as a source
- [ ] `audio.channels`: the list of 1-based input channels to mix into the mono
  signal (default `[1]` = today's behavior; `[1, 2]` = stereo downmix).
  Open the stream with `max(channels)` channels, average the chosen ones
  in the PortAudio callback, and compute the level meter from the mix.
  Validate in the schema: channels must exist on the device, checked at
  start, with a clear error.
- [ ] **NDI receive as an audio source**: the device dropdown lists microphones
  **and** the NDI sources discovered on the network (cyndilib finder), e.g.
  `ndi:MIXER (Program)`.
  - New `NdiAudioCapture` behind the same interface as `MicCapture`
    (`level`, frame queue, `stop()`), so the feed thread, the level meter and
    `_restart_audio_capture` don't change.
  - The NDI frame comes in (usually 48 kHz float planar, N channels) →
    channel select (same `audio.channels`) → resample to 16 kHz.
  - Exclude narrator's own NDI output from the list.
  - The source disappearing or reconnecting doesn't crash: log, show the
    meter at 0, reconnect when it reappears.
- [ ] Hardware needed to verify: an NDI audio sender on the network (e.g. OBS
  + DistroAV output, or NDI Test Patterns). Not possible in the sandbox:
  say so.

## C3. Session recording for later correction and accuracy work
Off by default, and switchable live from the UI and a shortcut (C1).
- [ ] When on, each session writes under `var/recordings/<timestamp>/`:
  - `audio.flac`: the exact mono 16 kHz signal whisper received (after
    channel select), so the recording replays exactly what the decoder saw;
  - `live.srt`: what went on air (reuse `captions/transcript.py`);
  - `draft.srt`: generated after the session by an **offline** best-effort
    pass (large-v3, the `eval_transcript.py` style spans). It's better than
    the live captions, so it's the starting point for a human fix. Runs in
    the background at low priority, never while ASR is live on the same
    GPU unless there is headroom.
- [ ] Recordings list in the web UI: duration, size, status (live / draft
  ready / corrected). Download the SRT, upload the corrected SRT back.
- [ ] **Purge**: a "Delete all recordings" button with confirmation, plus
  per-recording delete. Optional `recording.retention_days` auto-purge.
  Show the disk usage; stop recording below `min_free_disk_mb` (the same
  guard as the autoconfigure cache).
- [ ] **Into the reference library**: "Use as reference" copies a recording with
  a *corrected* SRT into `scripts/test_audio/`, so `make configure` /
  `make tune-presets` (A6) learn from real venues.
  - New SRT role: human-corrected ASR captions are the best reference there
    is. Update `real_fixtures`' tag → role mapping (e.g.
    `(Polish_corrected)` = objective).
  - The title hash puts it in dev or holdout as usual.
- [ ] **Privacy**: this records people speaking. Show a visible recording
  indicator in the UI (and optionally a small on-air dot in the NDI output,
  off by default). Add a README note about consent/GDPR for EU venues.
  Recordings never leave the machine unless the operator downloads them.
- [ ] **Training (research, later)**: once there are enough corrected hours
  (roughly tens of hours before fine-tuning beats prompting), fine-tune
  whisper (LoRA via HF transformers, on the NVIDIA card), convert to ggml
  and use it as a custom model path. Judge it only on the holdout split
  of the library, with recordings from venues it wasn't trained on, to
  catch overfitting to one room/speaker. Until then, the recordings help
  through `make configure` alone.

## C4. Newer whisper.cpp (moved up from B Phase 7)
- [ ] Bump the pywhispercpp/whisper.cpp source pin. **Hard requirement: the
  Vulkan backend still builds and works on the RX 6700 XT**, plus the
  `$ORIGIN` RPATH fix, the version pin and SSE4.2-only CPU compatibility in
  `build_vulkan_linux.sh`. Otherwise stay on v1.5.1.
- [ ] Re-verify on the new version whether these still hold:
  - `detect_language=True` (broken in 1.5.1);
  - beam search on GPU (the 'patience' error);
  - token API names;
  - `flash_attn` speed on RADV.
  Update CLAUDE.md for each.
- [ ] New capabilities to adopt:
  - DTW token timestamps → real word times in the transcript and
    recording SRTs (replaces the proportional estimate);
  - built-in ggml Silero VAD (optional; keep the 10 s quiet-reset behavior);
  - cross-attention access for C5.
- [ ] Re-run the bench and the autoconfigure model screen; the cost models
  are invalidated automatically by the code hash.

## C5. Decoder shoot-out: local agreement vs segmenter vs SimulStreaming
- [ ] Add **SimulStreaming / AlignAtt** (successor of whisper_streaming by the
  same authors): emit tokens when whisper's cross-attention shows the
  decoder is not yet looking at the newest audio, instead of waiting for N
  agreeing hypotheses. Expected: lower latency than local agreement at
  similar WER. Needs cross-attention weights from whisper.cpp (C4) → check
  that the binding exposes them, or add it to our source build.
- [ ] One `asr.decoder` setting: `local_agreement` (today) | `segmenter` (task B)
  | `simulstreaming`. All three run through `stream_sim` / autoconfigure on
  the same excerpts; winner = best pooled WER with delay p50 ≤ 3 s, judged
  by paired comparison + holdout. The losers stay selectable until one has
  clearly won across a full run; then delete them (keep the guard tests).
- [ ] Every decoder keeps the shared guards: repetition collapse, boundary
  dedupe, no-loss rollover, whitelist lock, drain-then-tick ingest.

## C6. NVIDIA backend + Canary/Parakeet models (an NVIDIA card is planned)
- [ ] Backend abstraction: `WhisperBackend` becomes one implementation of an
  `AsrBackend` interface (transcribe window, detect language, token
  confidences, close). Decoders and `stream_sim` talk to the interface only.
- [ ] whisper.cpp on NVIDIA: a CUDA build of the same pin (or Vulkan on NVIDIA;
  bench both). `asr.device: cuda`, `gpu_device` selection across vendors.
  **The configured device stays final** (no silent fallback across backends
  or vendors either).
- [ ] NeMo models as a second backend (separate process, like the whisper
  worker):
  - **Canary-1B-v2** (25 EU languages incl. Polish; also does X↔English
    translation);
  - **Parakeet-TDT-0.6B-v3** (25 languages, much faster).
  Runtime: NeMo (PyTorch/CUDA) or ONNX via sherpa-onnx/onnxruntime-gpu.
  Pick by the bench. Check licenses (believed CC-BY-4.0; verify) and
  whether the language whitelist maps onto their language handling.
- [ ] They plug into the decoder shoot-out (C5) and autoconfigure like any
  model; presets (A5) gain NVIDIA tiers. Expected wins to verify: far less
  hallucination over music/applause, native streaming.
- [ ] `make setup`: detect the NVIDIA driver/CUDA, and install the matching
  runtime the way NVIDIA/NeMo document it (same rule as the OBS step).

## C7. Suggestion list (event vocabulary) + profanity list
Both are edited in the web UI with the same list-editor pattern as the
language whitelist / text filters, and hot-reload without restarting ASR
where possible.
- [ ] **Suggestion list** `asr.vocabulary`: names, places and jargon for an
  event (e.g. speaker names, the venue, sponsor names).
  - Passed as a hint at the front of whisper's `initial_prompt` on every
    call, ahead of any `prompt_carryover_chars` text. Keep it within
    whisper's prompt token budget (~224 tokens, shared with the carry-over),
    truncating the carry-over first.
  - Optional fuzzy snapping after decode: a caption word within a small
    edit distance of a list entry (normalized, diacritics-aware, ≥5 chars
    only) is replaced by the entry. It runs before agreement, like
    `text_filters`, so it can't block commits. Off by default; risk:
    over-snapping ordinary words.
  - Measure both with autoconfigure on excerpts whose SRT contains the names
    (prompt-only vs prompt+snap vs off). As with `text_filters`, autoconfigure
    never writes the list itself (operator data) but keys cached results on it.
  - For C6 backends without a prompt (NeMo): only the snapping applies;
    check whether they offer context biasing / word boosting natively.
  - Per-event lists: load/save named lists (e.g. `config/vocabulary/<event>.txt`),
    so the operator switches by event instead of retyping.
- [ ] **Profanity list** `captions.blocklist`: words that must never reach the
  screen.
  - Whole-word, case- and diacritics-insensitive matching that covers Polish
    inflected forms by stem/prefix entries (e.g. `kurw*`), not just exact
    words.
  - Action per config: mask (`k***a`, the default), replace with a fixed
    string, or drop the word.
  - Applied at the same point as `text_filters` (before agreement), so live
    captions, `live.srt` and the transcript match what aired. **Recordings'
    `draft.srt` stay unfiltered**, so the corrected references and any
    training data stay true to the audio.
  - Ships empty plus an optional bundled starter list for Polish/English that
    the operator can enable. Applies in manual caption mode too (a typo-safe
    guard for operators).
  - Tests: inflection matching, mask/replace/drop, no false hits inside
    longer words, and the draft.srt exemption.

---

# B. Move narrator's ASR to the obs-localvocal architecture (AMD + Vulkan)

Reference: https://github.com/royshil/obs-localvocal, studied at `191c7ef`
(2026-09-09). Its pipeline lives in `src/whisper-utils/vad-processing.cpp`,
`whisper-processing.cpp`, `token-buffer-thread.cpp` and
`src/transcription-filter-callbacks.cpp`.

The target is still the RX 6700 XT through whisper.cpp's **Vulkan** backend
(the pywhispercpp `v1.5.1` source build with `GGML_VULKAN=1`).

---

## 1. What changes, in one picture

**Today (narrator):** a local-agreement streaming decoder.
- whisper re-decodes a growing window every `step_ms`.
- Each new hypothesis is diffed against the last `agreement_passes` hypotheses.
- A word prefix is committed (FINAL) only after it has been stable across all of them.
- Words trickle out mid-utterance. Latency ≈ `step_ms × agreement_passes` plus decode cost.

**Target (LocalVocal):** VAD-segmented, one-shot decoding.
- Silero VAD cuts the audio into speech segments.
- When a segment ends, the **whole segment is decoded once**. That result is the FINAL.
- Optional *partials*: while a segment is still open, every `partial_latency` ms
  (default 1100) the whole segment so far is re-decoded (peek, not pop). The
  result replaces the previous partial on screen. There is no agreement or diff.
  A partial is never committed: the FINAL decode of the closed segment replaces it.
- Each decode result is filtered for quality before it reaches the screen:
  - mean token probability below a threshold → dropped;
  - timestamp tokens beyond the audio length → treated as a hallucination and dropped;
  - special and `[bracketed]` tokens are stripped.
- Context carries across segments as the last N final sentences, passed as `initial_prompt`.
- Display can optionally go through a paced "token buffer": a typewriter-style
  feed into N lines × M chars, where an incoming partial pops the previous
  partial's tokens off the tail.

```
mic ─▶ feed ─▶ [ASR worker process]
                 drain all frames ─▶ resample/accumulate 16 kHz
                 ─▶ segmenter (mode: vad | hybrid | fixed)
                      ├─ segment open, partial due ─▶ decode(peek)  ─▶ PARTIAL
                      └─ segment closed / capped   ─▶ decode(pop)   ─▶ FINAL
                 decode = pad ─▶ whisper_full ─▶ token filter ─▶ text filters
             ─▶ event queue ─▶ CaptionBuffer (+ optional paced token feed)
             ─▶ renderer ─▶ NDI
```

## 2. What we keep from narrator, and why

LocalVocal is an OBS filter running in-process. Several narrator decisions are
better for this product, or were measured as necessary on this hardware. **They stay:**

| Keep | Reason |
|---|---|
| ASR in a separate `spawn`ed process, supervisor with backoff, quick-crash stop | LocalVocal runs whisper on a thread inside OBS, so a native crash takes OBS down. We isolate it. `spawn` is required for Vulkan re-init (fork SIGSEGVs). |
| **No silent GPU→CPU fallback** (configured device is final) | LocalVocal falls back to CPU on a bad GPU index. Narrator policy forbids that. |
| Drain-then-tick worker loop | LocalVocal does the same thing: `get_data_from_buf_and_resample` pops *everything* queued per loop. Same shape, keep it. |
| Margin-gated `language_whitelist` | LocalVocal only does forced language or plain auto. Our whitelist is the operator's single language control and is better. Re-wire it to per-segment detection (§4.4). |
| Caption retention: CLEAR prunes only off-screen history, WIPE is opt-in | LocalVocal blanks the caption `max_sub_duration` (3 s) after the last result. Narrator explicitly forbids blanking what the audience is reading. **Do not port that timer.** |
| `_collapse_repetitions` + `_drop_repeated_boundary` text cleanup | large-v3 repetition loops are a whisper property, not a decoder one. LocalVocal has no equivalent. It has only user regex replace, which we add as well. |
| `temperature_inc=0.0`, `audio_ctx` sized to the window, greedy sampling | LocalVocal defaults to beam search. Beam search is measured unusable in pywhispercpp 1.5.1 on GPU (the 'patience' error leads to CPU fallback and 104 s decodes). The temperature ladder caused 17 s stalls. |
| `initial_prompt` set on every call | pywhispercpp keeps params between calls. LocalVocal sets it on every call too. |
| Rendering, NDI, config hot-reload, GUI, autoconfigure *framework* | These are outside the ASR core. Only their ASR-facing parts change. |
| Silero state hygiene | LocalVocal resets Silero state whenever the previous chunk was non-speech (`process(..., reset=!vad_on)`). That covers our measured LSTM drift on long non-speech runs. Adopt that rule and keep the 10 s quiet-reset as a backstop until the new rule is proven on the 88-min recording. |

## 3. Things LocalVocal has that we deliberately skip

- **Translation** (CTranslate2 M2M100/NLLB, cloud providers). Out of scope; the
  product shows Polish captions. Revisit only on request.
- OBS-specific outputs: text source, stream CEA-608 captions, WebVTT-in-SEI.
  NDI is our output.
- Model downloader UI. We use pywhispercpp's model cache and paths.
- HIP/ROCm build. The target stays Vulkan.

---

## 4. Work plan

Each phase ends green: `pytest tests/` passes, and the stress harness /
`stream_eval.py` numbers are recorded before moving on. Changes stay
uncommitted for review (project rule).

### Phase 0: Baseline (before touching code)
- [ ] Record current numbers so the rewrite is judged on data, not taste:
  - `scripts/stress_full_pipeline.py --model large-v3` (WER, finalize latency)
  - `scripts/stream_eval.py` on 3 fixed excerpts (WER, delay p50/p90, busy)
  - the autoconfigure dev-set score of the current config (22.1% WER,
    delay p50 per the 2026-10-02 run)
- [ ] Save them to `scripts/autoconfigure_reports/baseline-localagreement.md`.

### Phase 1: Decode core (`asr/backend.py`)
Port `run_whisper_inference` behavior:
- [ ] `WhisperBackend.decode_segment(audio, *, prompt, language) -> SegmentResult`
  returning text, per-token `(id, text, p, t0, t1)`, detected language, and decode time.
  Use the `_pywhispercpp` token API. It is verified present in our build:
  `whisper_full_n_tokens`, `whisper_full_get_token_data/_p/_text`.
- [ ] Padding:
  - segments shorter than 1 s → pad to 1.01 s, centred, with ±0.01 white noise;
  - 10 ms of silence on each side of every segment;
  - skip segments shorter than 50 ms.
- [ ] Token filter: *(partial: confidence ignores ids ≥ `whisper_token_eot`; `[...]` stripped at text level by `strip_bracketed_annotations`; the trailing-`.` drop is not done)*
  - drop special tokens (`id >= eot`; use `whisper_token_eot`, **not** the
    hard-coded 50256, because large-v3's vocabulary differs);
  - drop `[...]`-bracketed tokens and the trailing `.` before EOT.
- [x] **Hallucination guard** *(done as `asr.max_timestamp_ratio`, off by default, searched by autoconfigure)*: if any timestamp token's time ÷ the real audio
  duration exceeds `duration_filter_threshold` (default 2.25), return SILENCE.
- [x] **Confidence gate** *(done as `asr.min_mean_token_p`, off by default: 0.4 rejects real Polish on small)*: if the mean `p` of the kept tokens is below
  `min_mean_token_p` (default 0.4), return SILENCE.
- [x] Keep `temperature_inc=0`, greedy, `audio_ctx` sizing, and per-call `initial_prompt`.
- [x] Expose `flash_attn` (present in `whisper_context_default_params()` of our
  build). Bench it on RADV before defaulting it on.
- [x] Unit tests with a fake token stream for each filter rule.

### Phase 2: Segmenter (new `asr/segmenter.py`, replaces `streaming_decoder.py`)
Port `vad_based_segmentation` / `hybrid_vad_segmentation` / `vad_disabled_segmentation`
as one class with a `mode`:
- [ ] **`vad` (default)**:
  - run Silero over all newly drained audio, in whole windows only, with at
    least 8 windows buffered before processing;
  - a speech start pulls in 100 ms of pre-roll (subsumes `vad_preroll_ms`);
  - a speech end inside the buffer closes the segment → FINAL decode (pop);
  - speech still running → keep accumulating, and emit a PARTIAL decode (peek)
    when ≥ `partial_latency_ms` of new audio has arrived since the last partial.
- [ ] **`hybrid`**: accumulate everything and close a segment every
  `segment_duration_ms` (7000). Partials are run only if Silero sees speech in
  the buffer; if it doesn't, drop all but the last 250 ms.
- [ ] **`fixed`** (LocalVocal's "disabled"): fixed-length segments, no VAD.
- [ ] **Segment cap (narrator addition; LocalVocal's Silero `max_speech` is ∞)**:
  - close at `max_segment_ms`, preferring the first ≥160 ms pause once the
    segment is 60% full (port today's rollover rule);
  - nothing may be trimmed or lost: the guarantee of
    `test_long_speech_loses_no_words_across_rollovers` must be re-expressed
    against the segmenter, and pass.
- [ ] Silero state reset on the transition into non-speech (LocalVocal rule),
  plus the existing quiet-reset backstop.
- [ ] The segment carries audio-time `t0`/`t1` for caption delay metrics and an optional SRT file.
- [ ] Post-decode text pipeline, applied to PARTIAL and FINAL alike: *(regex filters done as `asr.text_filters`, applied before agreement in today's decoder)*
  - `_collapse_repetitions`;
  - the configurable regex filter/replace list (LocalVocal's
    `filter_words_replace`, case-insensitive);
  - `_drop_repeated_boundary` against the previous FINAL only;
  - empty or `.` text → no event.
- [ ] Context: keep the last `context_sentences` FINAL texts (LocalVocal's
  `n_context_sentences`) and join them as the prompt. This replaces
  `prompt_carryover_chars`. Reset it on WIPE and on a language change.
- [ ] Silence timers (prune/wipe) stay driven by VAD silence exactly as today.

### Phase 3: Language whitelist on segments
- [ ] Detection runs once per segment, on the segment audio, once it is ≥ 1.5 s
  (via the dedicated `auto_detect_language()`; the `detect_language=True`
  param is still broken in 1.5.1).
- [ ] Partials before the lock are skipped (as today). The FINAL always has a language:
  - a decisive margin locks it;
  - an ambiguous result falls back to the previous segment's language, then to
    the top candidate.
- [ ] A capped segment rolling over keeps its lock; a real VAD pause re-arms it.
- [ ] Port the existing whitelist tests onto the segmenter.

### Phase 4: Events, caption buffer, display pacing
- [ ] PARTIAL semantics become "replace the whole in-progress segment text"
  (today it is "uncommitted tail"). Check that `CaptionBuffer` replaces rather
  than appends partials, and that a FINAL replaces the open partial atomically
  (no flash of both).
- [ ] `show_partial_captions` maps onto partial decoding: off = no peek decodes
  at all, which saves GPU time.
- [ ] Optional **paced output** (port `TokenBufferThread`), new `captions/pacer.py`:
  - segmentation by char, word, or sentence;
  - speed adapts to backlog (char: 100/66/33 ms, word: 200/150/100 ms);
  - an incoming partial pops the trailing partial tokens first.
  It sits between the event-drain thread and `CaptionBuffer`. Layout, wrap,
  `max_lines` and the derived font size stay ours (no `numPerSentence` char
  wrapping: wrap is pixel-based already). **No `maxTime` clear**: retention
  rules win.
- [x] Optional SRT/text transcript file of FINALs (LocalVocal "save to file"),
  using segment `t0`/`t1`.

### Phase 5: Config schema + GUI
- [ ] `asr` changes: *(partial: `min_mean_token_p`, `max_timestamp_ratio` (= `duration_filter_threshold`), `text_filters`, `flash_attn` exist, with GUI controls; the rest is open)*
  - remove: `step_ms`, `agreement_passes`, `vad_preroll_ms`, `prompt_carryover_chars`;
  - add: `segmentation_mode` (vad|hybrid|fixed), `partial_latency_ms` (1100),
    `segment_duration_ms` (7000), `max_segment_ms` (replaces
    `max_utterance_ms`/`max_context_seconds`), `min_mean_token_p` (0.4),
    `duration_filter_threshold` (2.25), `context_sentences` (1),
    `text_filters` (list of `{pattern, replace}`), `flash_attn` (false);
  - keep: `vad_threshold`, `end_silence_ms` (→ Silero min-silence), device,
    gpu_device, whitelist, and the silence timers.
- [ ] `model_validator(mode="before")` migration: old keys are accepted and
  mapped or dropped with a log line (same pattern as
  `clear_after_silence_seconds`).
- [ ] Cross-field validation (e.g. `partial_latency_ms < max_segment_ms`) goes
  in the schema's `model_validator`s.
- [ ] Add a `display` section for pacing, or put it under `layout`: `pacing`
  (off|char|word|sentence). Update `defaults.yaml`.
- [ ] GUI Audio/ASR tab: replace the step/agreement controls with mode, partial
  latency, confidence gate, and a filter editor (same list editor pattern as
  the whitelist).

### Phase 6: Simulation, eval and autoconfigure re-targeting
- [ ] `stream_sim.simulate_stream` drives the segmenter instead of
  `StreamingDecoder`. It is still a virtual clock, and each decode is charged
  its cost. The cost model must cover two decode kinds: peek (partial) and
  final, both a function of segment length.
- [ ] `scripts/stream_eval.py` and `stress_full_pipeline.py` are switched over.
  The code hash invalidates old cached results automatically.
- [ ] Autoconfigure `SEARCH_ORDER` gets the new levers: *(partial: the two quality gates are searched)* `partial_latency_ms`,
  `max_segment_ms`, `end_silence_ms`, `vad_threshold`, `min_mean_token_p`,
  `duration_filter_threshold`, `context_sentences`, and `segmentation_mode`.
  The rules stay unchanged: delay cap 3 s, paired comparison, holdout gate.
  FINAL delay is now segment-end-bound, so partials need their own metric:
  report the first-visible-text delay separately from the FINAL delay.
- [ ] Re-run the autoconfigure model screen: medium/small may now beat
  large-v3 on delay at equal WER.

### Phase 7: Vulkan / build
- [ ] No backend change is needed: the segmenter calls the same pywhispercpp
  Vulkan context. Verify per-segment decode costs on the RX 6700 XT:
  - 1 s, 3 s, 7 s, 11 s, 20 s segments;
  - flash_attn on/off;
  - large-v3 vs medium.
- [ ] Optional: evaluate bumping the whisper.cpp source pin. Newer versions
  have built-in ggml Silero VAD (`whisper_vad_*`, already exported by our
  bindings), which would let VAD run on the GPU and remove onnxruntime. Do this
  only if it keeps the `$ORIGIN` RPATH and version-pin fixes in
  `build_vulkan_linux.sh`, and SSE4.2-only CPU compatibility.
- [ ] PyInstaller spec: drop modules that are no longer used and add new ones.

### Phase 8: Cut-over and cleanup
- [ ] Phase 0 metrics vs new metrics, side by side. The new architecture
  becomes the default only if:
  - pooled WER is within noise or better, **and**
  - the FINAL delay p50 is ≤ 3 s (user goal).
  If it loses, keep both behind `segmentation_mode`, and keep the old decoder
  as `localagreement` until it wins.
- [ ] Delete `streaming_decoder.py` and its tests once the segmenter has
  absorbed every guard test:
  - repetition collapse;
  - boundary dedupe;
  - no-loss rollover;
  - whitelist lock;
  - the drain-then-tick ingest rate (43% audio drop regression).
- [ ] Update CLAUDE.md:
  - architecture (the Streaming ASR decode section is rewritten);
  - config fields;
  - autoconfigure stages and the measured numbers.
  Update README.md the same way.

---

## 5. Risks / open questions
- **Partial cost on large-v3**: re-decoding the whole open segment every 1.1 s
  is ~0.13 s + 0.031 s/word (calibrated). A 20 s segment means roughly 1.5–2 s
  per peek, which can starve the FINAL decode. Mitigations:
  - make the partial interval grow with segment length;
  - skip a partial if the previous one is still pending;
  - lower `max_segment_ms`.
- **Delay**: a FINAL appears only after `end_silence_ms` + one full decode.
  With no partials, the delay on continuous narration is up to `max_segment_ms`.
  So partials (or a short cap) are effectively required for the 3 s goal.
- **Large-v3 language drift on short segments** (Cyrillic captions for
  Polish): segment-level detection with ≥1.5 s and the whitelist should hold,
  but verify this on the reference library.
- **Mean-p gate on Polish**: the 0.4 threshold was tuned on English tiny
  models. Let autoconfigure tune it, and watch the deletion count.
