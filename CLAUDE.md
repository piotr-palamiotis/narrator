# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Live speech captioning, streamed as an NDI RGBA video source (transparent
background) so any NDI-capable app (OBS, vMix, NDI Studio Monitor, ...) can
receive and display it as a lower-third overlay. Speech-to-text runs locally
via whisper.cpp (through `pywhispercpp`), on CPU, NVIDIA GPU, or AMD GPU (via
Vulkan) with automatic fallback. Default configuration targets Polish
transcription with the `large-v3` model.

## Commands

```bash
# setup
make setup                  # one-shot bootstrap (packaging/setup/setup.sh): apt pkgs, groups, NDI
                            # check, venv + Vulkan build, --pick-preset, OBS + DistroAV, install, smoke
                            # check. Flags: OBS=skip OBS_ROUTE=ppa SERVICE=skip PRESET=skip TESTS=skip
                            # YES=1 DRY_RUN=1 (print commands only). Not for the sandbox beyond DRY_RUN=1.
make setup-obs ARGS=--check # OBS + DistroAV (packaging/setup/obs.sh); --check = report only
make install                # venv + narrator + Vulkan pywhispercpp source build
                            # + systemd user service (unit rendered, enabled --now, linger)
make install SERVICE=skip   # same without any systemd change (use this in the sandbox)
make unit UNIT_DIR=/scratch # render the unit only; check: systemd-analyze --user verify
make uninstall | status | logs   # remove unit only / systemctl status / journalctl -f
# or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e .
make asr-backend            # only if ASR wasn't built with a GPU backend
make asr-backend ASR_GPU=cuda   # CUDA instead of Vulkan (needs nvcc); default auto = cuda if nvcc+nvidia-smi work
                                # (this box: .venv has the CUDA build since 2026-11-16, auto = CUDA;
                                # back to Vulkan: ASR_GPU=vulkan FORCE=1 + asr.unit vulkan:0)

# run
narrator --list-devices     # sanity-check microphone enumeration
narrator                    # default config; web UI on port 8765 (all addresses), opens a browser if there is a display
narrator --local-only       # web UI for this computer only (127.0.0.1); a startup flag, never a config/UI setting
narrator --headless         # never opens a browser (the service); the web UI still runs (web.enabled)
narrator --no-web           # no web UI / OSC at all (--no-gui is the old alias)
narrator --set-password     # web UI login password (scrypt hash in web-password.json beside config.yaml)
narrator --open-ui          # open the RUNNING instance's web UI (URL from var/web-ui.json)
narrator --config config/config.yaml   # created from defaults.yaml if missing
narrator --headless --log-file         # also rotating var/log/narrator.log (+ narrator-asr.log)

# tests
pytest tests/               # all tests (no GPU/mic/NDI needed)
pytest tests/test_wrap.py -k some_test_name   # single test

# render sanity check without mic/GPU/NDI hardware
python scripts/render_preview.py   # writes scripts/output/*.png

# ASR stress test (Piper TTS -> PipeWire virtual mic -> full app -> WER/latency)
python scripts/gen_test_clips.py                    # byte-deterministic Polish clips + manifest
python scripts/stress_full_pipeline.py --model medium --step-ms 350 --repeats 3 --passes 2

# find the model + decode settings whose live captions best match the reference
# recordings in scripts/test_audio/ (audio + SRT), and persist them to config.
# A separate tool (src/narrator_tune, `[tune]` extra), not part of the app.
make configure                             # = narrator-tune configure; ~12 h budget, cached, reruns resume
narrator-tune configure --quick --dry-run  # ~1 h smoke run, config.yaml untouched
narrator-tune analyze                      # run reports -> local Ollama
make tune-presets                          # = narrator-tune presets: rewrites presets.yaml, hours per tier

# pick asr.preset for this machine (~1 min of real decodes per model, configured asr.unit)
narrator --pick-preset --dry-run           # report only; --download fetches missing models,
make pick-preset                           # --all-presets measures every tier (service config)

# standalone binary
./packaging/pyinstaller/build_linux.sh
.\packaging\pyinstaller\build_windows.ps1
```

There is no configured linter/formatter in this repo — don't assume one.

### GPU (Vulkan)

The PyPI `pywhispercpp` wheel no longer ships the Vulkan backend (verified
2026-10-03: the 1.5.1 wheel contains only CPU ggml, and that CPU path
SIGILLs at `whisper_init` on pre-AVX CPUs). ASR therefore comes from a
source build: pywhispercpp's `v1.5.1` bindings (still its newest release)
with the whisper.cpp submodule moved to **`v1.9.5`** (pywhispercpp 1.5.1
vendors 1.8.5; the C API change is additions only), `GGML_VULKAN=1`,
installed as version `1.5.1+whispercpp.1.9.5`. `make install` runs
`packaging/whisper_cpp/build_vulkan_linux.sh` whenever that exact version
is not installed. The script also: patches whisper.cpp ≥ 1.9's forced
`CMAKE_LIBRARY_OUTPUT_DIRECTORY` (otherwise libwhisper/libggml never reach
the wheel); applies the mandatory `$ORIGIN` RPATH fix (source-built wheels
bake in a dead RUNPATH to pip's temp build dir); pins
`SETUPTOOLS_SCM_PRETEND_VERSION` (a shallow tag clone otherwise builds as
0.0.0); puts the venv's bin on PATH (ggml-vulkan's shader generator is a
`-G Ninja` sub-build); and exports `GGML_VULKAN=1` itself (the script used
to rely on the caller's environment for it). Way back to the previous
build: `WHISPER_CPP_TAG=v1.8.5 make asr-backend` (verified; a prebuilt
`third_party/wheels/pywhispercpp-1.5.1+whispercpp.1.8.5-*.whl` is kept too,
it needs the same patchelf loop after `pip install`). The runtime needs only
the Vulkan loader + driver ICD (`libvulkan1` + `mesa-vulkan-drivers` on
AMD/Intel, the NVIDIA driver's own ICD on NVIDIA) — **not** the
`vulkan-tools` diagnostic package, and not the Vulkan SDK.

**`GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` is set by `WhisperBackend` for GPU
loads** (unless already set; `NARRATOR_VK_GRAPH_OPTIMIZE=1` keeps ggml's
default). ggml-vulkan's graph optimizer runs on the CPU for every graph,
i.e. every decoded token, and on the Pentium it costs more than it saves:
large-v3 decode 12.6 → 8.5 ms/token on whisper.cpp 1.9.5 (1.8.5: 9.2 →
8.5), identical text. Without it 1.9.5 is ~35% slower than 1.8.5 on this
box; with it the two are equal. Measured with `narrator.asr.bench`
(RTX 3070, large-v3, 2/5/10/20 s windows): 1.8.5 as shipped before
0.19/0.46/0.90/1.73 s, detection 0.15 s → 1.9.5 now 0.17/0.41/0.80/1.52 s,
detection 0.13 s; `small` 0.07/0.15/0.31/0.59 → 0.06/0.13/0.28/0.53 s.
Historical (RX 6700 XT via RADV, 1.8.5): 2s→0.24s, 10s→1.05s, 20s→1.96s,
detection 0.30s. Model load: large-v3 ~2 s warm, ~6–7 s cold.

The source build compiles ggml's CPU backend natively (`-march=native`), so
on the Pentium G4560 test box it is SSE4.2-only (verified: no ymm/VEX
instructions in libggml-cpu/libggml-base/libwhisper) — fine on this
machine, but a built wheel is not portable to other pre-AVX hosts without a
rebuild. whisper.cpp 1.9 also builds `libparakeet.so` (NVIDIA Parakeet
support in whisper.cpp itself, relevant to C6); pywhispercpp 1.5.1 has no
bindings for it.

Cross-attention from Python (for a C5 AlignAtt decoder), checked on 1.9.5:
the alignment-head cross-attention is computed only for DTW
(`dtw_token_timestamps`), in a separate re-decode pass per segment after the
decode, kept in whisper's private state (`aheads_cross_QKs`) and reduced to
one time per token (`whisper_token_data.t_dtw`). There is no C API or
binding for the attention weights themselves, nor a ggml eval callback in
`whisper_context_params`. Exposed to Python: the DTW context params
(`dtw_token_timestamps`, `dtw_aheads_preset`, `dtw_n_top`, `dtw_mem_size`;
not the custom `dtw_aheads` head list), `t_dtw`, the
`whisper_alignment_heads_preset` enum, `whisper_model_n_text_head/
n_audio_head/n_text_layer`, raw `whisper_encode`/`whisper_decode`/
`whisper_get_logits`, and a per-step `logits_filter_callback`. A true
AlignAtt needs a whisper.cpp patch (a getter for the decoder's per-step
alignment-head attention) plus a binding; an approximation is possible
today from `t_dtw` (hold back tokens whose aligned time is near the window
end).

CUDA build, verified 2026-11-16 on the RTX 3070 (CUDA toolkit 12.0 at
`/usr/bin/nvcc`, Ubuntu's `nvidia-cuda-toolkit`): `ASR_GPU=cuda
CMAKE_CUDA_ARCHITECTURES=86` built pywhispercpp 1.5.1 + whisper.cpp 1.9.5
with `libggml-cuda.so` in ~25 min on the Pentium (built into a scratch
venv, `~/.cache/narrator-cuda-test/venv`). **Since 2026-11-16 the repo's
`.venv` runs that CUDA wheel** (installed from
`~/.cache/narrator-cuda-test/src/dist/`, RUNPATH `$ORIGIN`, libggml-vulkan
removed; the user config has `asr.unit: cuda:0`, previous file kept as
`config.yaml.before-cuda`). Vulkan rollback wheel:
`~/.cache/narrator-vulkan-rollback/` (or `third_party/pywhispercpp-src/dist/`
until the next build): `pip install --force-reinstall --no-deps` it, run the
same patchelf `$ORIGIN` loop, set `asr.unit: vulkan:0`. Run through the real `WhisperBackend` with `asr.unit:
cuda:0`: "ASR backend: CUDA (NVIDIA GeForce RTX 3070)", model in VRAM, no
CPU fallback. large-v3 vs the Vulkan build, same process shape (median of
4 after a warm-up, test.mp3): load 8.7 vs 8.0 s; detection 0.25 vs 0.19 s;
2/5/10/20 s windows 0.08/0.16/0.36/0.70 vs 0.11/0.19/0.39/0.80 s (CUDA
7–23% faster decodes); text word-for-word identical. tiny on CUDA:
0.02/0.04/0.07/0.41 s. The first detection of the very first CUDA run
after the build took 12 s (one-off; later processes 0.17 s). Live app path
verified 2026-11-16: switching `asr.unit` to cuda:0 on a running app
restarted the worker onto CUDA0 (model 3094 MB in VRAM), captions flowed
from the virtual mic. Known script quirk: in a
fresh venv without numpy the build's final clean-environment import check
fails (the wheel is installed `--no-deps`); the build itself is fine.

NVIDIA notes (RTX 3070, driver 615): the Intel HD 610
iGPU enumerates as Vulkan device 1 (`vulkan:1`, PCI 0000:00:02.0) and
`asr.unit: auto` picks the discrete card (`vulkan:0`, PCI 0000:01:00.0);
ggml-vulkan's own list has the same order (verified 2026-10-17). The driver prints a harmless `NVVM compilation
failed: 3` on some runs. The first flash-attention run after a build change
once sat at 100% CPU for 20+ minutes with the model not yet on the GPU
(1.8.5; killed — presumably the driver compiling pipelines); later runs
loaded in 2 s. A one-off `corrupted size vs. prev_size` glibc abort at
process exit was seen once on the very first 1.9.5 large-v3 run and never
again in ~20 later load/decode/close cycles, including the real worker
path.

`packaging/whisper_cpp/README.md` covers source builds in full.

## Sandbox limitations (relevant when asked to "verify" something)

This environment has no microphone, display, or NDI receiver. It *does* have
a working NVIDIA GPU (RTX 3070, 8 GB, driver 615, Vulkan device 0; the
Intel HD 610 iGPU is Vulkan device 1 — never select it by accident). The
AMD RX 6700 XT used for earlier measurements is gone; numbers labeled RX
6700 XT below are historical. The CUDA toolkit 12.0 (`nvcc`) is installed
too. Cached models (2026-11-16): `small` and `large-v3` (re-downloaded,
SHA-1 ad82bf6a…, ~19 MB/s from Hugging Face) in
`~/.local/share/pywhispercpp/models/`; `tiny` only in
`~/.cache/narrator-cuda-test/`; the repo's `models/` is empty.
`pytest tests/` and `scripts/render_preview.py` work fully here. Live mic
capture (`sounddevice`/PortAudio) cannot be exercised end-to-end — say so
rather than claiming it was tested. The **web UI can** be tested here: run
`narrator --headless --config <scratch>/config.yaml` (a cached `small` model
starts in seconds) and drive it with curl (put a known password in the
scratch dir first: `python -c "from pathlib import Path; from narrator.web.security import
write_password_file as w; w(Path('<scratch>/web-password.json'), 'password1')"`;
login: `POST /auth` form `password=...` with `Origin:` sets the cookie; send
`Origin:` on every POST) or a
headless browser (Playwright/Chromium in a scratch venv works; the system
Firefox is a snap and is awkward to automate).

Whisper inference and Vulkan GPU acceleration *are* exercisable on this
machine (RTX 3070 + Pentium G4560). `test.mp3` at the repo root is the
reference clip; decode it with
`gst-launch-1.0 filesrc location=test.mp3 ! decodebin ! audioconvert ! audioresample ! audio/x-raw,format=S16LE,channels=1,rate=16000 ! wavenc ! filesink location=/tmp/test16k.wav`
(there is no ffmpeg here) and feed it through the real worker rather than
guessing at latency. `scripts/play_test_audio.sh` routes it into a PipeWire
virtual mic for a full end-to-end run.

## Architecture

### Process/thread model (see `app_controller.py`)

`AppController` wires everything together and is the one place that
understands the full concurrency picture:

- **Audio capture** (`narrator.audio.capture.MicCapture`): a `sounddevice`
  callback thread pushes frames into a local queue.
  `audio.channels` (1-based list, default `[1]`) picks the input channels
  averaged into the mono signal (`narrator.audio.mixing.mix_channels`, shared
  by both sources): the stream opens with `max(channels)` channels, the mix is
  made in the callback and the level meter reads the mix. A selection beyond
  what the device has raises `ChannelSelectionError` from `start()`.
  `audio.device: "ndi:<source name>"` swaps in `narrator.audio.ndi_capture.
  NdiAudioCapture` (picked by `narrator.audio.factory.create_capture`; same
  `start/stop/level/frames` surface, so the feed thread, meter and
  `_restart_audio_capture` don't care). A worker thread discovers the source
  (cyndilib `Finder`, never narrator's own `ndi.source_name` output), receives
  planar float audio, applies the channel mix, resamples to 16 kHz with the
  chunk-continuous `StreamResampler` and re-slices to `frame_ms` frames. A
  vanished source (no audio for 2s) is logged, the level goes to 0 and
  discovery resumes until it reappears; the dead `Receiver` is dropped and a
  new one is built (the `Source` pointer isn't safe to reuse). Only
  `--list-devices`/`list_device_choices` ever list NDI sources, with a short
  timeout, and return none when the NDI runtime is missing. The channel check
  for NDI happens on the first frame (the count isn't known earlier): logged
  as an error, level stays 0, no audio is fed.
- **Feed thread**: drains `MicCapture`'s queue into the ASR worker's
  `multiprocessing.Queue`.
- **ASR worker**: a *separate OS process* (`narrator.asr.worker.run_asr_worker`),
  not a thread — isolates whisper.cpp native-code crashes / stalled GPU calls
  from rendering, NDI output, and the web UI. Launched with the `spawn` start
  method (forced in `AppController.__init__`): forking from the multi-threaded
  app SIGSEGVs the child inside ggml's Vulkan init on any post-startup ASR
  restart (measured: exit -11 within ~2s, every time; the first spawn from a
  just-started app happened to work, which masked this for a long time).
  `AppController._supervise_asr_loop` watches it and restarts on death with
  backoff; after `_QUICK_CRASHES_BEFORE_HALT` (3) crashes within
  `_QUICK_CRASH_WINDOW_SECONDS` (15s) of spawn it **halts ASR** (no respawn,
  `_asr_halted`, `resolved_asr_backend = "stopped"`) and the "ASR stopped"
  problem appears in the web UI — **failure policy (user decision
  2026-10-17): nothing stops the app, and the app never changes the unit,
  backend or model by itself.** NDI output, the web UI and manual captions
  keep running; a fix from the problems panel (`apply_fix("asr.retry")`) or
  any `asr` config change starts a fresh generation with a clean streak. The
  old "stop the app, exit 1" policy and, before it, the crash-loop CPU force,
  the init-fallback and the transcribe-time fallback are all gone. "Use CPU
  instead" exists only as an operator-clicked fix (written to config.yaml
  like a Save). A dying worker sends a `CaptionEventType.ERROR` with its
  exception text first (`asr_last_error`, shown in the problem); native
  crashes are described from the exit code (-4 SIGILL, -11 SIGSEGV, ...).
  While halted the feed thread stops queueing frames (a full `mp.Queue`
  pipe nobody reads blocks its feeder thread and with it interpreter exit —
  seen live: SIGTERM never finished); `stop()` also drains the queue and
  calls `cancel_join_thread()`. Restart drains stale audio frames so the next
  generation doesn't caption pre-restart speech, and the worker releases the
  whisper model deterministically in a `finally` block.
- **Event-drain thread**: applies `CaptionEvent`s (PARTIAL/FINAL/CLEAR/WIPE/STATUS;
  ERROR = the dying worker's reason, stored only) from the worker's event queue to the renderer.
- **Render + NDI send**: a fixed-fps thread
  (`narrator.ndi.sender.run_render_send_loop`) independent of the ASR/audio
  threads, so caption cadence never blocks video output.
- **Web UI** (`narrator.web.server.WebServer`, attached by `main`): one
  thread running its own asyncio loop (aiohttp). It only reads controller
  state (`status_snapshot()`, plain attribute reads) and calls the operator
  actions; caption events reach it through `AppController.ui_events`, a
  bounded drop-oldest deque, and each browser has its own bounded drop-
  oldest queue, so a stalled tab can never slow the drain/render/ASR
  threads. Preview PNG encoding runs in the executor, never on the loop.
- **Config watcher**: a `watchdog` thread; changes are applied via one atomic
  in-memory swap (see below).
- **Setup checks** (`narrator.health.HealthMonitor`): one thread; see
  "Setup checks + problems panel" below. Never on the ASR/render/drain path.
- **Unit usage** (`narrator.health.usage.UnitUsageMonitor`): a ~1 Hz
  thread sampling every processing unit's load (below); never on the
  ASR/render/drain path, readers get the last snapshot.

### Processing units, model catalog, backend interface (todo C6)

- **`asr.unit`** (replaces `asr.device` + `asr.gpu_device`; the
  `_migrate_device_keys` before-validator maps them with a log line: cpu →
  cpu, an index → `vulkan:<n>`, else auto; an explicit `unit` wins):
  `cpu`, `vulkan:<n>` (ggml's Vulkan index), `cuda:<n>` (nvidia-smi index =
  PCI order; usable with the CUDA build, `make asr-backend ASR_GPU=cuda`,
  which `WhisperBackend._init_cuda` loads with gpu_device = that index and
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`; verified on the RTX 3070 2026-11-16 through
  `WhisperBackend` and in the live app — see "CUDA build" above), or `auto` = the recommended unit resolved at each worker start:
  CUDA > discrete Vulkan GPU > CPU > integrated GPU (an iGPU only when
  nothing else runs the model — a change from the old auto, which took an
  iGPU over the CPU). `asr.unit_identity` = "name @ PCI bus id" of the GPU
  the operator picked (the web UI sets it; "" = unchecked); a device at that
  index that doesn't match is a C8 problem and `WhisperBackend` refuses it
  (`GpuSelectionError`) — never a silent re-map. With name + PCI on both
  sides both must match (same card, other slot = mismatch on purpose).
- **Discovery** (`asr/units.py`, `discover_units(probes)`): every unit with
  `usable` + `reason` (CPU: `cpu_backend_problem`; CUDA: needs `ggml-cuda`
  in the pywhispercpp dist files; Vulkan: needs `ggml-vulkan`). Probes are
  duck-typed: `units.LocalProbes` (in-process: worker, bench, narrator-tune)
  and `health.checks.Environment` (cached, Vulkan in a subprocess). Vulkan
  enumeration (`asr/vulkan_devices.py`, moved out of backend.py, re-exported
  there) now asks for a Vulkan 1.1 instance and reads the PCI bus id
  (`VK_EXT_pci_bus_info`, only when the device has it), the device UUID
  (a second driver for the same GPU is dropped, as ggml does — ggml may
  keep the other driver's entry, listed last) and the largest device-local
  heap as VRAM; a 1.0-only loader falls back to name/type only. On this box:
  cpu, cuda:0 (RTX 3070; `.venv` has the CUDA build, so the vulkan:* units
  are listed but not usable there), vulkan:0 (RTX 3070, 8 GB),
  vulkan:1 (HD 610, integrated). NVIDIA and Vulkan units of one card are
  matched by PCI bus id (`same_card`). A future CUDA+Vulkan build counts
  whisper.cpp's `gpu_device` over every compiled-in GPU backend, which is
  why `build_vulkan_linux.sh` builds exactly one (`ASR_GPU=cuda|vulkan`);
  the backend refuses a unit whose kind the installed build lacks
  (`DevicePolicyError`), since its index would hit another device.
- **Model catalog** (`asr/catalog.py`): entry = model + backend
  (`whisper.cpp` / `onnx` / `nemo`) + files + size + unit kinds
  (`BACKEND_UNITS`: whisper.cpp cpu/cuda/vulkan, onnx cpu/cuda, nemo cuda).
  Only whisper.cpp entries exist (`EXTRA_ENTRIES` is empty; tests add ONNX/
  NeMo ones); an unknown name/path is a whisper.cpp model. Installed via
  `narrator.paths`. `AsrConfig._check_model_unit` rejects a model/unit-kind
  pair the catalog forbids; no pywhispercpp native import (schema-safe).
  Not done: download on request from the UI (the worker still downloads a
  missing known model when it starts).
- **Usage** (`health/usage.py`): NVIDIA from ONE long-lived `nvidia-smi
  --query-gpu=... --loop-ms=1000` process (started only while someone reads
  the snapshot, stopped after 30 s without a reader; a fresh nvidia-smi per
  second costs real CPU on the Pentium), AMD from sysfs (`gpu_busy_percent`,
  `mem_info_vram_*`, hwmon temp/power; no amd-smi/rocm-smi), CPU from
  /proc/stat (per core) + /proc/meminfo (psutil isn't a dependency). Missing
  → None ("n/a"); the Intel iGPU has no counters. Verified live: vulkan:0
  util 24-42% while decoding test.mp3 through the virtual mic, VRAM 939 MB
  with `small` loaded.
- **`AsrBackend`** (`asr/asr_backend.py`, a `typing.Protocol`): transcribe,
  decode_segment, detect_language, set_vocabulary, close, plus
  `resolved_device_description`, `last_word_times`, `rejected_decodes`; the
  shared types (`WordTime`, `SegmentToken`, `SegmentResult`,
  `DevicePolicyError`, `ModelLoadError`) live there (backend.py re-exports
  them). `WhisperBackend` subclasses it; `decoder_common`,
  `streaming_decoder`, `segmenter` and `stream_sim` type against it and never
  import backend.py; the worker, bench and preset probe build theirs with
  `create_backend(config)` (catalog → implementation). Decode behavior is
  unchanged, but narrator-tune's code signature covers backend.py, the
  decoders, stream_sim and now asr_backend.py, so this edit invalidated its
  cached decodes (results.jsonl); cost models are keyed by the unit's kind
  (`units.cache_device_key`: vulkan:0 → "vulkan", as asr.device was spelled)
  and stay valid.
- **Elsewhere**: bench / `--pick-preset` / preset_probe pass `unit` (+
  `unit_identity`) instead of device/gpu_device (bench still accepts the old
  payload keys); narrator-tune passes the unit as its `device` string and
  presets.yaml's `tune.unit` (`tune.device` still read) sets a tier's unit.

### Config: single source of truth, hot-reloadable

`narrator.config.loader` treats the YAML file at the platform user config dir
(`~/.config/narrator/config.yaml` on Linux) as the single source of truth.
Whether edited by hand or via the web UI's Save button, changes go through the
same path: file write → `watchdog` detects → `RootConfig.model_validate` →
atomic in-memory swap (`AppController._on_config_changed`). Invalid edits are
logged and ignored; the last-good config stays live. This keeps headless and
web-driven operation behaviorally identical.

**Language handling** (`asr.language_whitelist`, the single language control —
edited via the web UI's picker + Add / chip-remove editor on the ASR tab):
empty = auto-detect, each utterance decoded in whatever language whisper
reports (any of its 100). One entry = that language permanently (no
detection runs). Several entries = per-utterance selection: the
highest-confidence allowed language wins, decided by re-argmaxing the
detector's full probability table (backend.py `_restrict_to_whitelist` —
exact, not a guess; a pl+en venue can never produce German). The decision
is gated on the **margin** between the top two allowed candidates
(`_LANGUAGE_MIN_MARGIN` = 0.15): an ambiguous window (e.g. a 3-way split)
is not locked — the growing window is re-detected every step until decisive,
which is how confidence "builds up"; decode steps before the lock are
skipped and the post-lock decode covers the whole window. At force-finalize
an ambiguous result keeps the previous utterance's language (wrong-but-
consistent beats coin-flip mixing); with no history the top candidate is
used (dropping would hide audible speech). Detection cost: `auto_detect_
language` pads its mel to a fixed 30s window, so its cost is
**independent of slice length**: RX 6700 XT ~445ms on large-v3 (tiny 28ms /
small 82ms / medium 245ms; mel itself ~5ms); RTX 3070 + whisper.cpp 1.9.5
large-v3 0.19/0.19/0.21 s for 0.5/2/10 s slices, small 0.04–0.06 s. A smaller detection model is NOT viable: small/medium confuse
Polish with Russian/Ukrainian on mid-clip 2s slices (7/12 and 8/12 vs
large-v3's 12/12). Note: whisper transcribes; it has no X→Polish
translation. This replaced an older three-mode design (forced
`asr.language` + `language_switch_threshold` streak + "auto"), removed in
favor of the margin-gated whitelist — one knob, per-utterance, no streak
delay. Detection always goes through the dedicated `auto_detect_language()`
API (pywhispercpp 1.5.1's `detect_language=True` transcribe param is broken:
always zero segments — re-verified 2026-10-17 on whisper.cpp 1.9.5, still
zero; `language=""` alone does auto-detect inside the decode). The lock also needs ≥1.5s of audio (0.4s
slices misdetect 1-in-7 on large-v3); decode steps before it are skipped. A
long-utterance rollover keeps the lock (it is the same speech); a real pause
re-arms it. Large-v3's detector can still drift to a related language on the
live path (measured: Cyrillic captions for Polish), while medium/small do
not — a model-behavior limit, not a policy knob. True X→Polish translation is
impossible with whisper (its only translation target is English).

**Performance presets** (`narrator.config.presets`, `presets.yaml`, tiers
most accurate first: quality/balanced/light/cpu). `asr.preset: <tier>`
supplies only `model`, `step_ms`, `agreement_passes` (never device or
whitelist); `custom` (the schema default, so old configs are unchanged) =
none. Resolved in `AsrConfig._apply_preset` (`mode="before"`, on the raw
dict, so an explicit value in config.yaml always wins and the app only sees
plain values). `save_config` omits covered fields whose value **equals** the
preset's (`config_to_yaml_data`) — by value, not provenance, because the
GUI/web Save rebuilds AsrConfig from a dump + form and would otherwise freeze
preset values into explicit overrides. defaults.yaml ships `preset: quality`
= large-v3, step 500, 2 agreement passes, plus `vad_preroll_ms: 300` in
defaults.yaml. History (2026-11-17): an accuracy-first quality tier (step
1000 × 3 passes, dev WER 31.7% → 27.0%) was shipped and reverted the same
day — live, the user saw captions ~14 s behind speech and asked for ≤ ~5 s.
**Judge settings by ON-SCREEN delay, not FINAL delay**: `layout.pacing`
adds to it (replay of cached FINALs through `CaptionPacer` at 30 fps, p50 /
p90 on 91 dev excerpts): step 500/2 passes: off 1.9/4.1 s, char 3.7/5.6,
word 2.2/4.7, sentence 2.5/4.4; step 1000/3 passes: off 3.7/7.4, char
5.6/9.2. `char` pacing reveals 10 chars/s below 20 queued chars, slower than
Polish speech (~15/s), so it always queues. Screen-set singles (30 excerpts,
saved ± SE vs 500/2): preroll 300 +87 ± 72 (WER 26.8 → 24.8%, no delay),
step 1000 +36 ± 14, passes 3 +36 ± 12, prompt_carryover 200 −27 ± 51,
segmenter +46 ± 75 (p90 11 s), min_mean_token_p 0.85 vs 0 identical.
Reports: `var/tune/reports/accuracy-variants-r{1,2}.{json,log}`. All rows are `measured: false` (provisional,
from today's defaults and the stress-harness baselines) until
`narrator-tune presets` runs. `narrator --pick-preset`
(`narrator.asr.preset_picker` + `preset_probe` subprocess): bench.py screen,
~1 min of measured stream_sim decodes of the bundled Piper clip
(`asr/data/preset_clip_pl.wav`), fitted `DecodeCostModel`, one replay per
tier → first tier with load ≤ 0.7 and delay p50 ≤ 3 s; writes only
`asr.preset` (`loader.write_asr_preset`: line edit keeping comments,
`.bak`), never the device; skips models not cached locally unless
`--download`; warns when explicit covered fields in config.yaml override the
chosen preset. Measured on the RX 6700 XT (`small` only cached, Vulkan, historical): light = 23% load,
delay p50 1.13 s. Needs `python -m` subprocesses, so not in the frozen build.

Most fields hot-reload without a restart. Exceptions handled explicitly in
`_on_config_changed`: an `ndi`/`canvas` change tears down and restarts just
the NDI stream (`_restart_ndi_stream`, its own stop event separate from the
app-wide one); an `asr` change restarts the ASR worker process — except
one confined to `asr.vocabulary`/`vocabulary_list`/`vocabulary_snap` and/or
the `captions` section, which is sent as a `TextSettings` over the worker's
per-generation control queue and applied live (no model reload); an `audio`
change tears down and restarts just mic capture and its feed thread
(`_restart_audio_capture`, same pattern, its own stop event).

Schema lives in `narrator.config.schema` (Pydantic). **There is deliberately
no `font_size` field anywhere** — font size is always derived at render time
from canvas height, margins, line count, and interline gap (see
`narrator.captions.layout`); don't add one. Cross-field validation (e.g.
`margin_top_pct + margin_bottom_pct < 100`, interline gap must fit in the box
height) lives in `model_validator` methods on the schema — extend those
rather than validating ad hoc elsewhere. Shipped defaults:
`src/narrator/config/defaults.yaml`.

### Two live decoders, one setting (`asr.decoder`, `narrator.asr.decoder_common`)

`asr.decoder: local_agreement` (the default, `streaming_decoder.StreamingDecoder`,
below) or `segmenter` (`segmenter.Segmenter`, the obs-localvocal architecture,
see "Segmenter decode" further down). `decoder_common.create_decoder` picks
one for the ASR worker, `stream_sim`, narrator-tune's probe and the eval
scripts; both have the same driving surface (`CaptionDecoder`: drain-then-
tick `feed(frame, run_decode_step=False)` + one `tick()`, events through
`on_event`, `set_text_settings` for live TextSettings). The guards both must
keep live ONCE in decoder_common: `HypothesisText` (text filters → snap →
blocklist → repetition collapse), `BoundaryGuard` (cross-FINAL dedupe),
`LanguageLock` (margin-gated whitelist lock, ≥1.5 s, rollover keeps it, a
pause re-arms it), `SilenceTimers` (CLEAR prunes only off-screen history,
WIPE opt-in). Changing `asr.decoder` is an `asr` change: the worker restarts.
The enum leaves room for a third decoder (`simulstreaming`, todo C5); the
loser of the C5 shoot-out is deleted only then (todo B Phase 8 is NOT done).

### Streaming ASR decode (`narrator.asr.streaming_decoder.StreamingDecoder`, `asr.decoder: local_agreement`)

Implements the "whisper_streaming" local-agreement pattern: whisper.cpp is
re-run on a growing audio window every `step_ms`; the new hypothesis is
diffed word-by-word against the last `agreement_passes` hypotheses, and a
prefix is only committed (emitted as a FINAL `CaptionEvent`) once it's been
stable across all of them. This trades latency (bounded by`step_ms * agreement_passes`) for captions that don't visibly
backtrack. VAD (`narrator.asr.vad.SileroVad`, ONNX) detects speech/silence;
on silence past `end_silence_ms` whatever remains is force-finalized,
bounding worst-case latency. **Continuous speech rolls over** at the smaller
of `max_utterance_ms` and `max_context_seconds`: the utterance is
finalized and the next frame starts a fresh one (fresh whisper context and
hypotheses; the language lock carries over). Once the window is 60% full the
first ≥160ms pause triggers it (a clean cut between words), otherwise the
cap does. **The window is never trimmed mid-utterance**: it used to be
trimmed to `max_context_seconds`, which shifted every hypothesis under the
committed-word count, stalled commits, and made the finalize emit only the
tail — ~5s of speech silently lost per long utterance (documentary
narration runs past 20s without a pause for 40-65% of its speech).
`tests/test_streaming_decoder.py::test_long_speech_loses_no_words_across_rollovers`
guards this.
**The Silero v5 LSTM state drifts on long runs**: fed continuously for
minutes (music, crowd noise), probabilities collapse toward zero and stay
collapsed when speech resumes — measured on a real 88-min assembly
recording: after ~4 min of concert, speech scoring 0.58 mean on a fresh
state scored 0.004 on the drifted one, i.e. captions silently die.
`SileroVad` therefore auto-resets its state after
`_RESET_AFTER_QUIET_SECONDS` (10s) of continuously sub-threshold windows
(music reads as quiet, so the reset keeps firing through songs; a reset
can never land mid-utterance since end_silence_ms ≪ 10s). Keep that
mechanism when touching vad.py — removing it reintroduces dead captions
after long non-speech stretches.

whisper.cpp's built-in ggml Silero VAD was evaluated on 1.9.5 (2026-10-17)
and is **not** a replacement: (1) pywhispercpp 1.5.1's standalone VAD
binding is broken (`whisper_vad_init_from_file_with_params` raises
"Unable to convert function return value" — its context wrapper type is
never registered), and 1.9's streaming entry points
(`whisper_vad_detect_speech_no_reset`, `whisper_vad_reset_state`) are not
bound at all, so it cannot be driven frame by frame from Python; (2) the
only usable form is `vad=True` + `vad_model_path` inside a decode, which
cuts silence out of the window before decoding (CPU-side; +0.15 s on a
20 s large-v3 window, changes the text and merges segments) — it does not
tell the decoder when an utterance starts or ends. Our ONNX `SileroVad`
(10 s quiet-reset kept) stays.

**Ingest is deliberately decoupled from inference.** A decode step costs far
more than one frame of audio (0.3–1.7s vs 30ms), so the ASR worker drains
every queued frame with `feed(..., run_decode_step=False)` and then calls
`tick()` once. Feeding one frame per decode instead makes the worker consume
~1 frame per decode step while the mic produces ~33/second, so
`AppController`'s bounded queue saturates and its drop-oldest policy discards
most incoming speech — measured at 43% of audio dropped before this was
fixed. Keep any change to that loop drain-then-tick shaped.

**Hypotheses pass through `_collapse_repetitions`** before agreement
comparison and commit: streaming re-decode tends to emit immediately repeated
phrases ("niż niż", "15 15 tysięcy") that would otherwise reach the caption.
Repeats are matched case- and punctuation-blind (`_repetition_key`), up to
32-word phrases, keeping the first occurrence's text.
A cross-commit guard (`_drop_repeated_boundary`) additionally drops a new
FINAL's first word when it duplicates the previous FINAL's last word (the
same artifact straddling two commits). Tradeoff: a speaker's emphatic repeat
("tak, tak") becomes "tak" — right side of the trade for a lower-third.

Two whisper parameters in `backend.py` exist purely to bound decode latency
and should not be removed casually: `temperature_inc=0.0` disables whisper's
temperature-fallback ladder (which re-decodes a low-confidence window up to
six times — observed turning a 1.6s window into a **17-second** stall), and
`audio_ctx` shrinks the encoder's fixed 30s context to the actual window
length (~3x faster on short windows). `initial_prompt` is set on EVERY call
(pywhispercpp keeps decode params between calls, so a prompt would leak into
later decodes otherwise).

Two optional decode aids, measured by autoconfigure (pre-roll ships at 300
since 2026-11-17, carry-over stays off — it measured worse):
`asr.vad_preroll_ms` prepends the audio from just before the VAD fired
(Silero needs a window or two to cross its threshold, clipping first
syllables), and `asr.prompt_carryover_chars` passes the tail of the
already-captioned text as whisper's prompt for the next utterances (reset by
a WIPE and by a language change).

**Decode filters ported from obs-localvocal** (`backend.py`
`decode_reject_reason` / `strip_bracketed_annotations`, decoder
`_hypothesis_words`):

- Two quality gates check each decode's token data, which is read from the
  whisper context because pywhispercpp's `Segment` has no tokens. A rejected
  decode counts as silence for that step:
  - `asr.max_timestamp_ratio`: whisper's timestamps run past N× the audio
    length;
  - `asr.min_mean_token_p`: mean text-token probability falls below a
    threshold.
- Both gates are **off by default** and searched by autoconfigure. Measured
  2026-10-13 on `small` over 10 min of `test.aac` (WER 77.2% for both
  configs): the timestamp gate at 2.25 rejected 3 decodes and changed
  nothing; the mean-p gate at obs-localvocal's 0.4 rejected 73 decodes,
  including real Polish speech (genuine speech scores 0.37–0.39 on small),
  and was slightly worse.
- The timestamp ratio is measured against max(audio, 1s): the encoder
  context never shrinks below ~5s, so short windows legitimately end on late
  timestamps.
- `asr.strip_bracketed_annotations` (on by default) removes whisper's
  `[oklaski]`/`[muzyka]` annotations, which are mostly hallucinated over
  noise and applause. It removed 33 inserted words on the same slice.
- `asr.text_filters` holds operator find-and-replace regexes (validated in
  the schema), applied to every hypothesis **before** agreement, so a
  filtered tail cannot block commits.
- Autoconfigure never writes `text_filters` or `strip_bracketed_annotations`
  (operator decisions), but they key cached results.
- `asr.flash_attn` is opt-in: it runs fine on RADV but measured no speedup
  there, and it SIGILLs on other drivers. On the RTX 3070 (1.9.5, test.mp3)
  it works and is faster — 20 s window 0.69 → 0.57 s, 10 s 0.36 → 0.30 s —
  but the text changes (first 20 s: "Witam… Jest dużo głośna. Zastanawię"
  vs "Cześć… Jest dużo góry. Zastanawiam się"), so it stays off until
  autoconfigure has compared it on the library. Note whisper.cpp's default
  for `flash_attn` is ON since 1.8; the backend always passes it explicitly.
- The token API used here (`whisper_full_n_segments/_n_tokens/
  _get_token_data`, `whisper_token_to_bytes/_eot/_beg`; `whisper_token_data`
  fields id/p/plog/pt/ptsum/t0/t1/t_dtw/tid/vlen) is unchanged on 1.9.5.
  1.9's new `whisper_full_get_token_t0/t1` (VAD-mapped times) and
  `whisper_full_n_vad_segments` are not bound by pywhispercpp 1.5.1.
- Beam search, re-verified 2026-10-17 on the RTX 3070: it runs on the GPU
  in both 1.8.5 and 1.9.5 (beam 5, patience −1 or 1.0, no 'patience' error
  or CPU fallback) at about 2× greedy cost (20 s window 1.55 s vs ~0.7 s).
  The old failure was an RX 6700 XT/RADV observation. Still not used: greedy
  + `temperature_inc=0` is the latency guard.

**Event vocabulary + blocklist** (`asr/vocabulary.py`, `captions/blocklist.py`,
`asr/text_settings.py`):

- `asr.vocabulary` (+ the named list `asr.vocabulary_list` =
  `<config dir>/vocabulary/<name>.txt`, merged after the inline entries) is
  put at the FRONT of whisper's `initial_prompt` on every call
  (`build_initial_prompt`), ahead of the carry-over. whisper.cpp keeps only
  the last ~224 prompt tokens, i.e. it would cut the vocabulary first, so
  with a vocabulary the prompt is capped at `PROMPT_MAX_CHARS` and the
  carry-over is truncated first. The token count is a char estimate (1.5
  chars/token): pywhispercpp 1.5.1's `whisper_tokenize` binding has no usable
  output buffer. With an empty vocabulary the prompt is unchanged.
- `asr.vocabulary_snap` (off): `VocabularySnapper` replaces hypothesis
  words within edit distance 1 (2 from 9 letters) of a vocabulary word,
  ≥5 letters both sides, diacritics-blind; a prefix relation
  ("Kowalskiego"/"Kowalski") is inflection and is never snapped.
- `captions.blocklist`: whole-word, case/diacritics-blind (`ł` folds too),
  `stem*` entries match words STARTING with the stem only (no hits inside
  longer words); action mask (`k***a`)/replace/drop;
  `blocklist_builtin` adds `captions/data/blocklist_starter.txt`.
- Order in `StreamingDecoder._hypothesis_words`: text_filters → snap →
  blocklist → repetition collapse, all before agreement. The blocklist is
  also applied to manual captions (`submit_manual_text`). The backend's
  raw output is never filtered, and a decoder built without `text_settings`
  has no blocklist — that is the bypass for offline eval.
- Tuning keys cached results on `vocabulary`/`vocabulary_snap` and never
  writes them. It sees the inline vocabulary only (not `vocabulary_list`).
- Editing a list FILE is not watched: it is re-read when config.yaml changes.

**Removed 2026-11-16 (user decision): the transcript file and session
recording.** `narrator.captions.transcript`, `narrator.recording`, the
`transcript:` / `recording:` config sections, the recording shortcut,
`/api/recording*`, the OSC `/narrator/recording` address, the NDI REC dot,
the recordings disk check and the `soundfile` dependency are gone; old
configs with those keys load with a log line
(`RootConfig._drop_removed_features`). `asr/offline.py` keeps only
`speech_spans` (scripts/eval_transcript.py). Kept on purpose: the decoder's
word times below (CaptionEvent.start/end, `asr.dtw_word_timestamps`) — they
have no consumer in the app now, but removing them would change backend.py
and the decoders, i.e. invalidate narrator-tune's cached decodes.

**Caption-event word times** (`CaptionEvent.start/end`): the local-agreement
  decoder stamps a FINAL with **whisper's word times** of the decode that
  committed it: `WhisperBackend` decodes with `token_timestamps=True`
  (whisper's timestamp heuristic; free — 0.380 vs 0.381 s median over 30
  decodes, same text) and exposes `last_word_times` (`words_from_tokens`:
  tokens grouped at leading spaces, so the words are the text's
  whitespace split; bracketed annotations dropped like the caption drops
  them; times clamped to the window). `decoder_common.align_word_times`
  maps them onto the caption words after text filters / snapping /
  blocklist / repetition collapse (difflib on `_repetition_key`). With
  `asr.dtw_word_timestamps` (off) the word start comes from whisper's DTW
  alignment instead (alignment-head preset by model name; ignored while
  `flash_attn` is on, a whisper.cpp limitation). Without word times (test
  fakes) the old estimate applies: words placed proportionally over the
  window's speech. Times are converted with the fed-sample count and the
  clock of the latest feed, so they are free of the caption delay.
  Measured 2026-10-17 on Piper clips joined by known 0.5 s pauses
  (`|error|` of each clip's first-word start / last-word end, 24 points):
  proportional 0.41 s mean (small) / 1.95 s (large-v3, inflated by a
  repetition loop); token timestamps 0.14 / 0.25 s (median 0.15 / 0.08);
  DTW starts 0.15 / 0.07 s (max 0.26 / 0.15). DTW costs ~20% more decode
  time (large-v3 +0.15–0.36 s per 11–15 s window, small +0.06–0.11 s), hence
  off by default. The segmenter stamps whole segments from its own VAD
  bookkeeping and is unchanged.

### Segmenter decode (`narrator.asr.segmenter.Segmenter`, `asr.decoder: segmenter`)

The obs-localvocal architecture (todo B): Silero cuts the audio into
segments and each CLOSED segment is decoded ONCE (`WhisperBackend.
decode_segment`) — that result is the FINAL; no agreement, no diffing.
With `show_partial_captions` the open segment is re-decoded whole (a
"peek", `decode_segment(peek=True)`) after every `partial_latency_ms` of new
audio and shown as a PARTIAL that REPLACES the previous one; partials are
never committed. Partials off = no peek decodes at all (saves GPU).

- **Decode core** (`backend.py`, port of LocalVocal's
  `run_whisper_inference`): `pad_segment` (under 1 s → centred in 1.01 s of
  ±0.01 noise seeded by the length, so replays pad identically; 10 ms of
  silence each side; under 50 ms → not decoded), the same latency params as
  `transcribe` (greedy, `temperature_inc=0`, `audio_ctx`, prompt on every
  call), tokens read from the whisper context, `filter_segment_tokens`
  (special/timestamp ids from the model's own `whisper_token_eot` — never
  50256, large-v3's vocabulary differs; with `drop_trailing_period` a final
  lone "." or ellipsis is dropped: used for peeks and mid-speech cuts, a
  segment that ended at a pause keeps its sentence end), byte-joined text
  (Polish letters split across tokens), bracket stripping, then both
  quality gates. `SegmentResult` carries text, tokens (id/text/p/t0/t1),
  language, decode time, reject reason.
- **Modes** (`asr.segmentation_mode`): `vad` (default; a segment = one
  stretch of speech, opened at the first speech window with 100 ms of
  pre-roll, closed after `end_silence_ms`), `hybrid` (`segment_duration_ms`
  segments; peeks only once Silero heard speech; a speechless buffer keeps
  its last 250 ms and is never decoded), `fixed` (LocalVocal's "VAD off":
  fixed segments, no VAD, so no silence timers).
- **Segment cap** (vad mode, narrator's addition; LocalVocal's is ∞):
  `max_segment_ms`, the first ≥160 ms pause once 60% full cuts, else the
  cap; a cut at the cap opens the next segment AT THE NEXT SAMPLE — nothing
  is trimmed or lost (`test_long_speech_loses_no_words_across_segment_caps`
  is the segmenter's copy of the no-loss guard). The language lock carries
  across a cut; a real pause re-arms it. hybrid/fixed segments decide their
  language afresh (they are not speech-aligned).
- **Silero hygiene**: `SileroVad.reset_state()` on every transition into
  non-speech (LocalVocal's rule), and the 10 s quiet-reset backstop stays.
- **Language**: detection once per segment on the segment's audio (≥1.5 s,
  margin gate; an ambiguous result is retried at the peek cadence, not per
  tick). Peeks wait for the lock; with partials off the FINAL detects on
  the whole segment. FINAL fallback: previous segment's language, then the
  top candidate.
- **Text**: PARTIAL and FINAL both go through `HypothesisText`; the FINAL
  through `BoundaryGuard.apply`, a peek through `BoundaryGuard.peek`
  (display-only, does not advance it). Empty / punctuation-only text → no
  event; a FINAL that comes out empty sends PARTIAL "" so no stale partial
  stays on air.
- **Context**: the last `context_sentences` FINALs (default 1, 0 = none)
  are whisper's prompt (after the vocabulary); reset by WIPE and on a
  language change. Replaces `prompt_carryover_chars` for this decoder.
- **Times**: FINAL `start/end` = segment start (incl. pre-roll) / end minus
  the trailing silence, on the feed clock.
- **Peek cost** (todo B §5): the interval grows 0.1 s per held second, and
  a peek waits until 2× the previous peek's cost has passed since it
  started (cost measured at the next tick — live the peek ran inside the
  previous tick, in stream_sim ticks only run when the worker is free).
  FINAL decodes run in `feed()` the moment a segment closes and never wait
  for peeks.
- **Display**: a PARTIAL replaces the whole in-progress text
  (`CaptionRenderer.set_partial_preview`), a FINAL replaces it under the
  renderer lock (no frame shows both). `layout.pacing` (off|char|word|
  sentence|smooth, `captions/pacer.py`, port of LocalVocal's TokenBufferThread)
  types text out between the event drain and the buffer; speed adapts to
  the backlog; a new partial pops the previous one's unshared units; no
  `maxTime` clear (retention rules win). Works with either decoder.
  `smooth` (narrator's own, 2026-11-17, user request "continuous rather
  than bigger chunks"): characters at a steady rate = the speaker's chars/s
  over the last `SMOOTH_RATE_WINDOW` (6 s) + (backlog − rate ×
  `SMOOTH_LAG` 0.8 s) / `SMOOTH_CATCHUP` 1.5 s, clamped 8–80 chars/s.
  Replay of cached FINALs (91 dev excerpts, step 500/2 passes; on-screen
  delay p50/p90, CV of chars per 0.5 s while typing, empty 0.5 s windows):
  word 2.2/4.7 s, 0.89, 28% stalled; char 3.7/5.6 s, 0.43, 4%; smooth
  3.0/5.2 s, 0.56, 6%; off 1.9/4.1 s, 1.28, 39%. Steadier settings (bigger
  lag) cost delay roughly 1:1.
- **Config** (`AsrConfig`): `decoder`, `segmentation_mode`,
  `partial_latency_ms` (1100), `segment_duration_ms` (7000),
  `max_segment_ms` (15000), `context_sentences` (1). Local-agreement-only
  fields (`step_ms`, `agreement_passes`, `max_utterance_ms`,
  `max_context_seconds`, `vad_preroll_ms`, `prompt_carryover_chars`) stay
  for that decoder. `_migrate_decoder_keys` (before-validator, log line):
  decoder aliases (`localagreement`, `segment`, ...), `segmentation_mode:
  disabled` → fixed, `duration_filter_threshold` → `max_timestamp_ratio`,
  `n_context_sentences` → `context_sentences`, and for a segmenter config
  without `max_segment_ms` the smaller of `max_utterance_ms` /
  `max_context_seconds`. `_check_segmenter` cross-checks only what is read
  (decoder segmenter; cap/end_silence in vad mode; segment duration in
  hybrid/fixed; partial cadence with partials on). Presets never touch
  the decoder. The web UI's ASR tab shows each decoder's fields only while
  it is selected (HINTS `when`, `applyConditions` in app.js).

- **Measured 2026-10-17** (`scripts/autoconfigure_reports/segmenter-vs-localagreement.md`,
  31 screen excerpts, paired against local agreement): defaults large-v3 24.5% vs 26.4% WER
  (+85 errors, SE 46), medium 26.8% vs 28.5%; busy 7% (24% with partials) vs 61%; but FINAL delay
  p50 4.8 s / p90 10.7 s (first visible with partials 1.1 s / 2.4 s). Caps short enough for a 3 s
  FINAL p50 (4-6 s) lose to local agreement on WER (cap 6000: 28.9%); hybrid is worse than vad at
  any length. Hence the default stays `local_agreement` (Phase 8 rule: WER within noise AND FINAL
  p50 <= 3 s); C5 decides.

### Rendering pipeline

`narrator.captions.buffer` (rolling caption buffer, what text is currently
"live") → `narrator.captions.wrap` (word-wrap) → `narrator.captions.layout`
(derives font size from canvas/margin/line-count/interline geometry — the
enforcement point for the "no explicit font_size" rule) → `narrator.render.canvas.CaptionRenderer`
(Pillow compositing: text, `background_band`/`shadow`/`stroke` style
overrides) → `narrator.ndi.sender` (`cyndilib`-backed NDI RGBA send,
transparent background).

**Caption retention (do not regress this)**: the ASR worker's silence
timers must never blank what the audience is currently reading.
`prune_history_after_silence_seconds` (renamed from
`clear_after_silence_seconds`; a `model_validator(mode="before")`
migration still accepts the old key) emits CLEAR, which drops only lines
already invisible under `layout.max_lines` (`CaptionBuffer.prune_history`)
— the visible lower-third is byte-identical after. A true screen clear is
operator-initiated (the web UI Clear button / shortcut / control API) or comes as WIPE after
`wipe_screen_after_silence_seconds` of silence (default 0 = off: stale
captions are usually better than blank ones; raise it when very old text
would misattribute to new speech). In manual caption mode both events are
ignored like any other ASR output.

### Setup checks + problems panel (`narrator.health`, todo C8)

- `problems.Problem`: id (stable, dotted), severity error/warning/info,
  subsystem asr/audio/ndi/web/setup, message, `fixes` (action id +
  label). Server fixes: `asr.retry`, `asr.use_cpu`, `asr.pin_unit` (record
  the identity of the device at the configured index), `asr.unit:<unit id>`
  (switch to that unit + record its identity), `audio.retry`, `ndi.retry`,
  `health.recheck`; `ui:<tab>` fixes only open a web UI tab.
- `checks.py`: pure functions over the config, an `Environment` (all probes;
  tests pass a fake) and a `RuntimeState` snapshot
  (`AppController.runtime_state()`). Static: `check_asr_unit` — pywhispercpp
  installed; model/unit kind valid per the catalog; cpu: the source build
  `+whispercpp.X` is `-march=native` = fine, the PyPI wheel needs AVX or it
  SIGILLs; vulkan:<n>: Vulkan build (`ggml-vulkan` in the dist files),
  `libvulkan.so.1`, GPUs, the index present (`asr.unit.missing`, offering
  the recorded card at its new index), identity matches
  (`asr.unit.identity_mismatch` error / `identity_unrecorded` info with
  "Remember this card"), iGPU = info; cuda:<n>: card present, CUDA build
  installed (`asr.unit.no_backend`, offering the same card's Vulkan unit,
  matched by PCI bus id); auto: resolves to CUDA/discrete GPU = fine, to the
  CPU/iGPU = warning saying why, nothing usable = error. Model (`paths.find_model_file`, else
  downloadable per `pywhispercpp.constants.AVAILABLE_MODELS` and disk space
  for it), mic name + channels (only while capture is down: PortAudio does
  not rescan while open), NDI runtime (`import cyndilib`), disk (tune
  cache). Runtime: ASR halted / restarting,
  audio capture failed, NDI audio source status (`NdiAudioCapture.status`:
  runtime_unavailable / searching > 5 s / channel_error), NDI output failed.
  "Use CPU instead" is offered only when the CPU backend runs here, the
  unit isn't cpu already, the model has a CPU build (catalog), and the model
  isn't the cause.
- Vulkan is enumerated in a subprocess (`python -m
  narrator.health.vulkan_probe`, JSON with index/name/type/PCI bus id/VRAM;
  frozen build: in-process), nvidia-smi likewise, cached until a config
  change / fix / re-check (one lock per probe in `Environment._cached`, so
  the health and unit-usage threads never run the Vulkan probe twice): the
  GPU driver is never loaded into the app process. (The first in-thread
  probe SIGSEGVed the app: `_vulkan_devices` passed physical-device handles
  as bare ints, i.e. 32-bit C ints, which truncates the high addresses
  handles get off the main thread; fixed with `c_void_p` / explicit
  argtypes, now in `asr/vulkan_devices.py`.)
- `HealthMonitor`: checks at start, on `request_check()` (config change,
  part failing/restarting) and every 30 s; checks are serialized and a result
  is dropped if the config changed meanwhile. `problems()` = live runtime
  problems + last static list, errors first; new error/warning problems are
  logged once (the journal is the record on a headless box).
- `AppController`: `_start_ndi_stream` / `_start_audio_capture` never raise
  (failure → `_ndi_error` / `_audio_error`, part left stopped, rest of the
  app starts); `apply_fix(action)` re-runs the checks, accepts only fixes a
  current problem offers (else `LookupError` → 409), restarts only that part
  (`_restart_asr_process`, `_restart_audio_capture(rescan=True)` — PortAudio
  re-init so a newly plugged mic is seen, `_restart_ndi_stream`), and
  `asr.use_cpu` writes `asr.unit: cpu` through `save_config` (watcher
  applies it; `asr.unit:<id>` / `asr.pin_unit` the same way, with the
  unit's identity, `_write_asr_unit`). `WhisperBackend` raises `ModelLoadError` when pywhispercpp
  returns a model without a path (it logs "Invalid model name" and wraps
  whisper.cpp's NULL context instead of raising; the first decode would then
  crash natively).
- whisper.cpp loads the model on the **CPU** without an error for a
  `gpu_device` index it doesn't have (measured 2026-10-17, RTX 3070:
  `gpu_device: 5`), so `select_gpu_index` raises `GpuSelectionError` for a
  `vulkan:<n>` outside the enumerated devices, and `WhisperBackend` raises it
  too when `asr.unit_identity` no longer matches the device at that index
  (verified live 2026-10-17: vulkan:1 recorded as the RTX → refused, ASR
  halted, `asr.unit.identity_mismatch` + "Use vulkan:0" fix, which brought
  ASR back): the worker fails, ASR halts and the problem explains it.
  Nothing enumerated (no loader in-process) → the index is passed through.
  `cuda:<n>` on a build without CUDA (or `vulkan:<n>` on a CUDA-only build)
  raises `DevicePolicyError`; `cuda:<n>` absent from nvidia-smi or with a
  mismatching `unit_identity` raises `GpuSelectionError`.

### Web UI (`narrator.web`) — replaced the Tk GUI

aiohttp app on its own thread/loop (`server.py`), plain HTML/JS/CSS in
`web/static/` (package data + PyInstaller datas; no build step). Optional:
`web.enabled`, `--no-web`; the old `gui:` section is migrated to
`web.enabled` by a `RootConfig` before-validator.

- **Security** (`security.py`): listens on 0.0.0.0 by default, so other
  computers can reach it (user decision 2026-11-16, replacing the old
  loopback-only default); `narrator --local-only` (`WebServer(local_only=)`)
  binds 127.0.0.1 and then every request's Host must be localhost/127.x/
  [::1] *with the served port* (DNS rebinding). This is deliberately a
  startup flag: `web.bind` / `web.allowed_hosts` were removed from the
  schema (a before-validator drops them from old configs with a log line).
  Origin, when present, must equal the request's Host (scheme http, same
  host:port) on non-GET requests and the WebSocket (CSRF). Auth (user decision
  2026-11-16: password only, no token): the operator's password, scrypt hash
  in `web-password.json` beside config.yaml (0600, `narrator
  --set-password` or `POST /api/password` {current, new} from the Web &
  remote tab, which also ends every other session), checked at `POST /auth`
  (one check at a time, 1 s pause after a miss, file re-read each time) for
  an HttpOnly SameSite=Strict session cookie. Sessions are in memory: a
  restart logs everyone out (also the user's choice). No password set =
  a browser on this computer (loopback client AND loopback Host, so a
  rebinding page is excluded) gets admin without a login and can set the
  first password on the Web & remote tab (no current password asked; that
  browser gets a session); other computers cannot log in until a password
  exists (`/api/login-info` {password_set, login_required} tells the login
  page; startup warnings say so). No startup token, no Bearer admin access, no `GET /auth`; a stale
  `var/web-token.json` is deleted at start; `var/web-ui.json` (url + pid)
  serves `--open-ui`. `web.api_tokens` (named, revocable by deletion) reach
  only the control API. Static files are served `Cache-Control: no-cache`
  (a heuristically cached old app.js with a new index.html broke the UI).
- **Config**: Save POSTs the whole form; `RootConfig.model_validate` →
  422 with `[{loc, msg}]` shown next to the fields (no JS validation) →
  `loader.save_config` (keeps preset semantics) → the watcher applies it
  like a hand edit. The form is generated from `RootConfig.model_json_schema()`
  plus a HINTS table in `app.js`, so a new schema field shows up without UI
  work; give it a label/widget hint there. Widgets (user request 2026-11-16, "more
  welcoming"): `choice` (common values + "Other…" free input; used for
  channels, fps, lines, wipe timer, ...), `slider` (range + number),
  `segmented` (small enums), `enum` with friendly `labels`, `model` (grouped
  "On this computer" / "Downloaded on first use", from `/api/choices`
  `catalog`), `device`/`font`/`vocablist` (dropdown + "Other…"),
  `resolution` (sets canvas width + height), booleans as switches. `adv:
  true` puts a field under its section's "Advanced settings" fold;
  `FIELD_ORDER` reorders a section; `SECTION_INTROS` adds a line under a
  section title. Text effects (`style.overrides`, widget `effects`): quick
  looks (set only the `enabled` flags) + one card per effect with its
  `enabled` switch in the title (HINTS `toggle`) and the other fields shown
  only while on. "Sample text" (WebSocket `{"type": "sample"}`,
  `PreviewSync.refresh(sample_text=)`) previews a sample caption in that
  browser only — never on air; turned on the first time the Look & layout
  tab is opened. `web.*` changes restart the
  listener via `WebServer.on_config_changed` (called by the controller).
- **Page layout** (user request 2026-11-16): one column — the full-width
  NDI preview with the live controls (Auto/Manual, Clear, Pause, manual
  entry), the settings tabs under it. The "On air" and "What ASR hears" text
  boxes were removed (the preview shows what is on air); the server still
  sends `on_air` and caption events, the page ignores them.
- **Live**: WebSocket `/ws` streams state at 10 Hz (mic level re-read from
  `controller._mic` every tick — `_restart_audio_capture` swaps it; ASR
  Active = `_asr_process.is_alive()`, not the STATUS string), caption events,
  new log lines (`logbuffer.RingBufferHandler`, installed by main before
  start), and preview PNGs (`render.preview_sync.PreviewSync`: the live
  renderer's own frame, or a private renderer for unsaved display drafts) at
  ≤ `web.preview_fps`, only when the frame changed. Text from the server is
  rendered with `textContent` only.
- **Processing units** (ASR tab, `unit` widget + `refreshUnits` in app.js):
  `status_snapshot()` carries `units` (every unit + its live `usage`,
  `recommended`, `selected`), `unit_recommended`, `unit_resolved` (what
  `asr.unit` means now, e.g. auto → vulkan:0), from
  `health.usage.UnitUsageMonitor`. The picker is a compact list (user
  request 2026-11-16: small windows, no usage figures): "Auto (<resolved>)"
  + one "<backend> — <short device name>" per unit (e.g. "Vulkan — NVIDIA
  GeForce RTX 3070", "CPU — Intel Pentium G4560"; `#<index>` only for two
  same-named devices of one backend; "(integrated)" / "(not available)" /
  "(not for this model)" tags, the reason in the tooltip). The usage table
  was removed from the page; the snapshot still carries `usage`. It sets
  `asr.unit_identity` with the pick (hidden field); the model datalist offers
  only models that run on the chosen unit's kind (`/api/choices`
  `catalog`). Checked in headless Chromium.
- **Problems panel** (`#problems` above the page, `renderProblems` in
  app.js, re-rendered only when the list changes): `status_snapshot()`
  carries `problems` (plus `asr_halted`, `audio_running`, `ndi_running`), so
  both `/api/state` and the WebSocket state have them. Fix buttons POST
  `/api/fix` `{"action": ...}` — admin session only (not
  `web.api_tokens`: `asr.use_cpu` writes config), Origin-checked like every
  POST, run in the executor; `ui:` fixes switch tabs client-side.
- **Control API** (`actions.py`, shared with OSC): `GET /api/state`,
  `POST /api/clear|mode|manual-text|pause`. Pause
  (`AppController.set_captions_paused`) works like manual mode: the worker
  keeps running, its output is ignored.
- **OSC** (`osc.py`, `web.osc`, off by default, UDP on loopback): minimal
  built-in OSC 1.0 parser, `/narrator/clear|mode|manual-text|pause`.
- **Shortcuts** (`web.shortcuts`, `config/shortcuts.py` mirrored by
  `comboFromEvent` in app.js — keep them in step): document-level, work while
  typing (plain single keys don't fire in text fields; plain `submit` fires
  only in the manual entry). Browser-reserved combos (Ctrl+N/T/W/Q/Tab,
  Ctrl+1-9, F11, ...) are rejected by the schema — a page cannot intercept
  them, which is why clear defaults to Escape, not the Tk GUI's Ctrl+N.
- The ASR tab's preset dropdown sets model/step_ms/agreement_passes to the
  preset's values (Save then omits them); "Re-run benchmark" runs
  `narrator --pick-preset` as a subprocess, output to the log view.

## Repo layout

- `src/narrator/` — package source (see architecture above for the pipeline through `asr/` → `audio/` → `captions/` → `config/` → `ndi/` → `render/`, plus `web/` for the admin UI).
- `tests/` — pytest unit tests (config validation, layout/wrap, streaming decoder, app controller, stream simulator — all GPU-free via fakes). `tests/tune/`: the tuning tool (library evaluation, autoconfigure search, also GPU-free); `tests/test_no_tune_dependency.py` guards that narrator runs without `narrator_tune`.
- `scripts/stream_eval.py` — streaming-pipeline accuracy harness for one recording slice: replays the app's real decoder (`--decoder local_agreement|segmenter`, plus the segmenter's levers)+WhisperBackend through `narrator.asr.stream_sim` (live drain-then-tick sequencing, each decode charged its real duration, no wall-clock waiting) and scores FINAL captions against the SRT reference. This measures the algorithm that would caption a live event; `eval_transcript.py` measures best-effort offline transcription.
- `scripts/render_preview.py` — headless rendering sanity check, no audio/GPU/NDI required.
- `scripts/eval_transcript.py` — offline ASR accuracy vs an SRT reference: WER + worst-window word diff. Handles live-caption rollup (collapses repeated text), masks `[Muzyka]`/`[Aplauz]` windows and foreign-language song sections (detected by diacritic density — a forced-`pl` whisper *translates* them, which is not an error), and decodes in 60s spans split at the SRT's own non-speech boundaries (monolithic decode lets music hallucination loops poison later speech via prompt carryover — measured: 99% vs 36% WER). `--reuse-segments` rescoring avoids re-decoding. Audio is decoded through a GStreamer pipe (`narrator_tune.real_fixtures.load_audio_16k_mono`); never write decoded audio to temp files — an earlier temp-WAV version leaked 41 × 170 MB into /tmp and filled the root disk.
- `scripts/live_language_demo.py` — plays a pl→en→pl schedule through the virtual mic into the real app and prints the observed caption timeline vs expectations (dynamic language switching end-to-end).
- `scripts/gen_test_clips.py` — generates byte-deterministic Polish TTS fixtures (piper,
  zero noise scales) with known transcripts into `scripts/test_clips/`.
- `scripts/stress_full_pipeline.py` — full-pipeline ASR stress harness: plays the fixtures
  through a PipeWire virtual mic into a real `AppController`, records FINAL events, and
  reports per-clip WER, finalize latency, and cross-pass determinism. Baselines on the
  earlier test box (RX 6700 XT, historical): `small`+step 600 ≈ 7% WER / 0.8s, `medium`+step 350 ≈
  7–10% WER / 0.5–0.8s; `large-v3` used to collapse under streaming (repetition loops,
  multi-second decode steps) despite being clean offline — beam search was tested
  as a loop mitigation and was unusable on the RX 6700 XT (GPU-path 'patience'
  error → silent CPU fallback → 104s decode steps; on the RTX 3070 it runs on the
  GPU at ~2× greedy cost, see Decode filters). **Fixed at the text layer**
  (streaming_decoder.py): large-v3's dominant failure was whole-sentence repetition
  loops (same 6–9 word sentence 5–6× inside one hypothesis), and `_collapse_repetitions`
  now runs to phrase length 32 (a 17-word loop was later measured on the reference
  library) comparing case/punctuation-blind keys (loops vary "jezdnie."/"jezdnie,"), collapsing those loops — harness WER 1.28 → 0.14 mean
  (mixed 11.7 → 0.9, numbers → 0.00). whisper.cpp has no no_repeat_ngram_size (that's
  faster-whisper) and `entropy_thold`/`suppress_regex` don't break in-decode loops, so
  this collapse is the enforcement point. The default model remains large-v3 at the
  operator's explicit choice.
- `src/narrator_tune/` (`autoconfigure.py`, `library_eval.py`, `quality_probe.py`,
  `real_fixtures.py`, `report_library.py`, `ollama_analyzer.py`, `cli.py`) — the separate
  `narrator-tune` tool (autoconfigure). It **imports** narrator's real decode path
  (`narrator.asr` `backend`, `streaming_decoder`, `vad`, `stream_sim`, `bench`, `metrics` stay
  in the app), and narrator never imports it (the web UI, the CLI and the PyInstaller build have
  no autoconfigure; a leftover `autoconfigure:` key in config.yaml is dropped with a log
  line). Its own settings live in `config/tune.yaml` (`narrator_tune.config.TuneConfig`;
  created on first run from that leftover section or `tune_defaults.yaml`). `narrator-tune
  configure` searches the
  model and decode settings under which the LIVE captioner best matches the reference
  recordings in `scripts/test_audio/` (audio + same-title SRT), and writes them into
  config.yaml — only `asr.model` and the decode settings; never the device, the language
  whitelist, display settings or silence timers (previous file kept as `config.yaml.bak`).
  `narrator-tune presets` (`narrator_tune/presets.py`) runs the same search once per preset
  tier on a scratch config (tier model fixed, only step_ms/agreement_passes searched, the
  tier's `tune.max_load`/`tune.device`) and rewrites presets.yaml with `measured: true` rows.
  - **Decoders**: `decoder` is the first searched setting; settings only one decoder reads are
    skipped while the incumbent uses the other (`autoconfigure.DECODER_SPECIFIC`; segmenter levers
    `max_segment_ms`, `segmentation_mode`, `partial_latency_ms` (partials on only),
    `context_sentences`, plus the shared `end_silence_ms`, `vad_threshold`, both quality gates).
    The delay cap is on the FINAL delay; the first-visible delay (partials) is reported beside it.
    `show_partial_captions` keys segmenter results (peeks load the worker) but not local-agreement
    ones. The result key's code hash is per decoder (`library_eval.code_signature(decoder)`:
    the other decoder's module is left out), so segmenter edits keep the local-agreement cache.
    `scripts/baseline_eval.py --variant NAME:FIELD=JSON,...` scores configs paired against the
    base on the same excerpts (the Phase 0 / segmenter comparison driver).
  - **Instrument**: `stream_sim.simulate_stream` replays the configured decoder on a
    virtual clock — frames arrive at audio time, every backend call occupies the worker for
    its cost, `tick()` runs only when the worker is free (the worker's drain-then-tick),
    FINALs are stamped with the completing call's time. Inference is real; only waiting is
    skipped, so a run costs the decode time (large-v3 on dense narration ≈ 0.7× real time).
    The old probe replayed in real time with `sleep` and timed out every candidate. The cost
    (`DecodeCostModel`: base + per window second + per emitted word; constant language
    detection) is benched, then **calibrated** on real streaming decodes (`mode="measured"`)
    and stored in `var/tune/cache/cost_models.json` — bench windows alone
    underestimated large-v3's streaming decodes (dense speech emits more tokens). With a
    fixed cost model the replay is deterministic, so configs are compared without GPU jitter.
  - **Library**: SRT tag decides the role (`(Polish_ASR)`/untagged → objective; `(Polish)`
    human subtitles → separate check; other languages excluded — an English translation of
    Polish audio cannot be scored). Title-hash split dev 75% / holdout 25%; 90s excerpts, ≤4
    per recording, round-robin and hash-ordered (a sample is a prefix of any larger one, so
    adding files never moves cached excerpts). Reference words carry interpolated times
    (latency, and excerpt edges by word time, not whole overlapping cues). Excerpts that even
    OFFLINE large-v3 cannot match (`max_ceiling_wer`) are dropped as bad references.
  - **Scoring**: pooled WER (Σerrors/Σreference words — a mean of per-excerpt WERs
    over-weights near-empty excerpts), S/D/I, caption delay p50/p90 (FINAL emit time −
    reference word time), worker busy fraction, doubled-word and non-Latin rates, top
    inserted phrases. YouTube auto-captions are noisy references, so configs are compared
    PAIRED on the same excerpts (`paired_delta`; noise = paired standard error).
  - **Stages**: bench + calibrate → offline ceiling → model screen (current settings; ranked
    by WER under the load limits only — delay is tunable, so it is not held against a model
    here) → coordinate search on the best model (`SEARCH_ORDER`, delay levers first when the
    start is over the cap; each value is piloted on 4 excerpts
    first and pruned if it clearly overloads the worker or the delay cap; a value is adopted
    only if it saves ≥3 errors and ≥1 SE while keeping delay p50 ≤ `max_latency_s`, busy ≤ 0.9,
    doubled ≤ 0.5%) → dev confirmation (tuned best, runner-up, current; a faster model within
    noise wins) →
    holdout gate (winner ≤ current + 0.5 pt) → write. A report-only whitelist check scores
    the final settings with the reference language alone.
  - **Caching/budget**: excerpt audio in `var/tune/cache/excerpts/` (int16
    .npy, refused below `min_free_disk_mb`), raw decodes in `results.jsonl` keyed by
    (excerpt, model, mode, decode settings, cost model, hash of the streaming_decoder/backend/
    vad/stream_sim/ring_buffer code with comments and docstrings ignored) — reruns resume,
    code edits invalidate. Probes stay
    in subprocesses; a decode that keeps crashing one is skipped. `time_budget_minutes` bounds
    new decoding (search stops early, keeping time for the final stages). Summaries go to
    `var/tune/reports/`; `narrator-tune analyze` feeds them to local Ollama. (The old
    `scripts/autoconfigure_{cache,reports}/` stay in use until moved there — `narrator_tune.paths`.)
  - **Measured on the reference machine (2026-10-02, RX 6700 XT, whisper.cpp 1.8.5)**, 32 dev excerpts / 4865 reference
    words, large-v3 with `[pl, en]`, step 800: offline-ceiling WER 19.5%; streaming WER
    28.8% before the rollover fix (581 deletions: ~5s lost per 25s utterance) → 23.9% after
    it (258 deletions) → 22.1% with the case/punctuation-blind 32-word repetition collapse
    too (insertions 402 → 312, doubled words 0.28% → 0; total paired gain 328 errors, ~4 SE).
    Whitelist `[pl]` instead of `[pl, en]` on the same excerpts: 22.8% — fewer English-decoded
    passages but more hallucination over music; no significant difference. Fidelity on 3
    excerpts: simulated 36.0% / delay p50 3.2s, measured-time replay 35.8% / 3.4s, true real-time run 36.5% /
    3.5s. Calibrated large-v3 cost ≈ 0.13s + 0.031s per emitted word (window length barely
    matters); repetition loops (60+ words from a 2s window) cost up to 3.4s per decode.
- `packaging/pyinstaller/` — standalone binary build (spec + per-OS build scripts). Verified 2026-10-17 on Linux (133 MB onefile; `--help`, `--list-devices`, headless run with the web UI, units listed, Vulkan ASR worker up). Frozen rules: `main()` calls `multiprocessing.freeze_support()` (the spawn-started ASR worker re-executes the binary); shipped files are found via `paths.bundled_assets_dir()` (`sys._MEIPASS`), never `Path(__file__).parents[n]`; `--pick-preset` (and so the web UI's benchmark) needs `python -m narrator.asr.preset_probe` and does NOT work frozen; Windows build untested.
- `packaging/systemd/narrator.service.in` — user-service template (`@REPO@` placeholder, rendered by
  `make unit`/`make install`): `--headless --config <repo>/config/config.yaml`, `Restart=on-failure`
  with `RestartSteps` backoff + `StartLimit*`, SIGTERM stop, journald. Windows service is out of scope.
- **Self-contained layout** (`narrator.paths`, shared by `narrator_tune.paths`; all gitignored): `config/config.yaml`
  (used only when `--config` is passed; bare `narrator` keeps the platformdirs path), `models/`, `var/`
  (`log/`, `web-ui.json`, `tune/`), `third_party/` (pywhispercpp source + build tree of
  `build_vulkan_linux.sh`; `wheels/` keeps the 1.8.5 rollback wheel). `REPO_ROOT` is detected from the checkout (`pyproject.toml` beside
  `src/narrator`; `NARRATOR_HOME` overrides; None when frozen/site-packages, where every helper falls
  back to platformdirs). Models: `paths.model_load_args` passes a found file's full path to
  `Model()` (looked up in `models/`, then pywhispercpp's legacy user cache so large-v3 is not
  re-downloaded) or, when missing, the name plus `models_dir=<repo>/models` (pywhispercpp's own
  parameter, no monkeypatching). The web UI model dropdown (`config/choices.py`) and `narrator_tune.autoconfigure.model_available`
  use `paths.cached_model_names/find_model_file` — never read `pywhispercpp.constants.MODELS_DIR`
  directly. Not in the repo: apt packages (Vulkan loader/driver, toolchain, PortAudio, GStreamer)
  and the NDI runtime.
- Headless shutdown (`main.run_headless`): SIGTERM/SIGINT set an event, then `AppController.stop()`
  (the one shutdown path; idempotent; stops the web UI first) runs; exit 0 for a requested stop. Since the
  2026-10-17 failure policy no code path stops the app by itself; the "stopped itself → exit 1" branch is
  only a safety net. So the unit's `Restart=on-failure` now applies only when the main process dies
  (unhandled exception at startup such as an unreadable config.yaml, a native crash, the OOM killer).
- `packaging/whisper_cpp/README.md` — GPU/Vulkan runtime requirements, plus source-build instructions for the rare case a non-Vulkan ggml backend is needed.
- `assets/` — bundled font (Polish-capable) and Silero VAD ONNX model, referenced by `packaging/pyinstaller/narrator.spec` for the standalone build.

## Notes

- NDI SDK/runtime is a separate operator install (https://ndi.video), not bundled — check current NDI SDK redistribution terms before packaging the runtime library into a distributed build.
- Full design rationale: `/home/xxx/.claude/plans/i-want-you-to-shimmering-duckling.md`.
