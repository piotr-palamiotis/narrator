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
python -m venv .venv
source .venv/bin/activate
pip install -e .

# run
narrator --list-devices     # sanity-check microphone enumeration
narrator                    # default config, opens GUI unless gui.enabled: false
narrator --no-gui           # force headless
narrator --config path/to/config.yaml

# tests
pytest tests/               # all 34 tests
pytest tests/test_wrap.py -k some_test_name   # single test

# render sanity check without mic/GPU/NDI hardware
python scripts/render_preview.py   # writes scripts/output/*.png

# standalone binary
./packaging/pyinstaller/build_linux.sh
.\packaging\pyinstaller\build_windows.ps1
```

There is no configured linter/formatter in this repo — don't assume one.

### GPU (Vulkan) build

`pip install` gives a CPU-only `pywhispercpp` build. Real Vulkan
acceleration requires rebuilding it from source with `GGML_VULKAN=1` — see
`packaging/whisper_cpp/README.md`. This is a build-machine step, not needed
for the CPU-fallback path.

## Sandbox limitations (relevant when asked to "verify" something)

This environment typically has no microphone, GPU, display, or NDI receiver.
`pytest tests/` and `scripts/render_preview.py` work fully here. Live mic
capture (`sounddevice`/PortAudio), real whisper.cpp inference (sandbox CPUs
often lack AVX2), the Tkinter GUI, and true GPU/Vulkan acceleration cannot be
exercised end-to-end in that setup — say so rather than claiming they were
tested.

## Architecture

### Process/thread model (see `app_controller.py`)

`AppController` wires everything together and is the one place that
understands the full concurrency picture:

- **Audio capture** (`narrator.audio.capture.MicCapture`): a `sounddevice`
  callback thread pushes frames into a local queue.
- **Feed thread**: drains `MicCapture`'s queue into the ASR worker's
  `multiprocessing.Queue`.
- **ASR worker**: a *separate OS process* (`narrator.asr.worker.run_asr_worker`),
  not a thread — isolates whisper.cpp native-code crashes / stalled GPU calls
  from rendering, NDI output, and the GUI. `AppController._supervise_asr_loop`
  watches it and restarts on death with backoff; after
  `_QUICK_CRASHES_BEFORE_CPU_FALLBACK` (3) crashes within
  `_QUICK_CRASH_WINDOW_SECONDS` (15s) of spawn, it force-falls-back to CPU
  for subsequent respawns until the device is deliberately changed.
- **Event-drain thread**: applies `CaptionEvent`s (PARTIAL/FINAL/CLEAR/STATUS)
  from the worker's event queue to the renderer.
- **Render + NDI send**: a fixed-fps thread
  (`narrator.ndi.sender.run_render_send_loop`) independent of the ASR/audio
  threads, so caption cadence never blocks video output.
- **Config watcher**: a `watchdog` thread; changes are applied via one atomic
  in-memory swap (see below).

### Config: single source of truth, hot-reloadable

`narrator.config.loader` treats the YAML file at the platform user config dir
(`~/.config/narrator/config.yaml` on Linux) as the single source of truth.
Whether edited by hand or via the GUI's Save button, changes go through the
same path: file write → `watchdog` detects → `RootConfig.model_validate` →
atomic in-memory swap (`AppController._on_config_changed`). Invalid edits are
logged and ignored; the last-good config stays live. This keeps headless and
GUI-driven operation behaviorally identical.

Most fields hot-reload without a restart. Exceptions handled explicitly in
`_on_config_changed`: an `ndi`/`canvas` change tears down and restarts just
the NDI stream (`_restart_ndi_stream`, its own stop event separate from the
app-wide one); an `asr` change restarts the ASR worker process; an `audio`
change just logs a warning that a full app restart is needed.

Schema lives in `narrator.config.schema` (Pydantic). **There is deliberately
no `font_size` field anywhere** — font size is always derived at render time
from canvas height, margins, line count, and interline gap (see
`narrator.captions.layout`); don't add one. Cross-field validation (e.g.
`margin_top_pct + margin_bottom_pct < 100`, interline gap must fit in the box
height) lives in `model_validator` methods on the schema — extend those
rather than validating ad hoc elsewhere. Shipped defaults:
`src/narrator/config/defaults.yaml`.

### Streaming ASR decode (`narrator.asr.streaming_decoder.StreamingDecoder`)

Implements the "whisper_streaming" local-agreement pattern: whisper.cpp is
re-run on a growing audio window every `step_ms`; the new hypothesis is
diffed word-by-word against the last `agreement_passes` hypotheses, and a
prefix is only committed (emitted as a FINAL `CaptionEvent`) once it's been
stable across all of them. This trades latency (bounded by
`step_ms * agreement_passes`) for captions that don't visibly backtrack. VAD
(`narrator.asr.vad.SileroVad`, ONNX) detects speech/silence; on silence past
`end_silence_ms` whatever remains is force-finalized, bounding worst-case
latency. The audio window is bounded by unconditionally trimming to
`max_context_seconds` — deliberately not trimmed at precise committed-word
boundaries, since that needs per-token timestamps for a marginal-latency edge
case (see the module docstring for the full rationale).

### Rendering pipeline

`narrator.captions.buffer` (rolling caption buffer, what text is currently
"live") → `narrator.captions.wrap` (word-wrap) → `narrator.captions.layout`
(derives font size from canvas/margin/line-count/interline geometry — the
enforcement point for the "no explicit font_size" rule) → `narrator.render.canvas.CaptionRenderer`
(Pillow compositing: text, `background_band`/`shadow`/`stroke` style
overrides) → `narrator.ndi.sender` (`cyndilib`-backed NDI RGBA send,
transparent background).

### GUI (`narrator.gui`)

Tkinter + `sv-ttk` theme, optional (`gui.enabled` in config, or `--no-gui`).
`app.py` is the main window and Save-button config path (goes through the
same validate-then-swap path as hand edits); `preview.py` renders a live
canvas preview; `tray.py` is the system tray icon (`pystray`); `widgets.py`
holds reusable controls (e.g. opacity/color pickers). The GUI also supports a
manual caption mode (`AppController.set_caption_mode`) where an operator
types captions directly instead of ASR driving them — the ASR worker keeps
running underneath so switching back to auto is instant, but its events are
ignored while manual is active except STATUS (keeps the backend indicator
live).

## Repo layout

- `src/narrator/` — package source (see architecture above for the pipeline through `asr/` → `audio/` → `captions/` → `config/` → `gui/` → `ndi/` → `render/`).
- `tests/` — pytest unit tests (config validation, layout/wrap, streaming decoder, app controller).
- `scripts/render_preview.py` — headless rendering sanity check, no audio/GPU/NDI required.
- `packaging/pyinstaller/` — standalone binary build (spec + per-OS build scripts).
- `packaging/whisper_cpp/README.md` — building `pywhispercpp` with Vulkan (`GGML_VULKAN=1`) for real GPU acceleration.
- `assets/` — bundled font (Polish-capable) and Silero VAD ONNX model, referenced by `packaging/pyinstaller/narrator.spec` for the standalone build.

## Notes

- NDI SDK/runtime is a separate operator install (https://ndi.video), not bundled — check current NDI SDK redistribution terms before packaging the runtime library into a distributed build.
- Full design rationale: `/home/xxx/.claude/plans/i-want-you-to-shimmering-duckling.md`.
