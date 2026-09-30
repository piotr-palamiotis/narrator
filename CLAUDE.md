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

# ASR stress test (Piper TTS -> PipeWire virtual mic -> full app -> WER/latency)
python scripts/gen_test_clips.py                    # byte-deterministic Polish clips + manifest
python scripts/stress_full_pipeline.py --model medium --step-ms 350 --repeats 3 --passes 2

# pick the best ASR model/device/step_ms for this machine and persist to config
narrator --autoconfigure                            # needs scripts/test_clips/ fixtures for the WER probe

# standalone binary
./packaging/pyinstaller/build_linux.sh
.\packaging\pyinstaller\build_windows.ps1
```

There is no configured linter/formatter in this repo — don't assume one.

### GPU (Vulkan)

No build step is needed. The published `pywhispercpp` wheel (1.5.1) already
ships `libggml-vulkan.so`, so `pip install -e .` gives working GPU
acceleration; verified on a Radeon RX 6700 XT via RADV. The runtime needs
only the Vulkan loader + driver ICD (`libvulkan1`, `mesa-vulkan-drivers` on
Ubuntu/AMD) — **not** the `vulkan-tools` diagnostic package, and not the
Vulkan SDK.

Those wheels are also built for baseline x86-64 (zero AVX/AVX2/FMA
instructions in any shipped `.so`), so they run on pre-AVX CPUs — verified
on a Pentium G4560 (SSE4.2 only), on both the GPU and CPU-fallback paths.
An old CPU is not a reason to rebuild anything.

`packaging/whisper_cpp/README.md` covers source builds, which are now only
relevant for a non-Vulkan ggml backend.

## Sandbox limitations (relevant when asked to "verify" something)

This environment has no microphone, display, or NDI receiver. It *does* have
a working AMD GPU (Vulkan/RADV).
`pytest tests/` and `scripts/render_preview.py` work fully here. Live mic
capture (`sounddevice`/PortAudio) and the Tkinter GUI cannot be exercised
end-to-end — say so rather than claiming they were tested.

Whisper inference and Vulkan GPU acceleration *are* exercisable on this
machine (RX 6700 XT + Pentium G4560). `test.mp3` at the repo root is the
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
- **Feed thread**: drains `MicCapture`'s queue into the ASR worker's
  `multiprocessing.Queue`.
- **ASR worker**: a *separate OS process* (`narrator.asr.worker.run_asr_worker`),
  not a thread — isolates whisper.cpp native-code crashes / stalled GPU calls
  from rendering, NDI output, and the GUI. Launched with the `spawn` start
  method (forced in `AppController.__init__`): forking from the multi-threaded
  app SIGSEGVs the child inside ggml's Vulkan init on any post-startup ASR
  restart (measured: exit -11 within ~2s, every time; the first spawn from a
  just-started app happened to work, which masked this for a long time).
  `AppController._supervise_asr_loop` watches it and restarts on death with
  backoff; after `_QUICK_CRASHES_BEFORE_EXIT` (3) crashes within
  `_QUICK_CRASH_WINDOW_SECONDS` (15s) of spawn, it stops the app with an
  explanation instead of falling back to CPU — **the configured device is
  final by policy**: no code path downgrades GPU→CPU silently (the old
  init-fallback, the transcribe-time fallback, and the crash-loop CPU force
  all existed and are gone). Restart drains stale audio frames so the next
  generation doesn't caption pre-restart speech, and the worker releases the
  whisper model deterministically in a `finally` block.
- **Event-drain thread**: applies `CaptionEvent`s (PARTIAL/FINAL/CLEAR/WIPE/STATUS)
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

**Language handling** (`asr.language_whitelist`, the single language control —
edited via the GUI's dropdown+Add / list+Remove editor on the Audio/ASR tab):
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
language` pads its mel to a fixed 30s window, so it is ~445ms on large-v3
**regardless of slice length** (tiny 28ms / small 82ms / medium 245ms; mel
itself ~5ms). A smaller detection model is NOT viable: small/medium confuse
Polish with Russian/Ukrainian on mid-clip 2s slices (7/12 and 8/12 vs
large-v3's 12/12). Note: whisper transcribes; it has no X→Polish
translation. This replaced an older three-mode design (forced
`asr.language` + `language_switch_threshold` streak + "auto"), removed in
favor of the margin-gated whitelist — one knob, per-utterance, no streak
delay. `"auto"` instead follows the speaker verbatim per
utterance: pywhispercpp 1.5.1's `detect_language=True` transcribe param is
broken (always returns zero segments — verified), so `WhisperBackend` uses
the dedicated `auto_detect_language()` API; the StreamingDecoder locks one
language per utterance, gated on BOTH ≥1.5s of audio and ≥0.5 confidence
(0.4s slices misdetect 1-in-7 on large-v3), skipping decode steps before the
lock. All detection calls use `auto_detect_language`; there is no allow-list
anywhere. Large-v3's detector can still drift to a related language on the
live path (measured: Cyrillic captions for Polish), while medium/small do
not — a model-behavior limit, not a policy knob. True X→Polish translation is
impossible with whisper (its only translation target is English).

Most fields hot-reload without a restart. Exceptions handled explicitly in
`_on_config_changed`: an `ndi`/`canvas` change tears down and restarts just
the NDI stream (`_restart_ndi_stream`, its own stop event separate from the
app-wide one); an `asr` change restarts the ASR worker process; an `audio`
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

### Streaming ASR decode (`narrator.asr.streaming_decoder.StreamingDecoder`)

Implements the "whisper_streaming" local-agreement pattern: whisper.cpp is
re-run on a growing audio window every `step_ms`; the new hypothesis is
diffed word-by-word against the last `agreement_passes` hypotheses, and a
prefix is only committed (emitted as a FINAL `CaptionEvent`) once it's been
stable across all of them. This trades latency (bounded by
`step_ms * agreement_passes`) for captions that don't visibly backtrack. VAD
(`narrator.asr.vad.SileroVad`, ONNX) detects speech/silence; on silence past
`end_silence_ms` whatever remains is force-finalized, bounding worst-case
latency.**Ingest is deliberately decoupled from inference.** A decode step costs far
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
A cross-commit guard (`_drop_repeated_boundary`) additionally drops a new
FINAL's first word when it duplicates the previous FINAL's last word (the
same artifact straddling two commits). Tradeoff: a speaker's emphatic repeat
("tak, tak") becomes "tak" — right side of the trade for a lower-third.

Two whisper parameters in `backend.py` exist purely to bound decode latency
and should not be removed casually: `temperature_inc=0.0` disables whisper's
temperature-fallback ladder (which re-decodes a low-confidence window up to
six times — observed turning a 1.6s window into a **17-second** stall), and
`audio_ctx` shrinks the encoder's fixed 30s context to the actual window
length (~3x faster on short windows). The audio window is bounded by unconditionally trimming to
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

**Caption retention (do not regress this)**: the ASR worker's silence
timers must never blank what the audience is currently reading.
`prune_history_after_silence_seconds` (renamed from
`clear_after_silence_seconds`; a `model_validator(mode="before")`
migration still accepts the old key) emits CLEAR, which drops only lines
already invisible under `layout.max_lines` (`CaptionBuffer.prune_history`)
— the visible lower-third is byte-identical after. A true screen clear is
operator-initiated (Ctrl+N / the GUI Clear button) or comes as WIPE after
`wipe_screen_after_silence_seconds` of silence (default 0 = off: stale
captions are usually better than blank ones; raise it when very old text
would misattribute to new speech). In manual caption mode both events are
ignored like any other ASR output.

### GUI (`narrator.gui`)

Tkinter + `sv-ttk` theme, optional (`gui.enabled` in config, or `--no-gui`).
`app.py` is the main window and Save-button config path (goes through the
same validate-then-swap path as hand edits); `preview.py` renders a live
canvas preview; `tray.py` is the system tray icon (`pystray`); `widgets.py`
holds reusable controls (e.g. opacity/color pickers, the mic level meter)
plus the `CONTROL_WIDTH` standard that lines up every entry-style control.
Ctrl+N (bound at the root-window level, so it works regardless of which
widget has focus, including the manual caption entry) clears the current
captions at any time. The Audio/ASR tab carries two live indicators, both
polled on the Tk thread: the mic input-level meter reads `MicCapture.level`
— a plain float published from the PortAudio callback thread (the single-
float write is atomic under the GIL, so polling neither locks nor stalls
the callback; zeroed on `stop()`; scaled x^(1/3) with no decay/peak-hold by
design), and it re-reads `controller._mic` fresh each tick because
`_restart_audio_capture` swaps the MicCapture instance — don't cache it —
and the ASR worker's Active/Inactive label reflects *process liveness*
(`_asr_process.is_alive()`), not the STATUS event string (STATUS still
updates `resolved_asr_backend` on the controller). The Whisper model field
is a dropdown of pywhispercpp's `AVAILABLE_MODELS` plus locally cached
models, kept editable so a custom name or direct ggml path in config isn't
lost on autosave. The GUI also supports a
manual caption mode (`AppController.set_caption_mode`) where an operator
types captions directly instead of ASR driving them — the ASR worker keeps
running underneath so switching back to auto is instant, but its events are
ignored while manual is active except STATUS.

## Repo layout

- `src/narrator/` — package source (see architecture above for the pipeline through `asr/` → `audio/` → `captions/` → `config/` → `gui/` → `ndi/` → `render/`).
- `tests/` — pytest unit tests (config validation, layout/wrap, streaming decoder, app controller).
- `scripts/render_preview.py` — headless rendering sanity check, no audio/GPU/NDI required.
- `scripts/live_language_demo.py` — plays a pl→en→pl schedule through the virtual mic into the real app and prints the observed caption timeline vs expectations (dynamic language switching end-to-end).
- `scripts/gen_test_clips.py` — generates byte-deterministic Polish TTS fixtures (piper,
  zero noise scales) with known transcripts into `scripts/test_clips/`.
- `scripts/stress_full_pipeline.py` — full-pipeline ASR stress harness: plays the fixtures
  through a PipeWire virtual mic into a real `AppController`, records FINAL events, and
  reports per-clip WER, finalize latency, and cross-pass determinism. Baselines on this
  repo's hardware (RX 6700 XT): `small`+step 600 ≈ 7% WER / 0.8s, `medium`+step 350 ≈
  7–10% WER / 0.5–0.8s; `large-v3` used to collapse under streaming (repetition loops,
  multi-second decode steps) despite being clean offline — beam search was tested
  as a loop mitigation and is unusable in pywhispercpp 1.5.1 (GPU-path 'patience'
  error → silent CPU fallback → 104s decode steps). **Fixed at the text layer**
  (streaming_decoder.py): large-v3's dominant failure was whole-sentence repetition
  loops (same 6–9 word sentence 5–6× inside one hypothesis), and `_collapse_repetitions`
  now runs to phrase length 16, collapsing those loops — harness WER 1.28 → 0.14 mean
  (mixed 11.7 → 0.9, numbers → 0.00). whisper.cpp has no no_repeat_ngram_size (that's
  faster-whisper) and `entropy_thold`/`suppress_regex` don't break in-decode loops, so
  this collapse is the enforcement point. The default model remains large-v3 at the
  operator's explicit choice.
- `narrator/asr/autoconfigure.py` (+ `bench.py`, `quality_probe.py`, `metrics.py`) —
  `narrator --autoconfigure` benchmarks candidates in throwaway subprocesses (speed per
  window size + WER of the *streaming* decoder on the fixtures) and writes the winner's
  model/device/step_ms to config.yaml. Qualification gates: decode slower than 1.5x real
  time at any window, streaming WER > 0.30, or doubled words in FINAL text disqualify.
  The last gate guards the sentence-loop collapse fix: `doubled=True` means the text-layer
  fix is absent/broken, so the candidate is unrankable. Post-collapse-fix, large-v3
  streams cleanly (probe: WER 0.145, doubled False; raw decode 0.06x realtime at a 10s
  window) and qualifies; on capable hardware all models cap the speed term, so measured
  accuracy decides. The quality probe must stay streaming-shaped (real-time feed + tick) —
  offline transcription would miss exactly the failure modes these gates guard.
- `packaging/pyinstaller/` — standalone binary build (spec + per-OS build scripts).
- `packaging/whisper_cpp/README.md` — GPU/Vulkan runtime requirements, plus source-build instructions for the rare case a non-Vulkan ggml backend is needed.
- `assets/` — bundled font (Polish-capable) and Silero VAD ONNX model, referenced by `packaging/pyinstaller/narrator.spec` for the standalone build.

## Notes

- NDI SDK/runtime is a separate operator install (https://ndi.video), not bundled — check current NDI SDK redistribution terms before packaging the runtime library into a distributed build.
- Full design rationale: `/home/xxx/.claude/plans/i-want-you-to-shimmering-duckling.md`.
