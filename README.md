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
- **NDI SDK/Runtime**: required to actually send an NDI stream. Download
  from https://ndi.video (free). This is treated as a separate operator
  setup step, not bundled -- confirm current NDI SDK redistribution terms
  before packaging the runtime library into a distributed build.
- **Ubuntu**: `libportaudio2` for microphone capture
  (`sudo apt install libportaudio2`). For GPU acceleration, a working
  Vulkan driver (`mesa-vulkan-drivers` for AMD/Intel, or the NVIDIA driver's
  bundled Vulkan support) -- CPU fallback works with no extra packages.
- **Windows**: microphone capture and Vulkan (via GPU vendor drivers) work
  out of the box; no extra system packages needed.
- **No GPU build step is required.** The published `pywhispercpp` wheel
  already ships the Vulkan ggml backend, so a plain `pip install -e .` gets
  GPU acceleration (verified on a Radeon RX 6700 XT via RADV). You need the
  Vulkan loader and driver ICD only -- `libvulkan1` plus
  `mesa-vulkan-drivers` on AMD/Intel -- and *not* the `vulkan-tools`
  diagnostic package. `packaging/whisper_cpp/README.md` covers source
  builds, now only needed for a non-Vulkan ggml backend.

## Setup (development)

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\Activate.ps1 on Windows
pip install -e .
narrator --list-devices           # sanity-check microphone enumeration
narrator                          # runs with default config, opens the GUI unless gui.enabled: false
```

On first run, a default config is copied to the platform's user config
directory (e.g. `~/.config/narrator/config.yaml` on Linux,
`%APPDATA%\narrator\config.yaml` on Windows). Edit it by hand or through the
GUI's Save button -- both paths go through the same validation and hot-reload
(the running app picks up changes without a restart, except for canvas
size/NDI source name and audio device, which need a restart).

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
  available there) and the Tkinter GUI (no `tkinter`/display in that
  sandbox).

  Note: an earlier revision of this list claimed prebuilt whisper.cpp
  requires AVX2. It does not -- the shipped libraries contain no AVX/AVX2/FMA
  instructions at all and run fine on a pre-AVX Pentium G4560. Real
  whisper.cpp inference and Vulkan GPU acceleration have since been verified
  end-to-end on that machine against `test.mp3`.
- The ASR backend's Python API usage (`pywhispercpp.model.Model`,
  `context_params={"use_gpu": ...}`, model auto-download) and the NDI
  bindings (`cyndilib.Sender`/`VideoSendFrame`, `FourCC.RGBA`) were both
  confirmed against the actual installed packages, not guessed.
