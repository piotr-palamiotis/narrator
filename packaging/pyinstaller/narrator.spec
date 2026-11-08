# PyInstaller spec for narrator. Build with (from anywhere):
#   pyinstaller packaging/pyinstaller/narrator.spec --distpath dist --workpath build
#
# Paths inside a spec are resolved against the spec file's own directory
# (SPECPATH), not the invocation cwd, so everything is anchored to the repo
# root explicitly to keep the build working regardless of where it's run.
import os

from PyInstaller.utils.hooks import collect_all

block_cipher = None

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir, os.pardir))


def _root(*parts):
    return os.path.join(ROOT, *parts)


datas = [
    (_root("assets", "fonts", "NotoSans-Regular.ttf"), "assets/fonts"),
    (_root("assets", "vad", "silero_vad.onnx"), "assets/vad"),
    (_root("assets", "vad", "LICENSE-silero-vad.txt"), "assets/vad"),
    (_root("src", "narrator", "config", "defaults.yaml"), "narrator/config"),
    # asr.preset is resolved from presets.yaml whenever a config is loaded.
    (_root("src", "narrator", "config", "presets.yaml"), "narrator/config"),
    (_root("src", "narrator", "captions", "data", "blocklist_starter.txt"), "narrator/captions/data"),
    # --pick-preset's bundled Piper clip + reference words (asr/preset_picker.py DATA_DIR).
    (_root("src", "narrator", "asr", "data"), "narrator/asr/data"),
    # The web admin UI's static frontend (plain HTML/JS/CSS, no build step).
    (_root("src", "narrator", "web", "static"), "narrator/web/static"),
]
binaries = []
# Modules reached only by name or via `python -m` / lazy imports, which the
# static analysis can miss. narrator.health.vulkan_probe: the frozen app
# enumerates in-process instead of spawning it (health/checks.py);
# narrator.asr.preset_probe / recording.draft: subprocess entry points
# (draft via --draft-recording; preset_probe is NOT reachable when frozen).
hiddenimports = [
    "narrator.health.checks",
    "narrator.health.monitor",
    "narrator.health.problems",
    "narrator.health.usage",
    "narrator.health.vulkan_probe",
    "narrator.asr.vulkan_devices",
    "narrator.asr.units",
    "narrator.asr.catalog",
    "narrator.asr.asr_backend",
    "narrator.asr.decoder_common",
    "narrator.asr.segmenter",
    "narrator.asr.streaming_decoder",
    "narrator.asr.text_settings",
    "narrator.asr.offline",
    "narrator.asr.preset_picker",
    "narrator.asr.preset_probe",
    "narrator.recording.draft",
    "narrator.web.osc",
]

# These packages ship native extensions / data files (whisper.cpp bindings,
# NDI bindings, ONNX Runtime, PortAudio) that PyInstaller's default import
# analysis won't fully discover on its own -- collect_all pulls in their
# binaries/data/submodules explicitly rather than hand-listing exact .so
# paths, which would be fragile across platforms and package versions.
# soundfile: libsndfile (FLAC) for session recording, bundled in its wheel.
for package in ("pywhispercpp", "cyndilib", "onnxruntime", "sounddevice", "soundfile"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    [_root("src", "narrator", "main.py")],
    pathex=[_root("src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[_root("packaging", "pyinstaller", "hooks")],
    runtime_hooks=[],
    # The offline tuning tool (narrator-tune) is not part of the app; narrator
    # never imports it, and this keeps it out even if something drags it in.
    # The Tk GUI is gone (replaced by the web UI); keep Tk out of the bundle.
    excludes=["narrator_tune", "tkinter"],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="narrator",
    console=True,
    onefile=True,
)
