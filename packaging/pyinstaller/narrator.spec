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
]
binaries = []
hiddenimports = []

# These packages ship native extensions / data files (whisper.cpp bindings,
# NDI bindings, ONNX Runtime, PortAudio) that PyInstaller's default import
# analysis won't fully discover on its own -- collect_all pulls in their
# binaries/data/submodules explicitly rather than hand-listing exact .so
# paths, which would be fragile across platforms and package versions.
for package in ("pywhispercpp", "cyndilib", "onnxruntime", "sounddevice"):
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
    excludes=[],
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
