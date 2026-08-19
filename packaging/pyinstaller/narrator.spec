# PyInstaller spec for narrator. Build with:
#   pyinstaller packaging/pyinstaller/narrator.spec --distpath dist --workpath build
#
# Run from the repo root so relative paths below resolve correctly.
from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = [
    ("assets/fonts/NotoSans-Regular.ttf", "assets/fonts"),
    ("assets/vad/silero_vad.onnx", "assets/vad"),
    ("assets/vad/LICENSE-silero-vad.txt", "assets/vad"),
    ("src/narrator/config/defaults.yaml", "narrator/config"),
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
    ["src/narrator/main.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["packaging/pyinstaller/hooks"],
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
