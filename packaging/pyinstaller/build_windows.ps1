# Build a single-file Windows .exe. Run from the repo root in PowerShell:
#   .\packaging\pyinstaller\build_windows.ps1
#
# Prerequisites:
#   - A whisper.cpp build with Vulkan enabled has been installed such that
#     `import pywhispercpp` loads the Vulkan-capable whisper.dll (see
#     packaging/whisper_cpp/build_vulkan_windows.ps1). The plain PyPI wheel
#     is CPU-only and will NOT provide real GPU acceleration.
#   - The NDI SDK/Runtime redistributable is a separate operator install;
#     see README.md for the current licensing/bundling status.
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\.."

python -m pip install --upgrade pyinstaller
pyinstaller packaging\pyinstaller\narrator.spec --distpath dist --workpath build --noconfirm

Write-Host "Built: dist\narrator.exe"
