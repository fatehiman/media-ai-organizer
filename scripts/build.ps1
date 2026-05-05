# =============================================================================
#  Build Media Organizer (portable Windows folder)
# -----------------------------------------------------------------------------
#  Produces: dist\MediaOrganizer\
#       media-organizer.exe          (entry point)
#       _internal\                   (PyInstaller deps)
#       media-organizer.conf         (default config)
#       models\                      (ONNX + class names)
#       third_party\
#           tesseract\               (you provide the binary + tessdata)
#           ffmpeg\                  (you provide the binaries)
#       README.MD
#
#  Pre-requisites:
#       1. Activate your venv:        .\.venv\Scripts\Activate.ps1
#       2. pip install -r requirements.txt
#       3. python scripts\bootstrap_models.py            (one-time, online)
#       4. Copy portable Tesseract  -> third_party\tesseract\tesseract.exe
#                              + tessdata\eng.traineddata
#       5. Copy portable ffmpeg     -> third_party\ffmpeg\bin\ffmpeg.exe
#                                                          ffprobe.exe
#       6. .\scripts\build.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Repo root: $repoRoot"

# ---- preflight ---------------------------------------------------------------

$required = @(
    "models\mobilenetv3.onnx",
    "models\imagenet_classes.json",
    "models\silero_vad.onnx",
    "third_party\tesseract\tesseract.exe"
)
$optional = @(
    "third_party\ffmpeg\bin\ffmpeg.exe",
    "third_party\ffmpeg\bin\ffprobe.exe"
)
$missing = @()
foreach ($r in $required) {
    if (-not (Test-Path (Join-Path $repoRoot $r))) {
        $missing += $r
    }
}
if ($missing.Count -gt 0) {
    Write-Host "Missing required files:" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "    $m" -ForegroundColor Red }
    Write-Host "See header of build.ps1 for how to populate them." -ForegroundColor Yellow
    exit 1
}
foreach ($o in $optional) {
    if (-not (Test-Path (Join-Path $repoRoot $o))) {
        Write-Host "Optional asset missing: $o (cv2 will be used for video; that's fine)" -ForegroundColor Yellow
    }
}

# ---- clean prior build -------------------------------------------------------

if (Test-Path build)  { Remove-Item -Recurse -Force build }
if (Test-Path dist)   { Remove-Item -Recurse -Force dist }
Get-ChildItem -Filter "*.spec" | Remove-Item -Force -ErrorAction SilentlyContinue

# ---- run PyInstaller (from the venv, not from the system Python) ------------

$entry = "src\run_media_organizer.py"
$venvPyInst = Join-Path $repoRoot ".venv\Scripts\pyinstaller.exe"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPyInst)) {
    Write-Host "`nVenv pyinstaller not found at $venvPyInst." -ForegroundColor Red
    Write-Host "Run:  .\.venv\Scripts\python.exe -m pip install pyinstaller" -ForegroundColor Yellow
    exit 1
}
Write-Host "`nRunning PyInstaller from venv..." -ForegroundColor Cyan

& $venvPyInst `
    --name media-organizer `
    --onedir `
    --noconfirm `
    --clean `
    --console `
    --paths src `
    --collect-submodules librosa `
    --collect-submodules onnxruntime `
    --collect-data librosa `
    --collect-data soundfile `
    --collect-data pillow_heif `
    --collect-binaries onnxruntime `
    --collect-data onnxruntime `
    --hidden-import onnxruntime `
    --hidden-import onnxruntime.capi._pybind_state `
    --hidden-import pytesseract `
    --hidden-import pillow_heif `
    --hidden-import cv2 `
    $entry

if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

$distDir = Join-Path $repoRoot "dist\media-organizer"

# Rename folder to the final product name.
$finalDir = Join-Path $repoRoot "dist\MediaOrganizer"
if (Test-Path $finalDir) { Remove-Item -Recurse -Force $finalDir }
Rename-Item -Path $distDir -NewName "MediaOrganizer"

# ---- copy resources next to the exe ------------------------------------------

Write-Host "`nCopying resources..." -ForegroundColor Cyan
Copy-Item -Path "media-organizer.conf"   -Destination $finalDir
Copy-Item -Path "README.MD"              -Destination $finalDir -ErrorAction SilentlyContinue
Copy-Item -Recurse -Path "models"        -Destination $finalDir
Copy-Item -Recurse -Path "third_party"   -Destination $finalDir

Write-Host "`nBuild complete." -ForegroundColor Green
Write-Host "Output: $finalDir"
Write-Host "Run:    $finalDir\media-organizer.exe"
