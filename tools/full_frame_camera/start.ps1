param([string]$OutputDirectory = "", [int]$Camera = 0)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$pythonExe = (Get-Command python -ErrorAction Stop).Source
& $pythonExe -c "import cv2, PIL, tkinter"
if ($LASTEXITCODE -ne 0) {
    throw "Python dependencies missing. Run: python -m pip install -r tools/full_frame_camera/requirements.txt"
}
$appScript = Join-Path $repoRoot "src/bixolon_scanner/operations/camera_capture.py"
$launchArgs = @($appScript, "--camera", "$Camera")
if ($OutputDirectory) { $launchArgs += @("--output", $OutputDirectory) }
& $pythonExe @launchArgs
