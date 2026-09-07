param([string]$PythonExecutable = "python")
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$buildRoot = Join-Path $repoRoot ("artifacts/camera-crop-2048/1.0.0/build-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
$inno = @(
    (Join-Path $env:LOCALAPPDATA "Programs/Inno Setup 6/ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6/ISCC.exe")
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $inno) { throw "Inno Setup 6 is required." }
$entry = Join-Path $repoRoot "src/bixolon_scanner/operations/camera_crop.py"
$arguments = @(
    "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed", "--noupx",
    "--name", "CameraCrop2048", "--paths", (Join-Path $repoRoot "src"),
    "--distpath", (Join-Path $buildRoot "dist"),
    "--workpath", (Join-Path $buildRoot "work"), "--specpath", $buildRoot,
    "--version-file", (Join-Path $PSScriptRoot "version_info.txt"),
    "--copy-metadata", "numpy", "--copy-metadata", "Pillow", "--copy-metadata", "opencv-python",
    "--collect-submodules", "numpy._core", "--collect-submodules", "numpy.linalg"
)
foreach ($excluded in @("torch", "torchvision", "tensorflow", "matplotlib", "scipy", "pandas", "IPython", "PyQt5", "PySide6", "pytest", "onnxruntime")) {
    $arguments += @("--exclude-module", $excluded)
}
$arguments += $entry
& $PythonExecutable @arguments 2>&1 | Tee-Object -FilePath (Join-Path $buildRoot "pyinstaller.log")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
$payload = Join-Path $buildRoot "dist/CameraCrop2048"
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "README.md") -Destination (Join-Path $payload "README-KO.md")
& $PythonExecutable (Join-Path $repoRoot "src/bixolon_scanner/operations/camera_crop_package.py") $payload
if ($LASTEXITCODE -ne 0) { throw "Payload manifest failed." }
$smoke = Join-Path $buildRoot "package-smoke"
$process = Start-Process -FilePath (Join-Path $payload "CameraCrop2048.exe") -ArgumentList @("--verify-package", ('"' + $smoke + '"')) -WindowStyle Hidden -Wait -PassThru
if ($process.ExitCode -ne 0) { throw "Packaged smoke failed. See $smoke" }
& $inno "/DPayloadDir=$payload" "/DOutputDir=$buildRoot" (Join-Path $PSScriptRoot "CameraCrop2048.iss")
if ($LASTEXITCODE -ne 0) { throw "Installer compilation failed." }
$setup = Join-Path $buildRoot "CameraCrop2048-1.0.0-Setup.exe"
$version = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($setup)
if ($version.FileVersion.Trim() -ne "1.0.0.0") { throw "Unexpected Setup version." }
$hash = (Get-FileHash -LiteralPath $setup -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText("$setup.sha256", "$hash  $([System.IO.Path]::GetFileName($setup))`n")
Write-Output "Setup: $setup"
Write-Output "Payload: $payload"
Write-Output "SHA-256: $hash"
