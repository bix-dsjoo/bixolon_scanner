param(
    [string]$Version = "0.1.17",
    [string]$FlutterExecutable = "C:/Users/OMEN/development/flutter/bin/flutter.bat",
    [string]$PythonExecutable = "C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe",
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$app = Join-Path $repositoryRoot "apps/bakery_scanner_lite"
$config = Get-Content -LiteralPath (Join-Path $repositoryRoot "configs/versions/$Version.json") -Raw | ConvertFrom-Json
if ([string]$config.version -ne $Version) { throw "Lite version config mismatch." }
$appBuild = [int]$config.app_build
$output = Join-Path $repositoryRoot "artifacts/lite/$Version"
$payload = Join-Path $output "payload"
if ((Test-Path -LiteralPath $payload) -and -not $Force) {
    throw "Lite payload exists; use -Force to rebuild only this Lite output."
}
Push-Location $app
try {
    & $FlutterExecutable build windows --release --build-name $Version --build-number $appBuild
    if ($LASTEXITCODE -ne 0) { throw "Lite Flutter build failed." }
}
finally { Pop-Location }

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repositoryRoot "src"
    & $PythonExecutable (Join-Path $PSScriptRoot "prepare_lite_payload.py") --repository-root $repositoryRoot --version $Version
    if ($LASTEXITCODE -ne 0) { throw "Lite payload verification failed." }
}
finally { $env:PYTHONPATH = $previousPythonPath }

$inno = @(
    (Join-Path $env:LOCALAPPDATA "Programs/Inno Setup 6/ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6/ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6/ISCC.exe")
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $inno) { throw "Inno Setup 6 is required." }
$redist = Join-Path $repositoryRoot "artifacts/installers/$Version/BixolonBakeryAIScanner-$Version-Worker/vc_redist.x64.exe"
if (-not (Test-Path -LiteralPath $redist)) { throw "Verified $Version VC runtime is missing." }
$expectedRedist = (Get-Content -LiteralPath (Join-Path $repositoryRoot "artifacts/installers/$Version/installer-manifest.json") -Raw | ConvertFrom-Json).vc_redist_sha256
if ((Get-FileHash -LiteralPath $redist -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedRedist) {
    throw "VC runtime checksum mismatch."
}
$icon = Join-Path $app "windows/runner/resources/app_icon.ico"
& $inno "/DAppVersion=$Version" "/DPayloadDir=$payload" "/DOutputDir=$output" "/DVcRedistPath=$redist" "/DSetupIconPath=$icon" (Join-Path $repositoryRoot "installer/windows/BixolonBakeryAIScannerLite.iss")
if ($LASTEXITCODE -ne 0) { throw "Lite setup build failed." }
$setup = Join-Path $output "BixolonBakeryAIScannerLite-$Version-Setup.exe"
$info = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($setup)
if ($info.ProductVersion.Trim() -ne $Version) { throw "Lite Setup version mismatch." }
$hash = (Get-FileHash -LiteralPath $setup -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText("$setup.sha256", "$hash  $([System.IO.Path]::GetFileName($setup))`n", [System.Text.UTF8Encoding]::new($false))
Write-Host "Lite Setup: $setup"
Write-Host "SHA-256: $hash"
