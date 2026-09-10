param(
    [string]$Version = '0.2.1',
    [string]$FlutterExecutable = 'C:/Users/OMEN/development/flutter/bin/flutter.bat',
    [string]$PythonExecutable = 'C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$config = Get-Content -LiteralPath (Join-Path $root "configs/versions/$Version.json") -Raw | ConvertFrom-Json
$output = Join-Path $root "artifacts/n100/$Version"
if (Test-Path -LiteralPath (Join-Path $output 'lite-payload')) { throw 'N100 Lite output exists; inspect before rebuilding.' }
if (-not (Test-Path -LiteralPath (Join-Path $root "artifacts/lite/$Version/payload/bakery_scanner_lite.exe"))) { throw 'Build the standard Lite payload before changing the Flutter compile profile.' }
Push-Location (Join-Path $root 'apps/bakery_scanner_lite')
try {
    & $FlutterExecutable build windows --release --build-name $Version --build-number ([string]$config.app_build) --dart-define=BIXOLON_EXECUTION_PROFILE=n100
    if ($LASTEXITCODE -ne 0) { throw 'N100 Flutter build failed.' }
} finally { Pop-Location }
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $root 'src'
    & $PythonExecutable -m bixolon_scanner.operations.n100_bundle --repository-root $root --version $Version --lite
    if ($LASTEXITCODE -ne 0) { throw 'N100 payload verification failed.' }
} finally { $env:PYTHONPATH = $previousPythonPath }
$redist = Join-Path $root "artifacts/installers/$Version/BixolonBakeryAIScanner-$Version-Worker/vc_redist.x64.exe"
$expected = (Get-Content -LiteralPath (Join-Path $root "artifacts/installers/$Version/installer-manifest.json") -Raw | ConvertFrom-Json).vc_redist_sha256
if ((Get-FileHash -LiteralPath $redist -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'VC runtime checksum mismatch.' }
$inno = Join-Path $env:LOCALAPPDATA 'Programs/Inno Setup 6/ISCC.exe'
& $inno "/DAppVersion=$Version" "/DPayloadDir=$output/lite-payload" "/DOutputDir=$output" "/DVcRedistPath=$redist" "/DSetupIconPath=$root/apps/bakery_scanner_lite/windows/runner/resources/app_icon.ico" '/DN100Profile=1' (Join-Path $root 'installer/windows/BixolonBakeryAIScannerLite.iss')
if ($LASTEXITCODE -ne 0) { throw 'N100 Setup build failed.' }
$setup = Join-Path $output "BixolonBakeryAIScannerLite-$Version-N100-Setup.exe"
$info = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($setup)
if ($info.ProductVersion.Trim() -ne $Version) { throw 'N100 Setup version mismatch.' }
$hash = (Get-FileHash -LiteralPath $setup -Algorithm SHA256).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText("$setup.sha256", "$hash  $([System.IO.Path]::GetFileName($setup))`n", [System.Text.UTF8Encoding]::new($false))
Write-Host "N100 Setup: $setup"
Write-Host "SHA-256: $hash"
