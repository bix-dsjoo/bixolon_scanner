param(
    [string]$Version = '0.2.0',
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
