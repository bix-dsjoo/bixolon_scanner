param(
    [string]$Version = "0.0.1",
    [string]$FlutterExecutable = "C:/Users/OMEN/development/flutter/bin/flutter.bat",
    [string]$PythonExecutable = "C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe"
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $repositoryRoot "configs/versions/$Version.json"
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Version config is missing: $configPath"
}
$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
if ([string]$config.version -ne $Version) {
    throw "Version config identity mismatch: $configPath"
}
$appBuild = [int]$config.app_build
$sourceDateEpoch = [long]$config.source_date_epoch
$versionRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $repositoryRoot ([string]$config.output_root + "/" + $Version)))
$workerOutput = "artifacts/versions/$Version/worker-build"
$sourceDirectory = Join-Path $repositoryRoot "src"

$previousPythonPath = $env:PYTHONPATH
$previousVersionRoot = $env:SCANNER_VERSION_ROOT
try {
    $env:PYTHONPATH = $sourceDirectory
    & $PythonExecutable -m bixolon_scanner.operations.version_bundle prepare `
        --config $configPath `
        --repository-root $repositoryRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Version preparation failed with exit code $LASTEXITCODE."
    }

    & (Join-Path $PSScriptRoot "build_worker.ps1") `
        -PythonExecutable $PythonExecutable `
        -OutputDirectory $workerOutput `
        -SourceDateEpoch $sourceDateEpoch
    if ($LASTEXITCODE -ne 0) {
        throw "Worker build failed with exit code $LASTEXITCODE."
    }

    & $PythonExecutable -m bixolon_scanner.operations.version_bundle verify `
        --config $configPath `
        --repository-root $repositoryRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Prepared version verification failed with exit code $LASTEXITCODE."
    }

    $env:SCANNER_VERSION_ROOT = $versionRoot
    Push-Location (Join-Path $repositoryRoot "apps/product_scanner")
    try {
        & $FlutterExecutable build windows --release `
            --build-name $Version `
            --build-number $appBuild `
            "--dart-define=BIXOLON_VERSION=$Version"
        if ($LASTEXITCODE -ne 0) {
            throw "Flutter Windows build failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:SCANNER_VERSION_ROOT = $previousVersionRoot
}

$sourceBundle = Join-Path $repositoryRoot "apps/product_scanner/build/windows/x64/runner/Release"
$targetBundle = Join-Path $versionRoot "bixolon-scanner-$Version"
if (Test-Path -LiteralPath $targetBundle) {
    throw "Version bundle already exists: $targetBundle"
}
[System.IO.Directory]::CreateDirectory($versionRoot) | Out-Null
$temporaryBundle = Join-Path $versionRoot ("." + [System.IO.Path]::GetRandomFileName())
[System.IO.Directory]::CreateDirectory($temporaryBundle) | Out-Null
try {
    Copy-Item -Path (Join-Path $sourceBundle "*") -Destination $temporaryBundle -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $versionRoot "staging/version.json") `
        -Destination $temporaryBundle
    Copy-Item -LiteralPath (Join-Path $versionRoot "staging/provenance.json") `
        -Destination $temporaryBundle

    $productExecutable = Join-Path $temporaryBundle "product_scanner.exe"
    $productVersion = [System.Diagnostics.FileVersionInfo]::GetVersionInfo(
        $productExecutable).ProductVersion
    if ($productVersion -ne $Version) {
        throw "Windows ProductVersion does not match ${Version}: $productVersion"
    }

    $manifestPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $sourceDirectory
        & $PythonExecutable -m bixolon_scanner.operations.version_bundle manifest `
            --config $configPath `
            --repository-root $repositoryRoot `
            --bundle $temporaryBundle
        if ($LASTEXITCODE -ne 0) {
            throw "Bundle manifest generation failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        $env:PYTHONPATH = $manifestPythonPath
    }
    [System.IO.Directory]::Move($temporaryBundle, $targetBundle)
}
catch {
    if (Test-Path -LiteralPath $temporaryBundle) {
        Remove-Item -LiteralPath $temporaryBundle -Recurse -Force
    }
    throw
}

$bundleManifest = Join-Path $targetBundle "bundle-manifest.json"
$bundleManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundleManifest).Hash.ToLowerInvariant()
Write-Host "Version bundle: $targetBundle"
Write-Host "Bundle manifest SHA-256: $bundleManifestHash"
