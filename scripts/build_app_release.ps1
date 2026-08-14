param(
    [string]$Composition = "configs/releases/bixolon_scanner_1.0.0.json",
    [string]$FlutterExecutable = "C:/Users/OMEN/development/flutter/bin/flutter.bat",
    [string]$PythonExecutable = "C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe",
    [string]$CudaRuntimeDirectory = $env:BIXOLON_CUDA_DLL_DIR,
    [string]$OutputRoot = "artifacts/releases"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$compositionPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Composition))
$compositionValue = Get-Content -Raw -LiteralPath $compositionPath | ConvertFrom-Json
$releaseName = [string]$compositionValue.release
$appVersion = [string]$compositionValue.versions.app
$workerVersion = [string]$compositionValue.versions.worker
$detectorVersion = [string]$compositionValue.versions.detector
$classifierVersion = [string]$compositionValue.versions.classifier
$datasetVersion = [string]$compositionValue.versions.dataset
$packagePath = [System.IO.Path]::GetFullPath(
    (Join-Path $repositoryRoot ([string]$compositionValue.model_package.path)))
$workerPath = Join-Path $repositoryRoot "artifacts/worker/bixolon-worker/bixolon-worker.exe"

if (-not (Test-Path -LiteralPath $packagePath -PathType Container)) {
    throw "Composed model package is missing: $packagePath"
}
if (-not (Test-Path -LiteralPath $workerPath -PathType Leaf)) {
    throw "Standalone Worker is missing: $workerPath"
}
if (-not $CudaRuntimeDirectory -or -not (Test-Path -LiteralPath $CudaRuntimeDirectory -PathType Container)) {
    throw "A complete CUDA runtime directory is required for a Release bundle."
}

& $PythonExecutable -m bixolon_scanner.operations.release_composition `
    --composition $compositionPath `
    --repository-root $repositoryRoot
if ($LASTEXITCODE -ne 0) {
    throw "Release composition verification failed."
}

$previousComposition = $env:SCANNER_RELEASE_COMPOSITION
$previousCuda = $env:BIXOLON_CUDA_DLL_DIR
try {
    $env:SCANNER_RELEASE_COMPOSITION = $compositionPath
    $env:BIXOLON_CUDA_DLL_DIR = [System.IO.Path]::GetFullPath($CudaRuntimeDirectory)
    Push-Location (Join-Path $repositoryRoot "apps/product_scanner")
    try {
        & $FlutterExecutable build windows --release `
            "--dart-define=BIXOLON_APP_VERSION=$appVersion" `
            "--dart-define=BIXOLON_WORKER_VERSION=$workerVersion" `
            "--dart-define=BIXOLON_DETECTOR_VERSION=$detectorVersion" `
            "--dart-define=BIXOLON_CLASSIFIER_VERSION=$classifierVersion" `
            "--dart-define=BIXOLON_DATASET_VERSION=$datasetVersion"
        if ($LASTEXITCODE -ne 0) {
            throw "Flutter Windows Release build failed."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    $env:SCANNER_RELEASE_COMPOSITION = $previousComposition
    $env:BIXOLON_CUDA_DLL_DIR = $previousCuda
}

$sourceBundle = Join-Path $repositoryRoot "apps/product_scanner/build/windows/x64/runner/Release"
$releaseRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$targetBundle = Join-Path $releaseRoot "bixolon-scanner-$appVersion"
if (Test-Path -LiteralPath $targetBundle) {
    throw "Release bundle already exists: $targetBundle"
}
[System.IO.Directory]::CreateDirectory($releaseRoot) | Out-Null
$temporaryBundle = Join-Path $releaseRoot ("." + [System.IO.Path]::GetRandomFileName())
[System.IO.Directory]::CreateDirectory($temporaryBundle) | Out-Null
try {
    Copy-Item -Path (Join-Path $sourceBundle "*") -Destination $temporaryBundle -Recurse -Force
    $files = Get-ChildItem -LiteralPath $temporaryBundle -Recurse -File |
        Sort-Object FullName |
        ForEach-Object {
            [ordered]@{
                path = [System.IO.Path]::GetRelativePath($temporaryBundle, $_.FullName).Replace("\", "/")
                size_bytes = $_.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
    $manifest = [ordered]@{
        schema_version = "1.0"
        release = $releaseName
        app_version = $appVersion
        composition_path = [System.IO.Path]::GetRelativePath($repositoryRoot, $compositionPath).Replace("\", "/")
        file_count = @($files).Count
        files = @($files)
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 `
        -LiteralPath (Join-Path $temporaryBundle "bundle-manifest.json")
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
Write-Host "Release bundle: $targetBundle"
Write-Host "Bundle manifest SHA-256: $bundleManifestHash"
