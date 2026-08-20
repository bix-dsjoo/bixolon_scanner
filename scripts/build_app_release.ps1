param(
    [string]$Composition = "configs/releases/scanner_2.0.1.json",
    [string]$FlutterExecutable = "C:/Users/OMEN/development/flutter/bin/flutter.bat",
    [string]$PythonExecutable = "C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe",
    [string]$CudaRuntimeDirectory = $env:BIXOLON_CUDA_DLL_DIR,
    [string]$OutputRoot = "artifacts/releases"
)

$ErrorActionPreference = "Stop"

function Get-RelativeReleasePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $baseFullPath = [System.IO.Path]::GetFullPath($BasePath).TrimEnd("\") + "\"
    $targetFullPath = [System.IO.Path]::GetFullPath($TargetPath)
    if (-not $targetFullPath.StartsWith(
            $baseFullPath,
            [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Release path is outside its expected root: $targetFullPath"
    }
    return $targetFullPath.Substring($baseFullPath.Length).Replace("\", "/")
}

function Assert-AttestedDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Directory,
        [Parameter(Mandatory = $true)]
        [object]$Artifact
    )

    $root = [System.IO.Path]::GetFullPath($Directory).TrimEnd("\") + "\"
    $actualFiles = @(Get-ChildItem -LiteralPath $Directory -Recurse -File)
    if ($actualFiles.Count -ne [int]$Artifact.file_count) {
        throw "Attested directory file count mismatch: $Directory"
    }
    foreach ($row in $Artifact.files) {
        $path = [System.IO.Path]::GetFullPath((Join-Path $Directory ([string]$row.path)))
        if (-not $path.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase) -or
            -not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Attested file is missing or escapes its directory: $($row.path)"
        }
        $file = Get-Item -LiteralPath $path
        $sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($file.Length -ne [long]$row.size_bytes -or $sha256 -ne [string]$row.sha256) {
            throw "Attested file checksum mismatch: $($row.path)"
        }
    }
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$compositionPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Composition))
$compositionValue = Get-Content -Raw -LiteralPath $compositionPath | ConvertFrom-Json
$releaseName = [string]$compositionValue.release
$appVersion = [string]$compositionValue.versions.app
$workerVersion = [string]$compositionValue.versions.worker
$detectorVersion = [string]$compositionValue.versions.detector
$classifierVersion = [string]$compositionValue.versions.classifier
$datasetVersion = [string]$compositionValue.versions.dataset
$isScannerV2 = $null -ne $compositionValue.production_release
if ($isScannerV2) {
    $packagePath = [System.IO.Path]::GetFullPath(
        (Join-Path $repositoryRoot ([string]$compositionValue.production_release.runtime_path)))
    $catalogPath = [System.IO.Path]::GetFullPath(
        (Join-Path $repositoryRoot ([string]$compositionValue.production_release.catalog_path)))
    $workerRuntimePath = [System.IO.Path]::GetFullPath(
        (Join-Path $repositoryRoot ([string]$compositionValue.production_release.worker_bundle_path)))
    $workerPath = Join-Path $workerRuntimePath "bixolon-worker.exe"
}
else {
    $packagePath = [System.IO.Path]::GetFullPath(
        (Join-Path $repositoryRoot ([string]$compositionValue.model_package.path)))
    $workerPath = Join-Path $repositoryRoot "artifacts/worker/bixolon-worker/bixolon-worker.exe"
}

if (-not (Test-Path -LiteralPath $packagePath -PathType Container)) {
    throw "Composed model package is missing: $packagePath"
}
if (-not (Test-Path -LiteralPath $workerPath -PathType Leaf)) {
    throw "Standalone Worker is missing: $workerPath"
}
if (-not $CudaRuntimeDirectory -or -not (Test-Path -LiteralPath $CudaRuntimeDirectory -PathType Container)) {
    throw "A complete CUDA runtime directory is required for a Release bundle."
}

if ($isScannerV2) {
    if (-not (Test-Path -LiteralPath $catalogPath -PathType Container)) {
        throw "Composed Store Catalog is missing: $catalogPath"
    }
    $attestationPath = [System.IO.Path]::GetFullPath(
        (Join-Path $repositoryRoot ([string]$compositionValue.production_release.attestation_path)))
    $attestation = Get-Content -Raw -LiteralPath $attestationPath | ConvertFrom-Json
    if ($attestation.release -ne "2.0.1" -or
        $attestation.status -ne "production" -or
        $attestation.production_eligible -ne $true -or
        $attestation.independent_certified -ne $false -or
        $attestation.catalog_authentication -ne "CHECKSUM-SHA256" -or
        $attestation.attestation_sha256 -ne
            $compositionValue.production_release.attestation_sha256) {
        throw "Scanner 2.0.1 production attestation does not match the composition."
    }
    Assert-AttestedDirectory -Directory $packagePath -Artifact $attestation.artifacts.runtime
    Assert-AttestedDirectory -Directory $catalogPath -Artifact $attestation.artifacts.catalog
    $catalog = Get-Content -Raw -LiteralPath (Join-Path $catalogPath "catalog.json") |
        ConvertFrom-Json
    if ($catalog.authentication -ne "CHECKSUM-SHA256" -or
        (Test-Path -LiteralPath (Join-Path $catalogPath "signature.json"))) {
        throw "Scanner 2.0.1 Release requires a keyless checksum-only Catalog."
    }
    $workerSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $workerPath).Hash.ToLowerInvariant()
    if ($workerSha256 -ne $compositionValue.production_release.worker_executable_sha256) {
        throw "Standalone Worker checksum does not match the Scanner 2.0.1 composition."
    }
}
else {
    & $PythonExecutable -m bixolon_scanner.operations.release_composition `
        --composition $compositionPath `
        --repository-root $repositoryRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Release composition verification failed."
    }
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
                path = Get-RelativeReleasePath -BasePath $temporaryBundle -TargetPath $_.FullName
                size_bytes = $_.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
    $manifest = [ordered]@{
        schema_version = "1.0"
        release = $releaseName
        app_version = $appVersion
        composition_path = Get-RelativeReleasePath -BasePath $repositoryRoot -TargetPath $compositionPath
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
