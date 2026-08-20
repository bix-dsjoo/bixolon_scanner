param(
    [string]$Version = "0.0.1",
    [string]$FlutterExecutable = "C:/Users/OMEN/development/flutter/bin/flutter.bat",
    [string]$PythonExecutable = "C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe"
)

$ErrorActionPreference = "Stop"

function Get-RelativeVersionPath {
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
        throw "Version path is outside its expected root: $targetFullPath"
    }
    return $targetFullPath.Substring($baseFullPath.Length).Replace("\", "/")
}

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

    $required = @(
        "product_scanner.exe",
        "worker/bixolon-worker.exe",
        "worker/model-package/metadata.json",
        "worker/store-catalog/catalog.json",
        "worker/store-catalog/checksums.json",
        "worker/model-package/licenses/APACHE-2.0.txt",
        "worker/model-package/licenses/DINOV3-LICENSE.md",
        "worker/model-package/licenses/THIRD_PARTY_MODELS.md",
        "worker/cuda-runtime/cudart64_13.dll",
        "worker/cuda-runtime/cublas64_13.dll",
        "worker/cuda-runtime/cudnn64_9.dll"
    )
    foreach ($relative in $required) {
        $path = Join-Path $temporaryBundle $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Version bundle is missing a required file: $relative"
        }
    }
    if (Test-Path -LiteralPath (
            Join-Path $temporaryBundle "worker/store-catalog/signature.json")) {
        throw "Version bundle must not contain a Catalog signature."
    }

    $runtimeMetadata = Get-Content -Raw -LiteralPath (
        Join-Path $temporaryBundle "worker/model-package/metadata.json") | ConvertFrom-Json
    $catalogMetadata = Get-Content -Raw -LiteralPath (
        Join-Path $temporaryBundle "worker/store-catalog/catalog.json") | ConvertFrom-Json
    $componentVersions = @(
        [string]$runtimeMetadata.worker_version,
        [string]$runtimeMetadata.detector.version,
        [string]$runtimeMetadata.embedder.version,
        [string]$runtimeMetadata.detector_policy_version,
        [string]$runtimeMetadata.classifier_policy.version,
        [string]$catalogMetadata.catalog_version,
        [string]$catalogMetadata.embedder_version,
        [string]$catalogMetadata.classifier_policy_version
    )
    if (@($componentVersions | Where-Object { $_ -ne $Version }).Count -ne 0) {
        throw "Version bundle contains mixed component versions."
    }
    if ($null -ne $runtimeMetadata.promotion_status -or
        $null -ne $runtimeMetadata.promotion) {
        throw "Version bundle contains a lifecycle field."
    }

    $productExecutable = Join-Path $temporaryBundle "product_scanner.exe"
    $productVersion = [System.Diagnostics.FileVersionInfo]::GetVersionInfo(
        $productExecutable).ProductVersion
    if ($productVersion -ne $Version) {
        throw "Windows ProductVersion does not match ${Version}: $productVersion"
    }

    $files = Get-ChildItem -LiteralPath $temporaryBundle -Recurse -File |
        Sort-Object FullName |
        ForEach-Object {
            [ordered]@{
                path = Get-RelativeVersionPath `
                    -BasePath $temporaryBundle `
                    -TargetPath $_.FullName
                size_bytes = $_.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
    $manifest = [ordered]@{
        schema_version = "1.0"
        version = $Version
        app_build = $appBuild
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
Write-Host "Version bundle: $targetBundle"
Write-Host "Bundle manifest SHA-256: $bundleManifestHash"
