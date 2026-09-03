param(
    [string]$Version = "0.1.14",
    [string]$OutputRoot = "artifacts/external-sdk",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Assert-SafeChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Parent
    )
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $prefix = $resolvedParent + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the expected directory: $resolvedPath"
    }
}

function Write-JsonFile {
    param([string]$Path, [object]$Value)
    $json = $Value | ConvertTo-Json -Depth 30
    [System.IO.File]::WriteAllText(
        $Path,
        $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Get-RelativePackagePath {
    param([string]$Root, [string]$Path)
    $rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    $rootUri = [System.Uri]::new($rootPath)
    $pathUri = [System.Uri]::new([System.IO.Path]::GetFullPath($Path))
    return [System.Uri]::UnescapeDataString($rootUri.MakeRelativeUri($pathUri).ToString())
}

function Copy-DirectoryExact {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Source directory is missing: $Source"
    }
    [System.IO.Directory]::CreateDirectory($Destination) | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
    }
}

function Remove-DevelopmentState {
    param([string]$Root)
    $generatedDirectories = @(
        Get-ChildItem -LiteralPath $Root -Directory -Force -Recurse |
            Where-Object { $_.Name -in @(".dart_tool", "build") } |
            Sort-Object { $_.FullName.Length } -Descending
    )
    foreach ($directory in $generatedDirectories) {
        if (Test-Path -LiteralPath $directory.FullName) {
            Assert-SafeChildPath -Path $directory.FullName -Parent $Root
            Remove-Item -LiteralPath $directory.FullName -Recurse -Force
        }
    }
    Get-ChildItem -LiteralPath $Root -File -Force -Recurse |
        Where-Object { $_.Name -eq "pubspec.lock" } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
}

function Get-FileEntries {
    param([string]$Root, [string[]]$ExcludeRelativePaths = @())
    $excluded = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    foreach ($item in $ExcludeRelativePaths) {
        [void]$excluded.Add($item.Replace("\", "/"))
    }
    return @(
        Get-ChildItem -LiteralPath $Root -File -Recurse | ForEach-Object {
            $relativePath = Get-RelativePackagePath -Root $Root -Path $_.FullName
            if (-not $excluded.Contains($relativePath)) {
                [ordered]@{
                    path = $relativePath
                    size_bytes = $_.Length
                    sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
                }
            }
        } | Sort-Object path
    )
}

function Assert-DirectoryCopyMatches {
    param([string]$Source, [string]$Destination)
    $sourceEntries = Get-FileEntries -Root $Source
    $destinationEntries = Get-FileEntries -Root $Destination
    if (
        ($sourceEntries | ConvertTo-Json -Depth 5 -Compress) -ne
        ($destinationEntries | ConvertTo-Json -Depth 5 -Compress)
    ) {
        throw "Directory copy changed the payload: $Destination"
    }
}

function Write-ZipHash {
    param([string]$ZipPath)
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ZipPath).Hash.ToLowerInvariant()
    $line = "$hash  $([System.IO.Path]::GetFileName($ZipPath))" + [Environment]::NewLine
    [System.IO.File]::WriteAllText(
        "$ZipPath.sha256",
        $line,
        [System.Text.UTF8Encoding]::new($false)
    )
    return $hash
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

$storeId = [string]$config.catalog.store_id
$versionRoot = Join-Path $repositoryRoot "artifacts/versions/$Version"
$runtimeSource = Join-Path $versionRoot "staging/runtime"
$catalogSource = Join-Path $versionRoot "staging/catalog"
$workerSource = Join-Path $versionRoot "openvino-gpu-worker-build/bixolon-worker"
$flutterSdkSource = Join-Path $repositoryRoot "sdk/flutter/bixolon_scanner_sdk"
$mockWorkerSource = Join-Path $repositoryRoot "sdk/mock_worker"
$externalDocsSource = Join-Path $repositoryRoot "sdk/external"

foreach ($requiredDirectory in @(
    $runtimeSource,
    $catalogSource,
    $workerSource,
    $flutterSdkSource,
    $mockWorkerSource,
    $externalDocsSource
)) {
    if (-not (Test-Path -LiteralPath $requiredDirectory -PathType Container)) {
        throw "Required source directory is missing: $requiredDirectory"
    }
}

$runtimeMetadata = Get-Content -Raw -LiteralPath (Join-Path $runtimeSource "metadata.json") |
    ConvertFrom-Json
$catalogMetadata = Get-Content -Raw -LiteralPath (Join-Path $catalogSource "catalog.json") |
    ConvertFrom-Json
if (
    [string]$runtimeMetadata.schema_version -ne "2.0" -or
    [string]$catalogMetadata.schema_version -ne "2.0" -or
    [string]$runtimeMetadata.worker_version -ne $Version -or
    [string]$catalogMetadata.catalog_version -ne $Version -or
    [string]$catalogMetadata.store_id -ne $storeId
) {
    throw "Runtime/Catalog identity or schema is inconsistent with the version config."
}
if ([string]$catalogMetadata.authentication -ne "CHECKSUM-SHA256") {
    throw "Catalog authentication must be CHECKSUM-SHA256."
}
if (Test-Path -LiteralPath (Join-Path $catalogSource "signature.json")) {
    throw "signature.json must not be included in an active Catalog."
}

$resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$outputVersionRoot = Join-Path $resolvedOutputRoot $Version
if (Test-Path -LiteralPath $outputVersionRoot) {
    if (-not $Force) {
        throw "Output already exists. Pass -Force to replace it: $outputVersionRoot"
    }
    Assert-SafeChildPath -Path $outputVersionRoot -Parent $resolvedOutputRoot
    Remove-Item -LiteralPath $outputVersionRoot -Recurse -Force
}
[System.IO.Directory]::CreateDirectory($outputVersionRoot) | Out-Null

$integrationName = "BIXOLON-Scanner-External-SDK-$Version"
$integrationRoot = Join-Path $outputVersionRoot $integrationName
$storeName = "BIXOLON-Store-Model-$storeId-$Version"
$storeRoot = Join-Path $outputVersionRoot $storeName
[System.IO.Directory]::CreateDirectory($integrationRoot) | Out-Null
[System.IO.Directory]::CreateDirectory($storeRoot) | Out-Null

Copy-Item -LiteralPath (Join-Path $externalDocsSource "README-KO.md") `
    -Destination (Join-Path $integrationRoot "README-KO.md")
Copy-DirectoryExact -Source $flutterSdkSource `
    -Destination (Join-Path $integrationRoot "flutter-sdk/bixolon_scanner_sdk")
Copy-DirectoryExact -Source $mockWorkerSource `
    -Destination (Join-Path $integrationRoot "mock-worker")
Remove-DevelopmentState -Root (Join-Path $integrationRoot "flutter-sdk/bixolon_scanner_sdk")
Remove-DevelopmentState -Root (Join-Path $integrationRoot "mock-worker")
Copy-DirectoryExact -Source (Join-Path $externalDocsSource "deployment") `
    -Destination (Join-Path $integrationRoot "deployment")

$contractsRoot = Join-Path $integrationRoot "contracts"
$contractExamples = Join-Path $contractsRoot "examples/$Version"
[System.IO.Directory]::CreateDirectory($contractExamples) | Out-Null
Copy-Item -LiteralPath (Join-Path $repositoryRoot "docs/contracts/api.md") `
    -Destination (Join-Path $contractsRoot "api.md")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "docs/contracts/worker-integration-$Version.md") `
    -Destination (Join-Path $contractsRoot "worker-integration-$Version.md")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "schemas/scan-response.schema.json") `
    -Destination (Join-Path $contractsRoot "scan-response.schema.json")
Copy-DirectoryExact -Source (Join-Path $repositoryRoot "docs/contracts/examples/$Version") `
    -Destination $contractExamples

$integrationLicenses = Join-Path $integrationRoot "licenses"
[System.IO.Directory]::CreateDirectory($integrationLicenses) | Out-Null
Copy-Item -LiteralPath (Join-Path $repositoryRoot "licenses/APACHE-2.0.txt") `
    -Destination (Join-Path $integrationLicenses "APACHE-2.0.txt")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "licenses/DINOV3-LICENSE.md") `
    -Destination (Join-Path $integrationLicenses "DINOV3-LICENSE.md")

$runtimePayloadRoot = Join-Path $integrationRoot "runtime-payload/bixolon_runtime"
$workerDestination = Join-Path $runtimePayloadRoot "worker"
Copy-DirectoryExact -Source $workerSource -Destination $workerDestination
Assert-DirectoryCopyMatches -Source $workerSource -Destination $workerDestination

$forbiddenRuntimeFiles = @(
    Get-ChildItem -LiteralPath $runtimePayloadRoot -File -Recurse |
        Where-Object {
            $_.Name -match "(?i)(cuda|cudnn|cublas|cufft|nvrtc|tensorrt|directml)"
        }
)
if ($forbiddenRuntimeFiles.Count -gt 0) {
    throw "Non-OpenVINO provider payload found: $($forbiddenRuntimeFiles[0].FullName)"
}

$runtimeManifestPath = Join-Path $integrationRoot "runtime-payload/runtime-manifest.json"
$runtimeEntries = Get-FileEntries -Root $runtimePayloadRoot
Write-JsonFile -Path $runtimeManifestPath -Value ([ordered]@{
    schema_version = "1.0"
    product_version = $Version
    platform = "windows-x64"
    provider = "openvino"
    worker_entrypoint = "bixolon_runtime/worker/bixolon-worker.exe"
    file_count = $runtimeEntries.Count
    files = $runtimeEntries
})

Copy-DirectoryExact -Source $runtimeSource -Destination (Join-Path $storeRoot "model-package")
Copy-DirectoryExact -Source $catalogSource -Destination (Join-Path $storeRoot "store-catalog")
Assert-DirectoryCopyMatches -Source $runtimeSource -Destination (Join-Path $storeRoot "model-package")
Assert-DirectoryCopyMatches -Source $catalogSource -Destination (Join-Path $storeRoot "store-catalog")

$storeLicenses = Join-Path $storeRoot "licenses"
[System.IO.Directory]::CreateDirectory($storeLicenses) | Out-Null
Copy-Item -LiteralPath (Join-Path $repositoryRoot "licenses/APACHE-2.0.txt") `
    -Destination (Join-Path $storeLicenses "APACHE-2.0.txt")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "licenses/DINOV3-LICENSE.md") `
    -Destination (Join-Path $storeLicenses "DINOV3-LICENSE.md")

Write-JsonFile -Path (Join-Path $storeRoot "version.json") -Value ([ordered]@{
    schema_version = "1.0"
    version = $Version
    store_id = $storeId
})
Write-JsonFile -Path (Join-Path $storeRoot "provenance.json") -Value ([ordered]@{
    schema_version = "1.0"
    version = $Version
    store_id = $storeId
    source_candidate = [string]$config.source_candidate
    source_artifacts = [ordered]@{
        runtime = [ordered]@{
            path = [string]$config.runtime.path
            manifest_sha256 = [string]$config.runtime.manifest_sha256
        }
        catalog = [ordered]@{
            path = [string]$config.catalog.path
            manifest_sha256 = [string]$config.catalog.manifest_sha256
        }
    }
    evaluation_evidence = @($config.evaluation_evidence)
    transformation = [ordered]@{
        model_graph_or_weight_changed = $false
        catalog_authentication = "CHECKSUM-SHA256"
        lifecycle_fields_emitted = $false
    }
})
Write-JsonFile -Path (Join-Path $storeRoot "store-bundle.json") -Value ([ordered]@{
    schema_version = "1.0"
    bundle_kind = "BIXOLON_STORE_MODEL"
    store_id = $storeId
    product_version = $Version
    worker_runtime_schema = [string]$runtimeMetadata.schema_version
    catalog_schema = [string]$catalogMetadata.schema_version
    provider = "openvino"
    model_package_directory = "model-package"
    store_catalog_directory = "store-catalog"
    update_unit = "atomic_directory"
})

$storeManifestPath = Join-Path $storeRoot "bundle-manifest.json"
$storeEntries = Get-FileEntries -Root $storeRoot -ExcludeRelativePaths @("bundle-manifest.json")
Write-JsonFile -Path $storeManifestPath -Value ([ordered]@{
    schema_version = "1.0"
    bundle_kind = "BIXOLON_STORE_MODEL"
    product_version = $Version
    store_id = $storeId
    authentication = "CHECKSUM-SHA256"
    manifest_excludes = @("bundle-manifest.json")
    file_count = $storeEntries.Count
    files = $storeEntries
})

$integrationManifestPath = Join-Path $integrationRoot "integration-manifest.json"
$integrationEntries = Get-FileEntries -Root $integrationRoot `
    -ExcludeRelativePaths @("integration-manifest.json")
Write-JsonFile -Path $integrationManifestPath -Value ([ordered]@{
    schema_version = "1.0"
    distribution = "BIXOLON_SCANNER_EXTERNAL_SDK"
    product_version = $Version
    sdk_version = "1.0.0"
    platform = "windows-x64"
    provider = "openvino"
    manifest_excludes = @("integration-manifest.json")
    file_count = $integrationEntries.Count
    files = $integrationEntries
})

Add-Type -AssemblyName System.IO.Compression.FileSystem
$integrationZip = Join-Path $outputVersionRoot "$integrationName.zip"
$storeZip = Join-Path $outputVersionRoot "$storeName.zip"
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $integrationRoot,
    $integrationZip,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $true
)
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $storeRoot,
    $storeZip,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $true
)
$integrationZipHash = Write-ZipHash -ZipPath $integrationZip
$storeZipHash = Write-ZipHash -ZipPath $storeZip

$summaryPath = Join-Path $outputVersionRoot "build-summary.json"
Write-JsonFile -Path $summaryPath -Value ([ordered]@{
    schema_version = "1.0"
    product_version = $Version
    provider = "openvino"
    store_id = $storeId
    artifacts = @(
        [ordered]@{
            kind = "external_sdk"
            path = [System.IO.Path]::GetFileName($integrationZip)
            size_bytes = (Get-Item -LiteralPath $integrationZip).Length
            sha256 = $integrationZipHash
        },
        [ordered]@{
            kind = "store_model_bundle"
            path = [System.IO.Path]::GetFileName($storeZip)
            size_bytes = (Get-Item -LiteralPath $storeZip).Length
            sha256 = $storeZipHash
        }
    )
})

Write-Host "External SDK ZIP: $integrationZip"
Write-Host "Store Model ZIP: $storeZip"
Write-Host "Summary: $summaryPath"
