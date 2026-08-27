param(
    [string]$Version = "0.1.5",
    [string]$Stamp = "20260824",
    [string]$OutputRoot = "artifacts/handoff",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Assert-SafeChildPath {
    param([string]$Path, [string]$Parent)
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $prefix = $resolvedParent + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the candidate output: $resolvedPath"
    }
}

function Write-Utf8 {
    param([string]$Path, [string]$Text)
    [System.IO.File]::WriteAllText(
        $Path,
        $Text,
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

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $repositoryRoot "configs/versions/$Version.json"
$resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$candidateName = "n100-$Version-candidate-$Stamp"
$candidateRoot = Join-Path $resolvedOutputRoot $candidateName
$zipPath = "$candidateRoot.zip"
$zipHashPath = "$zipPath.sha256"
$versionRoot = Join-Path $repositoryRoot "artifacts/versions/$Version"
$workerSource = Join-Path $versionRoot "openvino-worker-build/bixolon-worker"
$staging = Join-Path $versionRoot "staging"
$runtimeSource = Join-Path $staging "runtime"
$catalogSource = Join-Path $staging "catalog"
$metadataPath = Join-Path $runtimeSource "metadata.json"

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Version config is missing: $configPath"
}
$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
if ([string]$config.version -ne $Version) {
    throw "Version config identity mismatch: $configPath"
}
$cpuReferenceEvidence = @(
    $config.evaluation_evidence | Where-Object {
        [string]$_.path -match "(^|/)full-valid-openvino\.json$"
    }
)
if ($cpuReferenceEvidence.Count -ne 1) {
    throw "Version config must pin exactly one full-valid OpenVINO reference."
}
$cpuReferencePath = Join-Path $repositoryRoot ([string]$cpuReferenceEvidence[0].path)
if (-not (Test-Path -LiteralPath $cpuReferencePath -PathType Leaf)) {
    throw "Pinned OpenVINO reference is missing: $cpuReferencePath"
}
$cpuReferenceSha256 = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $cpuReferencePath
).Hash.ToLowerInvariant()
if ($cpuReferenceSha256 -ne [string]$cpuReferenceEvidence[0].sha256) {
    throw "Pinned OpenVINO reference SHA-256 mismatch: $cpuReferencePath"
}

foreach ($required in @(
    (Join-Path $workerSource "bixolon-worker.exe"),
    (Join-Path $workerSource "_internal"),
    $metadataPath,
    (Join-Path $catalogSource "catalog.json"),
    (Join-Path $repositoryRoot "scripts/handoff/start-worker.ps1"),
    (Join-Path $repositoryRoot "scripts/handoff/benchmark-n100.ps1"),
    (Join-Path $repositoryRoot "scripts/handoff/N100-STAGE-TEST.ps1"),
    (Join-Path $repositoryRoot "scripts/handoff/RUN-N100-TEST.cmd"),
    (Join-Path $repositoryRoot "scripts/handoff/README-N100-KO.txt"),
    (Join-Path $repositoryRoot "configs/runtime/requirements-windows-openvino.lock")
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required N100 candidate input is missing: $required"
    }
}

$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
if ([string]$metadata.worker_version -ne $Version) {
    throw "Runtime version mismatch: $($metadata.worker_version)"
}
if (
    $null -ne $metadata.detector.ensemble -or
    [string]$metadata.detector.filename -ne "detector.onnx" -or
    [double]$metadata.detector.score_threshold -ne 0.65 -or
    $null -ne $metadata.count_verifier -or
    [string]$metadata.embedder.embedder_id -ne "dinov3-convnext-tiny" -or
    $null -eq $metadata.classifier_verification -or
    [double]$metadata.classifier_verification.ambiguity_maximum_approval_score -ne 0.5 -or
    [int]$metadata.classifier_verification.independent_embedder.fixed_batch_size -ne 1
) {
    throw "The N100 candidate does not match the selected consensus Runtime."
}

[System.IO.Directory]::CreateDirectory($resolvedOutputRoot) | Out-Null
foreach ($target in @($candidateRoot, $zipPath, $zipHashPath)) {
    if (Test-Path -LiteralPath $target) {
        if (-not $Force) {
            throw "Candidate output already exists; pass -Force to replace it: $target"
        }
        Assert-SafeChildPath -Path $target -Parent $resolvedOutputRoot
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

$temporaryRoot = Join-Path $resolvedOutputRoot ("." + [System.IO.Path]::GetRandomFileName())
Assert-SafeChildPath -Path $temporaryRoot -Parent $resolvedOutputRoot
[System.IO.Directory]::CreateDirectory($temporaryRoot) | Out-Null
try {
    $workerTarget = Join-Path $temporaryRoot "worker"
    [System.IO.Directory]::CreateDirectory($workerTarget) | Out-Null
    Copy-Item -LiteralPath (Join-Path $workerSource "bixolon-worker.exe") -Destination $workerTarget
    Copy-Item -LiteralPath (Join-Path $workerSource "_internal") -Destination $workerTarget -Recurse
    Copy-Item -LiteralPath $runtimeSource -Destination (Join-Path $workerTarget "model-package") -Recurse
    Copy-Item -LiteralPath $catalogSource -Destination (Join-Path $workerTarget "store-catalog") -Recurse
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "scripts/handoff/start-worker.ps1") `
        -Destination $temporaryRoot
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "scripts/handoff/benchmark-n100.ps1") `
        -Destination $temporaryRoot
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "scripts/handoff/N100-STAGE-TEST.ps1") `
        -Destination (Join-Path $temporaryRoot "N100-STAGE-TEST.ps1")
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "scripts/handoff/RUN-N100-TEST.cmd") `
        -Destination $temporaryRoot
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "scripts/handoff/README-N100-KO.txt") `
        -Destination $temporaryRoot
    Copy-Item -LiteralPath (
        Join-Path $repositoryRoot "configs/runtime/requirements-windows-openvino.lock"
    ) -Destination $temporaryRoot
    Copy-Item -LiteralPath $cpuReferencePath `
        -Destination (Join-Path $temporaryRoot "local-cpu-reference.json")

    $forbidden = Get-ChildItem -LiteralPath $temporaryRoot -File -Recurse | Where-Object {
        $_.Name -match "(?i)(directml|cuda|tensorrt|cudnn|cublas|cudart|cufft|nvrtc)"
    }
    if ($forbidden) {
        throw "CPU test candidate contains a GPU runtime file: $($forbidden[0].FullName)"
    }

    $files = @(
        Get-ChildItem -LiteralPath $temporaryRoot -File -Recurse | Sort-Object FullName |
            ForEach-Object {
                [ordered]@{
                    path = Get-RelativePackagePath -Root $temporaryRoot -Path $_.FullName
                    size_bytes = $_.Length
                    sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
                }
            }
    )
    $manifest = [ordered]@{
        schema_version = "1.0"
        artifact = "n100_openvino_cpu_latency_candidate"
        product_version = $Version
        provider = "OpenVINOExecutionProvider:CPU"
        detector_filename = [string]$metadata.detector.filename
        classifier_verifier_filename = [string]$metadata.classifier_verification.independent_embedder.filename
        embedder_id = [string]$metadata.embedder.embedder_id
        default_profile = [ordered]@{
            provider = "openvino"
            detector_workers = 1
            detector_threads_per_session = 0
            embedder_threads = 0
        }
        target_full_path_latency_ms = 500
        current_pc_reference = "local-cpu-reference.json"
        reference_evidence_path = [string]$cpuReferenceEvidence[0].path
        reference_evidence_sha256 = $cpuReferenceSha256
        file_count = $files.Count
        files = $files
    }
    Write-Utf8 -Path (Join-Path $temporaryRoot "candidate-manifest.json") `
        -Text (($manifest | ConvertTo-Json -Depth 20) + [Environment]::NewLine)
    [System.IO.Directory]::Move($temporaryRoot, $candidateRoot)
}
catch {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
    throw
}

Compress-Archive -LiteralPath $candidateRoot -DestinationPath $zipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
Write-Utf8 -Path $zipHashPath `
    -Text ("$zipHash  $([System.IO.Path]::GetFileName($zipPath))" + [Environment]::NewLine)

Write-Host "N100 candidate: $candidateRoot"
Write-Host "ZIP: $zipPath"
Write-Host "ZIP SHA-256: $zipHash"
