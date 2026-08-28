param(
    [string]$Version = "0.1.8",
    [string]$Python311Executable = "C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe",
    [string]$OutputRoot = "artifacts/handoff",
    [switch]$ReuseBuildEnvironment,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)."
    }
}

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

function Get-DirectoryRecords {
    param([Parameter(Mandatory = $true)][string]$Root)
    return @(
        Get-ChildItem -LiteralPath $Root -File -Recurse | Sort-Object FullName | ForEach-Object {
            [ordered]@{
                path = Get-RelativePackagePath -Root $Root -Path $_.FullName
                size_bytes = $_.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
    )
}

function Assert-DirectoryCopyMatches {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Target
    )
    $sourceRecords = Get-DirectoryRecords -Root $Source
    $targetRecords = Get-DirectoryRecords -Root $Target
    if (
        ($sourceRecords | ConvertTo-Json -Depth 4 -Compress) -ne
        ($targetRecords | ConvertTo-Json -Depth 4 -Compress)
    ) {
        throw "Directory copy changed a Runtime or Catalog payload: $Target"
    }
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Value
    )
    $json = $Value | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText(
        $Path,
        $json + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$sourceDirectory = Join-Path $repositoryRoot "src"
$configPath = Join-Path $repositoryRoot "configs/versions/$Version.json"
$lockPath = Join-Path $repositoryRoot "configs/runtime/requirements-windows-openvino.lock"
$benchmarkScript = Join-Path $repositoryRoot "scripts/handoff/N100-GPU-BENCHMARK.ps1"
$commandScript = Join-Path $repositoryRoot "scripts/handoff/RUN-N100-GPU-TEST.cmd"
$readmePath = Join-Path $repositoryRoot "scripts/handoff/README-N100-GPU-KO.txt"
$resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$packageName = "n100-$Version-openvino-cpu-gpu-test"
$packageRoot = Join-Path $resolvedOutputRoot $packageName
$zipPath = Join-Path $resolvedOutputRoot "$packageName.zip"
$zipHashPath = "$zipPath.sha256"
$buildEnvironmentRoot = Join-Path $repositoryRoot "artifacts/build-envs"
$buildEnvironment = Join-Path $buildEnvironmentRoot "worker-openvino-gpu-py311"
$versionRoot = Join-Path $repositoryRoot "artifacts/versions/$Version"
$workerOutput = Join-Path $versionRoot "openvino-gpu-worker-build"
$workerDist = Join-Path $workerOutput "bixolon-worker"
$stagingRoot = Join-Path $versionRoot "staging"

foreach ($requiredPath in @(
    $configPath,
    $lockPath,
    $benchmarkScript,
    $commandScript,
    $readmePath,
    (Join-Path $stagingRoot "runtime/metadata.json"),
    (Join-Path $stagingRoot "catalog/catalog.json"),
    (Join-Path $stagingRoot "version.json")
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required OpenVINO GPU test input is missing: $requiredPath"
    }
}
$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
if ([string]$config.version -ne $Version) {
    throw "Version config identity mismatch: $configPath"
}
$runtimeMetadataPath = Join-Path $stagingRoot "runtime/metadata.json"
$runtimeMetadata = Get-Content -Raw -LiteralPath $runtimeMetadataPath | ConvertFrom-Json
if (
    [string]$runtimeMetadata.worker_version -ne $Version -or
    $null -eq $runtimeMetadata.count_verifier -or
    [string]$runtimeMetadata.count_verifier.filename -ne "count-verifier.onnx" -or
    [string]$runtimeMetadata.count_verifier.comparison_mode -ne "object_presence" -or
    [double]$runtimeMetadata.count_verifier.confidence_threshold -ne 0.54
) {
    throw "OpenVINO GPU test requires the final 0.1.8 object-presence Runtime."
}
$countVerifierPath = Join-Path (
    Join-Path $stagingRoot "runtime"
) ([string]$runtimeMetadata.count_verifier.filename)
$countVerifierChecksumProperty = $runtimeMetadata.checksums.PSObject.Properties[
    [string]$runtimeMetadata.count_verifier.filename
]
if (
    -not (Test-Path -LiteralPath $countVerifierPath -PathType Leaf) -or
    $null -eq $countVerifierChecksumProperty -or
    (Get-FileHash -Algorithm SHA256 -LiteralPath $countVerifierPath).Hash.ToLowerInvariant() -ne
        [string]$countVerifierChecksumProperty.Value
) {
    throw "OpenVINO GPU test object-presence verifier checksum is invalid."
}

Invoke-Native -FailureMessage "Python 3.11 validation failed" -Command {
    & $Python311Executable -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
}

if (-not $ReuseBuildEnvironment -and (Test-Path -LiteralPath $buildEnvironment)) {
    Assert-SafeChildPath -Path $buildEnvironment -Parent $buildEnvironmentRoot
    Remove-Item -LiteralPath $buildEnvironment -Recurse -Force
}
if (-not (Test-Path -LiteralPath $buildEnvironment -PathType Container)) {
    [System.IO.Directory]::CreateDirectory($buildEnvironmentRoot) | Out-Null
    Invoke-Native -FailureMessage "OpenVINO GPU build environment creation failed" -Command {
        & $Python311Executable -m venv $buildEnvironment
    }
}
$buildPython = Join-Path $buildEnvironment "Scripts/python.exe"
Invoke-Native -FailureMessage "OpenVINO dependency installation failed" -Command {
    & $buildPython -m pip install --disable-pip-version-check --no-deps -r $lockPath
}
Invoke-Native -FailureMessage "OpenVINO ONNX Runtime validation failed" -Command {
    & $buildPython -c (
        "import onnxruntime as ort; " +
        "assert ort.__version__ == '1.24.1', ort.__version__; " +
        "providers=set(ort.get_available_providers()); " +
        "assert {'OpenVINOExecutionProvider','CPUExecutionProvider'} <= providers, providers; " +
        "assert providers.isdisjoint({'CUDAExecutionProvider','TensorrtExecutionProvider','DmlExecutionProvider'}), providers"
    )
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $sourceDirectory
    Invoke-Native -FailureMessage "Canonical version bundle verification failed" -Command {
        & $buildPython -m bixolon_scanner.operations.version_bundle verify --config $configPath --repository-root $repositoryRoot
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

if (Test-Path -LiteralPath $workerOutput) {
    if (-not $Force) {
        throw "OpenVINO GPU Worker output already exists; pass -Force to replace it: $workerOutput"
    }
    Assert-SafeChildPath -Path $workerOutput -Parent $versionRoot
    Remove-Item -LiteralPath $workerOutput -Recurse -Force
}
Invoke-Native -FailureMessage "OpenVINO GPU Worker build failed" -Command {
    & (Join-Path $PSScriptRoot "build_worker.ps1") `
        -PythonExecutable $buildPython `
        -OutputDirectory "artifacts/versions/$Version/openvino-gpu-worker-build" `
        -SourceDateEpoch ([long]$config.source_date_epoch) `
        -OpenVinoLibraryDirectory (Join-Path $buildEnvironment "Lib/site-packages/openvino/libs") `
        -IncludeOpenVinoGpu
}

$requiredWorkerFiles = @(
    (Join-Path $workerDist "bixolon-worker.exe"),
    (Join-Path $workerDist "_internal/onnxruntime/capi/onnxruntime.dll"),
    (Join-Path $workerDist "_internal/onnxruntime/capi/onnxruntime_providers_shared.dll"),
    (Join-Path $workerDist "_internal/onnxruntime/capi/onnxruntime_providers_openvino.dll"),
    (Join-Path $workerDist "_internal/openvino_intel_cpu_plugin.dll"),
    (Join-Path $workerDist "_internal/openvino_intel_gpu_plugin.dll")
)
foreach ($requiredPath in $requiredWorkerFiles) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required OpenVINO GPU Worker file is missing: $requiredPath"
    }
}
$forbiddenWorkerFiles = Get-ChildItem -LiteralPath $workerDist -File -Recurse | Where-Object {
    $_.Name -match "(?i)(directml|cuda|cudnn|cublas|cudart|cufft|nvrtc|tensorrt)"
}
if ($forbiddenWorkerFiles) {
    throw "OpenVINO GPU Worker contains a forbidden DirectML, CUDA, or TensorRT runtime file."
}

[System.IO.Directory]::CreateDirectory($resolvedOutputRoot) | Out-Null
foreach ($target in @($packageRoot, $zipPath, $zipHashPath)) {
    if (Test-Path -LiteralPath $target) {
        if (-not $Force) {
            throw "OpenVINO GPU test output already exists; pass -Force to replace it: $target"
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
    Copy-Item -LiteralPath (Join-Path $workerDist "bixolon-worker.exe") -Destination $workerTarget
    Copy-Item -LiteralPath (Join-Path $workerDist "_internal") -Destination $workerTarget -Recurse
    Copy-Item -LiteralPath (Join-Path $stagingRoot "runtime") -Destination (Join-Path $workerTarget "model-package") -Recurse
    Copy-Item -LiteralPath (Join-Path $stagingRoot "catalog") -Destination (Join-Path $workerTarget "store-catalog") -Recurse
    Assert-DirectoryCopyMatches -Source (Join-Path $stagingRoot "runtime") -Target (Join-Path $workerTarget "model-package")
    Assert-DirectoryCopyMatches -Source (Join-Path $stagingRoot "catalog") -Target (Join-Path $workerTarget "store-catalog")

    Copy-Item -LiteralPath $benchmarkScript -Destination $temporaryRoot
    Copy-Item -LiteralPath $commandScript -Destination $temporaryRoot
    Copy-Item -LiteralPath $readmePath -Destination $temporaryRoot
    Copy-Item -LiteralPath $lockPath -Destination $temporaryRoot
    Copy-Item -LiteralPath (Join-Path $stagingRoot "version.json") -Destination $temporaryRoot

    $runtimeRecords = Get-DirectoryRecords -Root (Join-Path $stagingRoot "runtime")
    $catalogRecords = Get-DirectoryRecords -Root (Join-Path $stagingRoot "catalog")
    $executionRuntimeRecords = @(
        foreach ($path in $requiredWorkerFiles[1..($requiredWorkerFiles.Count - 1)]) {
            [ordered]@{
                path = Get-RelativePackagePath -Root $workerDist -Path $path
                size_bytes = (Get-Item -LiteralPath $path).Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
            }
        }
    )
    $candidateManifest = [ordered]@{
        schema_version = "1.0"
        artifact = "n100_openvino_cpu_vs_intel_gpu_embedder_diagnostic"
        product_version = $Version
        provider = "openvino+openvino_gpu"
        target_full_path_latency_ms = 500
        provider_contract = [ordered]@{
            baseline_detector = "OpenVINOExecutionProvider:CPU"
            baseline_embedder = "OpenVINOExecutionProvider:CPU"
            candidate_detector = "OpenVINOExecutionProvider:CPU"
            baseline_object_presence_verifier = "OpenVINOExecutionProvider:CPU"
            candidate_object_presence_verifier = "OpenVINOExecutionProvider:GPU"
            candidate_object_presence_execution = "parallel_with_detector"
            candidate_embedder = "OpenVINOExecutionProvider:GPU"
            candidate_primary_embedder = "OpenVINOExecutionProvider:GPU"
            candidate_rotation_180_embedder = "OpenVINOExecutionProvider:GPU"
            candidate_independent_verifier_embedder = "OpenVINOExecutionProvider:GPU"
            silent_fallback_allowed = $false
        }
        transformation = [ordered]@{
            model_graph_or_weight_changed = $false
            decision_policy_changed = $false
            runtime_or_catalog_payload_changed = $false
        }
        source = [ordered]@{
            version_config_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $configPath).Hash.ToLowerInvariant()
            dependency_lock_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $lockPath).Hash.ToLowerInvariant()
        }
        runtime = [ordered]@{
            file_count = $runtimeRecords.Count
            files = $runtimeRecords
            object_presence_verifier = [ordered]@{
                filename = [string]$runtimeMetadata.count_verifier.filename
                comparison_mode = [string]$runtimeMetadata.count_verifier.comparison_mode
                confidence_threshold = [double]$runtimeMetadata.count_verifier.confidence_threshold
                sha256 = [string]$countVerifierChecksumProperty.Value
            }
        }
        catalog = [ordered]@{
            file_count = $catalogRecords.Count
            files = $catalogRecords
        }
        execution_runtime_files = $executionRuntimeRecords
        limitation = "Diagnostic package only; N100 field measurement is required."
    }
    Write-JsonFile -Path (Join-Path $temporaryRoot "candidate-manifest.json") -Value $candidateManifest

    $payloadRecords = Get-DirectoryRecords -Root $temporaryRoot
    $payloadManifest = [ordered]@{
        schema_version = "1.0"
        product_version = $Version
        artifact = "n100_openvino_cpu_vs_intel_gpu_embedder_diagnostic_payload"
        file_count = $payloadRecords.Count
        files = $payloadRecords
    }
    Write-JsonFile -Path (Join-Path $temporaryRoot "package-manifest.json") -Value $payloadManifest
    Move-Item -LiteralPath $temporaryRoot -Destination $packageRoot
}
catch {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
    throw
}

Compress-Archive -Path (Join-Path $packageRoot "*") -DestinationPath $zipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText(
    $zipHashPath,
    "$zipHash  $packageName.zip" + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "N100 OpenVINO CPU/GPU diagnostic package: $packageRoot"
Write-Host "ZIP: $zipPath"
Write-Host "SHA-256: $zipHash"
