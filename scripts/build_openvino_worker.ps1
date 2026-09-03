param(
    [string]$Version = "0.1.14",
    [string]$Python311Executable = "C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe",
    [switch]$ReuseBuildEnvironment,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Command,
        [Parameter(Mandatory = $true)][string]$FailureMessage
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

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$sourceDirectory = Join-Path $repositoryRoot "src"
$configPath = Join-Path $repositoryRoot "configs/versions/$Version.json"
$lockPath = Join-Path $repositoryRoot "configs/runtime/requirements-windows-openvino.lock"
$buildEnvironmentRoot = Join-Path $repositoryRoot "artifacts/build-envs"
$buildEnvironment = Join-Path $buildEnvironmentRoot "worker-openvino-gpu-py311"
$versionRoot = Join-Path $repositoryRoot "artifacts/versions/$Version"
$stagingRoot = Join-Path $versionRoot "staging"
$workerOutput = Join-Path $versionRoot "openvino-gpu-worker-build"
$workerDist = Join-Path $workerOutput "bixolon-worker"

foreach ($requiredPath in @(
    $configPath,
    $lockPath,
    (Join-Path $stagingRoot "runtime/metadata.json"),
    (Join-Path $stagingRoot "catalog/catalog.json"),
    (Join-Path $stagingRoot "version.json")
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required OpenVINO Worker input is missing: $requiredPath"
    }
}

$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
if ([string]$config.version -ne $Version) {
    throw "Version config identity mismatch: $configPath"
}
$runtimeMetadata = Get-Content -Raw -LiteralPath (
    Join-Path $stagingRoot "runtime/metadata.json"
) | ConvertFrom-Json
if (
    [string]$runtimeMetadata.worker_version -ne $Version -or
    $null -ne $runtimeMetadata.count_verifier -or
    [string]$runtimeMetadata.sources.detector.architecture -notmatch "SSDLite320" -or
    -not [bool]$runtimeMetadata.classifier_resolution_fallback.selective_roi_only -or
    [string]$runtimeMetadata.detector_class_mode -ne "class_agnostic" -or
    $null -ne $runtimeMetadata.detector_primary_classifier_routing -or
    [int]$runtimeMetadata.embedder.input_size[0] -ne 192 -or
    [int]$runtimeMetadata.classifier_resolution_fallback.embedder.input_size[0] -ne 224 -or
    [int]$runtimeMetadata.classifier_verification.independent_embedder.input_size[0] -ne 160 -or
    [bool]$runtimeMetadata.classifier_verification.verify_all_approved_candidates
) {
    throw "OpenVINO Worker requires the final class-agnostic consistent-evidence Runtime."
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
    Invoke-Native -FailureMessage "OpenVINO build environment creation failed" -Command {
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
        & $buildPython -m bixolon_scanner.operations.version_bundle verify `
            --config $configPath `
            --repository-root $repositoryRoot
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

if (Test-Path -LiteralPath $workerOutput) {
    if (-not $Force) {
        throw "OpenVINO Worker output already exists; pass -Force to replace it: $workerOutput"
    }
    Assert-SafeChildPath -Path $workerOutput -Parent $versionRoot
    Remove-Item -LiteralPath $workerOutput -Recurse -Force
}
Invoke-Native -FailureMessage "OpenVINO Worker build failed" -Command {
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
        throw "Required OpenVINO Worker file is missing: $requiredPath"
    }
}
$forbiddenWorkerFiles = Get-ChildItem -LiteralPath $workerDist -File -Recurse | Where-Object {
    $_.Name -match "(?i)(directml|cuda|cudnn|cublas|cudart|cufft|nvrtc|tensorrt)"
}
if ($forbiddenWorkerFiles) {
    throw "OpenVINO Worker contains a forbidden DirectML, CUDA, or TensorRT runtime file."
}

Write-Host "OpenVINO CPU detector + Intel GPU classifier Worker: $workerDist"
