param(
    [string]$Version = "0.1.14",
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
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Parent
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
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [object]$Value
    )
    $json = $Value | ConvertTo-Json -Depth 20
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

function Assert-DirectoryCopyMatches {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Target
    )
    $sourceFiles = @(
        Get-ChildItem -LiteralPath $Source -File -Recurse | Sort-Object FullName | ForEach-Object {
            [ordered]@{
                Path = Get-RelativePackagePath -Root $Source -Path $_.FullName
                Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash
            }
        }
    )
    $targetFiles = @(
        Get-ChildItem -LiteralPath $Target -File -Recurse | Sort-Object FullName | ForEach-Object {
            [ordered]@{
                Path = Get-RelativePackagePath -Root $Target -Path $_.FullName
                Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash
            }
        }
    )
    if (
        ($sourceFiles | ConvertTo-Json -Depth 4 -Compress) -ne
        ($targetFiles | ConvertTo-Json -Depth 4 -Compress)
    ) {
        throw "Directory copy changed a Runtime or Catalog payload: $Target"
    }
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$sourceDirectory = Join-Path $repositoryRoot "src"
$configPath = Join-Path $repositoryRoot "configs/versions/$Version.json"
$lockPath = Join-Path $repositoryRoot "configs/runtime/requirements-windows-openvino.lock"
$n100DiagnosticPath = Join-Path $repositoryRoot "docs/diagnostics/n100-worker-$Version.json"
$resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$handoffVersionRoot = Join-Path $resolvedOutputRoot $Version
$packageName = "bixolon-worker-$Version-windows-x64-openvino"
$packageRoot = Join-Path $handoffVersionRoot $packageName
$zipPath = Join-Path $handoffVersionRoot "$packageName.zip"
$zipHashPath = "$zipPath.sha256"
$buildEnvironment = Join-Path $repositoryRoot "artifacts/build-envs/worker-openvino-py311"
$workerOutput = "artifacts/versions/$Version/openvino-worker-build"
$workerOutputAbsolute = Join-Path $repositoryRoot $workerOutput

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Version config is missing: $configPath"
}
if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
    throw "OpenVINO dependency lock is missing: $lockPath"
}
$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
if ([string]$config.version -ne $Version) {
    throw "Version config identity mismatch: $configPath"
}
$n100Diagnostic = $null
$recommendedN100Profile = $null
if (Test-Path -LiteralPath $n100DiagnosticPath -PathType Leaf) {
    $n100Diagnostic = Get-Content -Raw -LiteralPath $n100DiagnosticPath | ConvertFrom-Json
    if (
        [string]$n100Diagnostic.product_version -ne $Version -or
        [string]$n100Diagnostic.provider -ne "openvino" -or
        -not [bool]$n100Diagnostic.response_contract_safe -or
        -not [bool]$n100Diagnostic.passes
    ) {
        throw "N100 diagnostic version, provider, or response/error checks are invalid."
    }
    $recommendedN100Profiles = @(
        $n100Diagnostic.profiles |
            Where-Object { $_.name -eq $n100Diagnostic.recommended_profile.name }
    )
    if ($recommendedN100Profiles.Count -ne 1) {
        throw "N100 diagnostic must contain exactly one recommended profile result."
    }
    $recommendedN100Profile = $recommendedN100Profiles[0]
}
$sourceDateEpoch = [long]$config.source_date_epoch
$versionRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $repositoryRoot ([string]$config.output_root + "/" + $Version)))
$stagingRoot = Join-Path $versionRoot "staging"

Invoke-Native -FailureMessage "Python 3.11 validation failed" -Command {
    & $Python311Executable -c (
        "import sys; assert sys.version_info[:2] == (3, 11), " +
        "f'Python 3.11 required, got {sys.version}'"
    )
}

if (-not $ReuseBuildEnvironment -and (Test-Path -LiteralPath $buildEnvironment)) {
    Assert-SafeChildPath -Path $buildEnvironment -Parent (Join-Path $repositoryRoot "artifacts/build-envs")
    Remove-Item -LiteralPath $buildEnvironment -Recurse -Force
}
if (-not (Test-Path -LiteralPath $buildEnvironment -PathType Container)) {
    [System.IO.Directory]::CreateDirectory((Split-Path -Parent $buildEnvironment)) | Out-Null
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
        "providers = ort.get_available_providers(); " +
        "assert 'OpenVINOExecutionProvider' in providers, providers; " +
        "assert 'CPUExecutionProvider' in providers, providers; " +
        "forbidden = {'CUDAExecutionProvider', 'TensorrtExecutionProvider', " +
        "'DmlExecutionProvider'}; " +
        "assert forbidden.isdisjoint(providers), providers"
    )
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $sourceDirectory
    Invoke-Native -FailureMessage "Version preparation failed" -Command {
        & $buildPython -m bixolon_scanner.operations.version_bundle prepare `
            --config $configPath `
            --repository-root $repositoryRoot
    }
    Invoke-Native -FailureMessage "Prepared version verification failed" -Command {
        & $buildPython -m bixolon_scanner.operations.version_bundle verify `
            --config $configPath `
            --repository-root $repositoryRoot
    }
    Invoke-Native -FailureMessage "OpenVINO Worker build failed" -Command {
        & (Join-Path $PSScriptRoot "build_worker.ps1") `
            -PythonExecutable $buildPython `
            -OutputDirectory $workerOutput `
            -SourceDateEpoch $sourceDateEpoch `
            -OpenVinoLibraryDirectory (
                Join-Path $buildEnvironment "Lib/site-packages/openvino/libs"
            )
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

$workerDist = Join-Path $workerOutputAbsolute "bixolon-worker"
$requiredSources = @(
    (Join-Path $workerDist "bixolon-worker.exe"),
    (Join-Path $workerDist "_internal"),
    (Join-Path $stagingRoot "runtime"),
    (Join-Path $stagingRoot "catalog"),
    (Join-Path $stagingRoot "version.json"),
    (Join-Path $stagingRoot "provenance.json"),
    (Join-Path $repositoryRoot "schemas/scan-response.schema.json"),
    (Join-Path $repositoryRoot "docs/contracts/worker-integration-$Version.md"),
    (Join-Path $repositoryRoot "docs/contracts/flutter-worker-client-example.md"),
    (Join-Path $repositoryRoot "docs/contracts/examples/$Version"),
    (Join-Path $repositoryRoot "scripts/handoff/RUN-COMMANDS.txt")
)
foreach ($source in $requiredSources) {
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required handoff input is missing: $source"
    }
}

[System.IO.Directory]::CreateDirectory($handoffVersionRoot) | Out-Null
foreach ($target in @($packageRoot, $zipPath, $zipHashPath)) {
    if (Test-Path -LiteralPath $target) {
        if (-not $Force) {
            throw "Handoff output already exists; pass -Force to replace it: $target"
        }
        Assert-SafeChildPath -Path $target -Parent $resolvedOutputRoot
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

$temporaryRoot = Join-Path $handoffVersionRoot ("." + [System.IO.Path]::GetRandomFileName())
Assert-SafeChildPath -Path $temporaryRoot -Parent $resolvedOutputRoot
[System.IO.Directory]::CreateDirectory($temporaryRoot) | Out-Null
try {
    $workerTarget = Join-Path $temporaryRoot "worker"
    [System.IO.Directory]::CreateDirectory($workerTarget) | Out-Null
    Copy-Item -LiteralPath (Join-Path $workerDist "bixolon-worker.exe") -Destination $workerTarget
    Copy-Item -LiteralPath (Join-Path $workerDist "_internal") -Destination $workerTarget -Recurse
    Copy-Item -LiteralPath (Join-Path $stagingRoot "runtime") `
        -Destination (Join-Path $workerTarget "model-package") -Recurse
    Copy-Item -LiteralPath (Join-Path $stagingRoot "catalog") `
        -Destination (Join-Path $workerTarget "store-catalog") -Recurse
    Assert-DirectoryCopyMatches `
        -Source (Join-Path $stagingRoot "runtime") `
        -Target (Join-Path $workerTarget "model-package")
    Assert-DirectoryCopyMatches `
        -Source (Join-Path $stagingRoot "catalog") `
        -Target (Join-Path $workerTarget "store-catalog")

    Copy-Item -LiteralPath (Join-Path $repositoryRoot "scripts/handoff/start-worker.ps1") `
        -Destination $temporaryRoot
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "scripts/handoff/benchmark-n100.ps1") `
        -Destination $temporaryRoot
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "scripts/handoff/RUN-COMMANDS.txt") `
        -Destination $temporaryRoot
    if ($null -ne $n100Diagnostic) {
        Copy-Item -LiteralPath $n100DiagnosticPath `
            -Destination (Join-Path $temporaryRoot "n100-reference-result.json")
    }
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "docs/contracts/worker-integration-$Version.md") `
        -Destination (Join-Path $temporaryRoot "API.md")
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "schemas/scan-response.schema.json") `
        -Destination $temporaryRoot
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "docs/contracts/flutter-worker-client-example.md") `
        -Destination (Join-Path $temporaryRoot "FLUTTER-README.md")
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "docs/contracts/examples/$Version") `
        -Destination (Join-Path $temporaryRoot "examples") -Recurse
    Copy-Item -LiteralPath $lockPath -Destination $temporaryRoot

    $flutterLib = Join-Path $temporaryRoot "flutter_example/lib"
    $scannerApiTarget = Join-Path $flutterLib "features/scanner/data"
    $dtoTarget = Join-Path $flutterLib "shared/models"
    [System.IO.Directory]::CreateDirectory($scannerApiTarget) | Out-Null
    [System.IO.Directory]::CreateDirectory($dtoTarget) | Out-Null
    Copy-Item -LiteralPath (
        Join-Path $repositoryRoot "apps/product_scanner/lib/features/scanner/data/scanner_api.dart"
    ) -Destination $scannerApiTarget
    Copy-Item -LiteralPath (
        Join-Path $repositoryRoot "apps/product_scanner/lib/shared/models/scan_models.dart"
    ) -Destination $dtoTarget

    Copy-Item -LiteralPath (Join-Path $stagingRoot "version.json") -Destination $temporaryRoot
    $provenance = Get-Content -Raw -LiteralPath (Join-Path $stagingRoot "provenance.json") |
        ConvertFrom-Json
    $runtimeMetadata = Get-Content -Raw -LiteralPath (Join-Path $stagingRoot "runtime/metadata.json") |
        ConvertFrom-Json
    $defaultProfile = if ($null -eq $n100Diagnostic) {
        [ordered]@{
            detector_workers = 1
            detector_intra_op_threads = 0
            embedder_intra_op_threads = 0
        }
    }
    else {
        [ordered]@{
            detector_workers = [int]$n100Diagnostic.recommended_profile.detector_workers
            detector_intra_op_threads = [int](
                $n100Diagnostic.recommended_profile.detector_threads_per_session
            )
            embedder_intra_op_threads = [int]$n100Diagnostic.recommended_profile.embedder_threads
        }
    }
    $n100Benchmark = if ($null -eq $n100Diagnostic) {
        $null
    }
    else {
        $latencyTargetMs = if ($null -ne $n100Diagnostic.target.full_path_latency_ms) {
            [double]$n100Diagnostic.target.full_path_latency_ms
        }
        else {
            300.0
        }
        $meanWithinTarget = if ($null -ne $n100Diagnostic.target.mean_within_target) {
            [bool]$n100Diagnostic.target.mean_within_target
        }
        else {
            [bool]$n100Diagnostic.target.mean_within_300ms
        }
        $p95WithinTarget = if ($null -ne $n100Diagnostic.target.p95_within_target) {
            [bool]$n100Diagnostic.target.p95_within_target
        }
        else {
            [bool]$n100Diagnostic.target.p95_within_300ms
        }
        [ordered]@{
            reference_result_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $n100DiagnosticPath
            ).Hash.ToLowerInvariant()
            sample_count = [int]$n100Diagnostic.sample_count
            full_path_count = [int]$recommendedN100Profile.full_path_count
            mean_ms = [double]$recommendedN100Profile.latency_ms.mean
            p50_ms = [double]$recommendedN100Profile.latency_ms.p50
            p95_ms = [double]$recommendedN100Profile.latency_ms.p95
            p99_ms = [double]$recommendedN100Profile.latency_ms.p99
            peak_working_set_bytes = [long]$recommendedN100Profile.peak_working_set_bytes
            response_contract_safe = [bool]$n100Diagnostic.response_contract_safe
            latency_target_ms = $latencyTargetMs
            mean_within_target = $meanWithinTarget
            p95_within_target = $p95WithinTarget
            limitation = "Diagnostic measurement only; not an SLA or certification."
        }
    }
    $provenance | Add-Member -NotePropertyName "worker_handoff" -NotePropertyValue ([ordered]@{
        platform = "windows-x64"
        provider = "OpenVINOExecutionProvider:CPU"
        onnxruntime_version = "1.24.1"
        dependency_lock_sha256 = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $lockPath
        ).Hash.ToLowerInvariant()
        default_profile = $defaultProfile
        n100_benchmark_status = if ($null -eq $n100Diagnostic) {
            "PENDING_FIELD_MEASUREMENT"
        } else {
            "MEASURED_DIAGNOSTIC"
        }
        n100_benchmark = $n100Benchmark
        model_graph_or_weight_changed = $false
        decision_policy_changed = $false
    })
    Write-JsonFile -Path (Join-Path $temporaryRoot "provenance.json") -Value $provenance

    $forbiddenFiles = Get-ChildItem -LiteralPath $temporaryRoot -File -Recurse | Where-Object {
        $_.Name -match "(?i)^(onnxruntime_providers_(cuda|tensorrt|dml)\.dll|cudnn.*\.dll|cublas.*\.dll|cudart.*\.dll|cufft.*\.dll|nvrtc.*\.dll|nvjitlink.*\.dll)$" -or
        $_.FullName -match "(?i)[\\/]cuda-runtime[\\/]"
    }
    if ($forbiddenFiles) {
        $paths = ($forbiddenFiles | ForEach-Object { $_.FullName }) -join ", "
        throw "OpenVINO handoff contains an unsupported GPU provider or CUDA file: $paths"
    }

    $files = Get-ChildItem -LiteralPath $temporaryRoot -File -Recurse | Sort-Object FullName |
        ForEach-Object {
            [ordered]@{
                path = Get-RelativePackagePath -Root $temporaryRoot -Path $_.FullName
                size_bytes = $_.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
    $manifest = [ordered]@{
        schema_version = "1.0"
        product_version = $Version
        platform = "windows-x64"
        provider = "OpenVINOExecutionProvider:CPU"
        file_count = $files.Count
        files = @($files)
        self_exclusion = "worker-manifest.json is covered by the external ZIP SHA-256"
    }
    Write-JsonFile -Path (Join-Path $temporaryRoot "worker-manifest.json") -Value $manifest
    [System.IO.Directory]::Move($temporaryRoot, $packageRoot)
}
catch {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
    throw
}

Compress-Archive -LiteralPath $packageRoot -DestinationPath $zipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText(
    $zipHashPath,
    "$zipHash  $([System.IO.Path]::GetFileName($zipPath))" + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "OpenVINO Worker handoff: $packageRoot"
Write-Host "ZIP: $zipPath"
Write-Host "ZIP SHA-256: $zipHash"
