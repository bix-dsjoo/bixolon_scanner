param(
    [string]$Version = "0.1.3",
    [string]$PythonExecutable = "C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe",
    [string]$InnoCompiler = "",
    [string]$VcRedistPath = "",
    [string]$N100DeviceMatrixPath = "",
    [string]$OutputRoot = "artifacts/installers",
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
        throw "Refusing to modify a path outside the installer output: $resolvedPath"
    }
}

function Find-InnoCompiler {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs/Inno Setup 6/ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6/ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6/ISCC.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    $command = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    throw "Inno Setup 6 compiler was not found. Install Inno Setup 6 or pass -InnoCompiler."
}

function Find-VcRedist {
    $roots = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/2022"),
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio/2022")
    )
    $candidates = foreach ($root in $roots) {
        if (Test-Path -LiteralPath $root -PathType Container) {
            Get-ChildItem -LiteralPath $root -Filter "vc_redist.x64.exe" -File -Recurse
        }
    }
    $selected = $candidates | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if (-not $selected) {
        throw (
            "Microsoft Visual C++ 2015-2022 x64 Redistributable was not found. " +
            "Pass its local path with -VcRedistPath."
        )
    }
    return $selected.FullName
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
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Target
    )
    $sourceRecords = Get-DirectoryRecords -Root $Source
    $targetRecords = Get-DirectoryRecords -Root $Target
    if (
        ($sourceRecords | ConvertTo-Json -Depth 4 -Compress) -ne
        ($targetRecords | ConvertTo-Json -Depth 4 -Compress)
    ) {
        throw "Installer payload copy changed a source file: $Target"
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
$openVinoLockPath = Join-Path $repositoryRoot "configs/runtime/requirements-windows-openvino.lock"
$installerScript = Join-Path $repositoryRoot "installer/n100/BixolonScanner-N100.iss"
$installerReadme = Join-Path $repositoryRoot "installer/n100/INSTALL-N100-KO.txt"
$installerLauncher = Join-Path $repositoryRoot "installer/n100/start-bixolon-scanner.ps1"
$workerLauncher = Join-Path $repositoryRoot "installer/n100/start-bixolon-worker.ps1"
$workerCommand = Join-Path $repositoryRoot "installer/n100/RUN-BIXOLON-WORKER.cmd"
$workerReadme = Join-Path $repositoryRoot "installer/n100/WORKER-N100-KO.txt"
if ([string]::IsNullOrWhiteSpace($N100DeviceMatrixPath)) {
    $N100DeviceMatrixPath = Join-Path (
        $repositoryRoot
    ) "docs/diagnostics/n100-$Version-openvino-device-matrix.json"
}
$n100DiagnosticPath = [System.IO.Path]::GetFullPath($N100DeviceMatrixPath)
$setupIconPath = Join-Path $repositoryRoot "apps/product_scanner/windows/runner/resources/app_icon.ico"
$resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$versionOutput = Join-Path $resolvedOutputRoot $Version
$payloadRoot = Join-Path $versionOutput "n100-payload"
$setupPath = Join-Path $versionOutput "BixolonScanner-N100-$Version-Setup.exe"
$setupHashPath = "$setupPath.sha256"
$installerManifestPath = Join-Path $versionOutput "installer-manifest.json"
$workerPackageName = "BixolonScanner-N100-$Version-Worker"
$workerPackageRoot = Join-Path $versionOutput $workerPackageName
$workerZipPath = Join-Path $versionOutput "$workerPackageName.zip"
$workerZipHashPath = "$workerZipPath.sha256"

foreach ($requiredPath in @(
    $configPath,
    $openVinoLockPath,
    $installerScript,
    $installerReadme,
    $installerLauncher,
    $workerLauncher,
    $workerCommand,
    $workerReadme,
    $n100DiagnosticPath,
    $setupIconPath
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required installer source is missing: $requiredPath"
    }
}

$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
if ([string]$config.version -ne $Version) {
    throw "Version config identity mismatch: $configPath"
}
$n100Diagnostic = Get-Content -Raw -LiteralPath $n100DiagnosticPath | ConvertFrom-Json
$selectedN100Profile = $n100Diagnostic.profiles.openvino_cpu_detector_intel_gpu_embedder
if (
    [string]$n100Diagnostic.evaluation -ne (
        "bixolon_worker_n100_openvino_cpu_vs_intel_gpu_embedder"
    ) -or
    -not [bool]$n100Diagnostic.completed -or
    -not [bool]$n100Diagnostic.hardware.target_cpu_detected -or
    -not [bool]$n100Diagnostic.hardware.target_intel_gpu_detected -or
    -not [bool]$n100Diagnostic.execution_contract.same_worker_executable -or
    -not [bool]$n100Diagnostic.execution_contract.same_runtime_catalog_and_policy -or
    [string]$n100Diagnostic.execution_contract.cpu_detector_gpu_embedder.detector -ne (
        "OpenVINOExecutionProvider:CPU"
    ) -or
    [string]$n100Diagnostic.execution_contract.cpu_detector_gpu_embedder.primary_embedder -ne (
        "OpenVINOExecutionProvider:GPU"
    ) -or
    [string]$n100Diagnostic.execution_contract.cpu_detector_gpu_embedder.rotation_180_embedder -ne (
        "OpenVINOExecutionProvider:GPU"
    ) -or
    [string]$n100Diagnostic.execution_contract.cpu_detector_gpu_embedder.independent_verifier_embedder -ne (
        "OpenVINOExecutionProvider:GPU"
    ) -or
    [bool]$n100Diagnostic.execution_contract.cpu_detector_gpu_embedder.silent_cpu_fallback_allowed -or
    [bool]$n100Diagnostic.integrity.model_graph_or_weight_changed -or
    [bool]$n100Diagnostic.integrity.decision_policy_changed -or
    $null -eq $selectedN100Profile -or
    -not [bool]$selectedN100Profile.completed -or
    $null -ne $selectedN100Profile.failure_code -or
    [string]$selectedN100Profile.detector_provider -ne "openvino" -or
    [string]$selectedN100Profile.embedder_provider -ne "openvino_gpu" -or
    [int]$selectedN100Profile.error_count -ne 0 -or
    [int]$selectedN100Profile.full_path_count -lt (
        [int]$n100Diagnostic.input.minimum_full_path_count
    ) -or
    -not [bool]$n100Diagnostic.comparison.provider_initialization_safe -or
    -not [bool]$n100Diagnostic.comparison.hybrid_mean_improved -or
    -not [bool]$n100Diagnostic.comparison.hybrid_p95_improved -or
    [int]$n100Diagnostic.parity.semantic_mismatch_count -ne 0 -or
    [int]$n100Diagnostic.parity.confidence_vector_mismatch_count -ne 0
) {
    throw (
        "N100 device matrix must prove the requested CPU Detector + Intel GPU Embedder " +
        "profile initialized, completed without errors, improved mean and p95, and kept " +
        "semantic output parity."
    )
}
$diagnosticVersion = [string]$n100Diagnostic.product_version
$diagnosticConfigPath = Join-Path $repositoryRoot "configs/versions/$diagnosticVersion.json"
if (-not (Test-Path -LiteralPath $diagnosticConfigPath -PathType Leaf)) {
    throw "N100 reference version config is missing: $diagnosticConfigPath"
}
$diagnosticConfig = Get-Content -Raw -LiteralPath $diagnosticConfigPath | ConvertFrom-Json
if (
    [string]$diagnosticConfig.runtime.manifest_sha256 -ne
        [string]$config.runtime.manifest_sha256 -or
    [string]$diagnosticConfig.catalog.manifest_sha256 -ne
        [string]$config.catalog.manifest_sha256
) {
    throw (
        "A previous-version N100 reference is valid only when the immutable Runtime and " +
        "Catalog source manifests are identical to the selected product version."
    )
}
$launcherSource = Get-Content -Raw -LiteralPath $installerLauncher
if (
    $launcherSource -notmatch 'BIXOLON_PROVIDER = "openvino"' -or
    $launcherSource -notmatch 'BIXOLON_EMBEDDER_PROVIDER = "openvino_gpu"' -or
    $launcherSource -notmatch 'BIXOLON_EMBEDDER_FALLBACK_PROVIDER = "same"' -or
    $launcherSource -notmatch 'BIXOLON_CPU_DETECTOR_WORKERS = "\d+"' -or
    $launcherSource -notmatch 'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "\d+"' -or
    $launcherSource -notmatch 'BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "\d+"'
) {
    throw "N100 launcher is missing the hybrid provider, fallback, or thread settings."
}
$recommendedN100Profile = $selectedN100Profile
$recommendedDetectorWorkers = [int]$selectedN100Profile.detector_workers
$recommendedDetectorThreads = [int]$selectedN100Profile.detector_threads_per_session
$recommendedEmbedderThreads = [int]$selectedN100Profile.embedder_threads
if (
    $recommendedDetectorWorkers -ne 1 -or
    $recommendedDetectorThreads -ne 4 -or
    $recommendedEmbedderThreads -ne 0
) {
    throw "N100 hybrid profile must use Detector 1x4 and GPU Embedder thread auto-selection."
}
$appBuild = [int]$config.app_build
$versionRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $repositoryRoot ([string]$config.output_root + "/" + $Version))
)
$canonicalBundle = Join-Path $versionRoot "bixolon-scanner-$Version"
$openVinoWorker = Join-Path $versionRoot "openvino-gpu-worker-build/bixolon-worker"

if (-not (Test-Path -LiteralPath $canonicalBundle -PathType Container)) {
    throw (
        "Canonical app bundle is missing: $canonicalBundle. " +
        "Run scripts/build_app.ps1 -Version $Version first."
    )
}
if (-not (Test-Path -LiteralPath $openVinoWorker -PathType Container)) {
    throw (
        "OpenVINO Worker build is missing: $openVinoWorker. " +
        "Run scripts/build_n100_gpu_test.ps1 -Version $Version first."
    )
}

$runtimeMetadataPath = Join-Path $canonicalBundle "worker/model-package/metadata.json"
$catalogChecksumsPath = Join-Path $canonicalBundle "worker/store-catalog/checksums.json"
$integrityReferenceBundle = $canonicalBundle
if ($diagnosticVersion -ne $Version) {
    $diagnosticVersionRoot = [System.IO.Path]::GetFullPath(
        (Join-Path $repositoryRoot (
            [string]$diagnosticConfig.output_root + "/" + $diagnosticVersion
        ))
    )
    $integrityReferenceBundle = Join-Path (
        $diagnosticVersionRoot
    ) "bixolon-scanner-$diagnosticVersion"
    if (-not (Test-Path -LiteralPath $integrityReferenceBundle -PathType Container)) {
        throw "N100 reference bundle is missing: $integrityReferenceBundle"
    }
    $runtimeMetadataPath = Join-Path (
        $integrityReferenceBundle
    ) "worker/model-package/metadata.json"
    $catalogChecksumsPath = Join-Path (
        $integrityReferenceBundle
    ) "worker/store-catalog/checksums.json"
}
if (
    (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeMetadataPath).Hash.ToLowerInvariant() -ne
        [string]$n100Diagnostic.integrity.runtime_metadata_sha256 -or
    (Get-FileHash -Algorithm SHA256 -LiteralPath $catalogChecksumsPath).Hash.ToLowerInvariant() -ne
        [string]$n100Diagnostic.integrity.catalog_checksums_sha256
) {
    throw "N100 device matrix does not describe the selected Runtime and Catalog payload."
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $sourceDirectory
    Invoke-Native -FailureMessage "Canonical version bundle verification failed" -Command {
        & $PythonExecutable -m bixolon_scanner.operations.version_bundle verify `
            --config $configPath `
            --repository-root $repositoryRoot
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

$requiredOpenVinoFiles = @(
    (Join-Path $openVinoWorker "bixolon-worker.exe"),
    (Join-Path $openVinoWorker "_internal/onnxruntime/capi/onnxruntime.dll"),
    (Join-Path $openVinoWorker "_internal/onnxruntime/capi/onnxruntime_providers_shared.dll"),
    (Join-Path $openVinoWorker "_internal/onnxruntime/capi/onnxruntime_providers_openvino.dll"),
    (Join-Path $openVinoWorker "_internal/openvino_intel_cpu_plugin.dll"),
    (Join-Path $openVinoWorker "_internal/openvino_intel_gpu_plugin.dll"),
    (Join-Path $openVinoWorker "_internal/openvino_onnx_frontend.dll")
)
foreach ($requiredPath in $requiredOpenVinoFiles) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "OpenVINO Worker file is missing: $requiredPath"
    }
}
$forbiddenOpenVinoFiles = Get-ChildItem -LiteralPath $openVinoWorker -File -Recurse | Where-Object {
    $_.Name -match "(?i)^(onnxruntime_providers_(cuda|tensorrt|dml)\.dll|cudnn.*\.dll|cublas.*\.dll|cudart.*\.dll|cufft.*\.dll|nvrtc.*\.dll|nvjitlink.*\.dll)$"
}
if ($forbiddenOpenVinoFiles) {
    $paths = ($forbiddenOpenVinoFiles | ForEach-Object { $_.FullName }) -join ", "
    throw "OpenVINO hybrid Worker contains a forbidden provider or CUDA runtime: $paths"
}

if (-not $InnoCompiler) {
    $InnoCompiler = Find-InnoCompiler
}
if (-not (Test-Path -LiteralPath $InnoCompiler -PathType Leaf)) {
    throw "Inno Setup compiler is missing: $InnoCompiler"
}
if (-not $VcRedistPath) {
    $VcRedistPath = Find-VcRedist
}
if (-not (Test-Path -LiteralPath $VcRedistPath -PathType Leaf)) {
    throw "Visual C++ Redistributable is missing: $VcRedistPath"
}

[System.IO.Directory]::CreateDirectory($resolvedOutputRoot) | Out-Null
[System.IO.Directory]::CreateDirectory($versionOutput) | Out-Null
foreach ($target in @(
    $payloadRoot,
    $setupPath,
    $setupHashPath,
    $installerManifestPath,
    $workerPackageRoot,
    $workerZipPath,
    $workerZipHashPath
)) {
    if (Test-Path -LiteralPath $target) {
        if (-not $Force) {
            throw "Installer output already exists; pass -Force to replace it: $target"
        }
        Assert-SafeChildPath -Path $target -Parent $resolvedOutputRoot
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

$temporaryPayload = Join-Path $versionOutput ("." + [System.IO.Path]::GetRandomFileName())
Assert-SafeChildPath -Path $temporaryPayload -Parent $resolvedOutputRoot
[System.IO.Directory]::CreateDirectory($temporaryPayload) | Out-Null
try {
    Get-ChildItem -LiteralPath $canonicalBundle -Force | Where-Object {
        $_.Name -notin @("worker", "bundle-manifest.json")
    } | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $temporaryPayload -Recurse
    }

    $workerTarget = Join-Path $temporaryPayload "worker"
    Copy-Item -LiteralPath $openVinoWorker -Destination $workerTarget -Recurse
    Assert-DirectoryCopyMatches `
        -Source $openVinoWorker `
        -Target $workerTarget
    Copy-Item -LiteralPath (Join-Path $canonicalBundle "worker/model-package") `
        -Destination (Join-Path $workerTarget "model-package") -Recurse
    Copy-Item -LiteralPath (Join-Path $canonicalBundle "worker/store-catalog") `
        -Destination (Join-Path $workerTarget "store-catalog") -Recurse
    Copy-Item -LiteralPath $installerReadme `
        -Destination (Join-Path $temporaryPayload "INSTALL-N100-KO.txt")
    $renderedLauncher = [regex]::Replace(
        $launcherSource,
        'BIXOLON_CPU_DETECTOR_WORKERS = "\d+"',
        ('BIXOLON_CPU_DETECTOR_WORKERS = "{0}"' -f $recommendedDetectorWorkers)
    )
    $renderedLauncher = [regex]::Replace(
        $renderedLauncher,
        'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "\d+"',
        ('BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "{0}"' -f $recommendedDetectorThreads)
    )
    $renderedLauncher = [regex]::Replace(
        $renderedLauncher,
        'BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "\d+"',
        ('BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "{0}"' -f $recommendedEmbedderThreads)
    )
    [System.IO.File]::WriteAllText(
        (Join-Path $temporaryPayload "start-bixolon-scanner.ps1"),
        $renderedLauncher,
        [System.Text.UTF8Encoding]::new($false)
    )
    Copy-Item -LiteralPath $n100DiagnosticPath `
        -Destination (Join-Path $temporaryPayload "n100-reference-result.json")

    Assert-DirectoryCopyMatches `
        -Source (Join-Path $canonicalBundle "worker/model-package") `
        -Target (Join-Path $workerTarget "model-package")
    Assert-DirectoryCopyMatches `
        -Source (Join-Path $canonicalBundle "worker/store-catalog") `
        -Target (Join-Path $workerTarget "store-catalog")

    $productExecutable = Join-Path $temporaryPayload "product_scanner.exe"
    $productVersion = [System.Diagnostics.FileVersionInfo]::GetVersionInfo(
        $productExecutable
    ).ProductVersion
    if ($productVersion -ne $Version) {
        throw "Windows ProductVersion does not match ${Version}: $productVersion"
    }

    $canonicalManifest = Join-Path $canonicalBundle "bundle-manifest.json"
    $n100Provenance = [ordered]@{
        schema_version = "1.0"
        version = $Version
        app_build = $appBuild
        target = [ordered]@{
            platform = "windows-x64"
            processor_profile = "Intel Processor N100"
            detector_provider = "OpenVINOExecutionProvider:CPU"
            embedder_provider = "OpenVINOExecutionProvider:GPU"
            embedder_fallback_provider = "OpenVINOExecutionProvider:CPU"
            python_required_on_target = $false
            flutter_required_on_target = $false
            cuda_required_on_target = $false
        }
        default_profile = [ordered]@{
            detector_workers = $recommendedDetectorWorkers
            detector_intra_op_threads = $recommendedDetectorThreads
            embedder_intra_op_threads = $recommendedEmbedderThreads
            provider_selection = "gpu_embedder_then_explicit_cpu_fallback"
            request_timeout_seconds = 60
        }
        source = [ordered]@{
            canonical_bundle_manifest_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $canonicalManifest
            ).Hash.ToLowerInvariant()
            openvino_dependency_lock_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $openVinoLockPath
            ).Hash.ToLowerInvariant()
            vc_redist_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $VcRedistPath
            ).Hash.ToLowerInvariant()
            n100_device_matrix_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $n100DiagnosticPath
            ).Hash.ToLowerInvariant()
        }
        transformation = [ordered]@{
            model_graph_or_weight_changed = $false
            decision_policy_changed = $false
            runtime_or_catalog_payload_changed = $false
            cuda_runtime_included = $false
            openvino_gpu_plugin_included = $true
            hybrid_worker_substituted = $true
        }
        diagnostics = [ordered]@{
            n100_benchmark_status = if ($diagnosticVersion -eq $Version) {
                "MEASURED_DIAGNOSTIC"
            } else {
                "REFERENCE_MEASURED_DIAGNOSTIC"
            }
            reference_product_version = $diagnosticVersion
            sample_count = [int]$n100Diagnostic.input.sample_count
            full_path_count = [int]$recommendedN100Profile.full_path_count
            mean_ms = [double]$recommendedN100Profile.client_total_ms.full_path.mean
            p50_ms = [double]$recommendedN100Profile.client_total_ms.full_path.p50
            p95_ms = [double]$recommendedN100Profile.client_total_ms.full_path.p95
            p99_ms = [double]$recommendedN100Profile.client_total_ms.full_path.p99
            error_count = [int]$recommendedN100Profile.error_count
            semantic_mismatch_count = [int]$n100Diagnostic.parity.semantic_mismatch_count
            maximum_confidence_delta = [double]$n100Diagnostic.parity.maximum_confidence_delta
            confidence_tolerance = [double]$n100Diagnostic.parity.confidence_tolerance
            cpu_only_mean_speedup_ratio = [double](
                $n100Diagnostic.comparison.full_path_mean_speedup_ratio
            )
            cpu_only_p95_speedup_ratio = [double](
                $n100Diagnostic.comparison.full_path_p95_speedup_ratio
            )
            peak_working_set_bytes = [long]$recommendedN100Profile.peak_working_set_bytes
            n100_latency_target_applied = $false
            latency_or_sla_claimed = $false
        }
        distribution = [ordered]@{
            payload_integrity = "SHA-256"
            publisher_authentication = "UNSIGNED"
        }
    }
    Write-JsonFile `
        -Path (Join-Path $temporaryPayload "n100-provenance.json") `
        -Value $n100Provenance

    $payloadRecords = Get-DirectoryRecords -Root $temporaryPayload
    $payloadManifest = [ordered]@{
        schema_version = "1.0"
        version = $Version
        app_build = $appBuild
        target = "windows-x64-n100-openvino-cpu-detector-gpu-embedder"
        file_count = $payloadRecords.Count
        files = $payloadRecords
    }
    Write-JsonFile `
        -Path (Join-Path $temporaryPayload "installer-payload-manifest.json") `
        -Value $payloadManifest

    $gpuPayloadFiles = Get-ChildItem -LiteralPath $temporaryPayload -File -Recurse | Where-Object {
        $_.FullName -match "(?i)[\\/]cuda-runtime[\\/]" -or
        $_.Name -match "(?i)^(onnxruntime_providers_(cuda|tensorrt|dml)\.dll|cudnn.*\.dll|cublas.*\.dll|cudart.*\.dll|cufft.*\.dll|nvrtc.*\.dll|nvjitlink.*\.dll)$"
    }
    if ($gpuPayloadFiles) {
        $paths = ($gpuPayloadFiles | ForEach-Object { $_.FullName }) -join ", "
        throw "N100 installer payload contains a forbidden provider or CUDA runtime: $paths"
    }

    [System.IO.Directory]::Move($temporaryPayload, $payloadRoot)
}
catch {
    if (Test-Path -LiteralPath $temporaryPayload) {
        Remove-Item -LiteralPath $temporaryPayload -Recurse -Force
    }
    throw
}

$temporaryWorkerPackage = Join-Path $versionOutput (
    ".worker-" + [System.IO.Path]::GetRandomFileName()
)
Assert-SafeChildPath -Path $temporaryWorkerPackage -Parent $resolvedOutputRoot
[System.IO.Directory]::CreateDirectory($temporaryWorkerPackage) | Out-Null
try {
    Copy-Item -LiteralPath (Join-Path $payloadRoot "worker") `
        -Destination (Join-Path $temporaryWorkerPackage "worker") -Recurse
    Assert-DirectoryCopyMatches `
        -Source (Join-Path $payloadRoot "worker") `
        -Target (Join-Path $temporaryWorkerPackage "worker")
    Copy-Item -LiteralPath $workerLauncher `
        -Destination (Join-Path $temporaryWorkerPackage "start-bixolon-worker.ps1")
    Copy-Item -LiteralPath $workerCommand -Destination $temporaryWorkerPackage
    Copy-Item -LiteralPath $workerReadme -Destination $temporaryWorkerPackage
    Copy-Item -LiteralPath $openVinoLockPath -Destination $temporaryWorkerPackage
    Copy-Item -LiteralPath $VcRedistPath `
        -Destination (Join-Path $temporaryWorkerPackage "vc_redist.x64.exe")
    Copy-Item -LiteralPath (Join-Path $payloadRoot "version.json") `
        -Destination $temporaryWorkerPackage
    Copy-Item -LiteralPath (Join-Path $payloadRoot "n100-provenance.json") `
        -Destination $temporaryWorkerPackage
    Copy-Item -LiteralPath (Join-Path $payloadRoot "n100-reference-result.json") `
        -Destination $temporaryWorkerPackage
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "schemas/scan-response.schema.json") `
        -Destination $temporaryWorkerPackage
    Copy-Item -LiteralPath (
        Join-Path $repositoryRoot "docs/contracts/worker-integration-$Version.md"
    ) -Destination (Join-Path $temporaryWorkerPackage "API.md")

    $workerRecords = Get-DirectoryRecords -Root $temporaryWorkerPackage
    $workerManifest = [ordered]@{
        schema_version = "1.0"
        product_version = $Version
        target = "windows-x64-n100-openvino-cpu-detector-gpu-embedder"
        default_provider = [ordered]@{
            detector = "OpenVINOExecutionProvider:CPU"
            embedder = "OpenVINOExecutionProvider:GPU"
            embedder_fallback = "OpenVINOExecutionProvider:CPU"
        }
        file_count = $workerRecords.Count
        files = $workerRecords
        self_exclusion = "worker-manifest.json is covered by the external ZIP SHA-256"
    }
    Write-JsonFile `
        -Path (Join-Path $temporaryWorkerPackage "worker-manifest.json") `
        -Value $workerManifest
    [System.IO.Directory]::Move($temporaryWorkerPackage, $workerPackageRoot)
}
catch {
    if (Test-Path -LiteralPath $temporaryWorkerPackage) {
        Remove-Item -LiteralPath $temporaryWorkerPackage -Recurse -Force
    }
    throw
}

Compress-Archive `
    -Path (Join-Path $workerPackageRoot "*") `
    -DestinationPath $workerZipPath `
    -CompressionLevel Optimal
$workerZipHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $workerZipPath
).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText(
    $workerZipHashPath,
    "$workerZipHash  $([System.IO.Path]::GetFileName($workerZipPath))" + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Invoke-Native -FailureMessage "Inno Setup compilation failed" -Command {
    & $InnoCompiler `
        "/DAppVersion=$Version" `
        "/DPayloadDir=$payloadRoot" `
        "/DOutputDir=$versionOutput" `
        "/DSetupIconPath=$setupIconPath" `
        "/DVcRedistPath=$VcRedistPath" `
        $installerScript
}

if (-not (Test-Path -LiteralPath $setupPath -PathType Leaf)) {
    throw "Installer executable was not created: $setupPath"
}
$setupFile = Get-Item -LiteralPath $setupPath
$setupHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $setupPath).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText(
    $setupHashPath,
    "$setupHash  $($setupFile.Name)" + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)
$installerManifest = [ordered]@{
    schema_version = "1.0"
    version = $Version
    target = "windows-x64-n100-openvino-cpu-detector-gpu-embedder"
    setup = [ordered]@{
        filename = $setupFile.Name
        size_bytes = $setupFile.Length
        sha256 = $setupHash
        authenticode = "UNSIGNED"
    }
    worker_zip = [ordered]@{
        filename = [System.IO.Path]::GetFileName($workerZipPath)
        size_bytes = (Get-Item -LiteralPath $workerZipPath).Length
        sha256 = $workerZipHash
    }
    payload_manifest_sha256 = (
        Get-FileHash -Algorithm SHA256 `
            -LiteralPath (Join-Path $payloadRoot "installer-payload-manifest.json")
    ).Hash.ToLowerInvariant()
    vc_redist_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $VcRedistPath
    ).Hash.ToLowerInvariant()
}
Write-JsonFile -Path $installerManifestPath -Value $installerManifest

Write-Host "N100 installer: $setupPath"
Write-Host "Installer SHA-256: $setupHash"
Write-Host "SHA-256 file: $setupHashPath"
Write-Host "Installer manifest: $installerManifestPath"
Write-Host "N100 hybrid Worker ZIP: $workerZipPath"
Write-Host "Worker ZIP SHA-256: $workerZipHash"
