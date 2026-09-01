param(
    [string]$PythonExecutable = "artifacts/build-envs/worker-openvino-gpu-py311/Scripts/python.exe",
    [string]$RuntimeDirectory = "",
    [string]$CatalogDirectory = "",
    [string]$OutputRoot = "artifacts/handoff",
    [switch]$ReuseWorker,
    [switch]$Force,
    [switch]$Ssdlite,
    [switch]$ClassAgnostic,
    [switch]$DetectorPrimary
)

$ErrorActionPreference = "Stop"

$selectedModeCount = @($Ssdlite, $ClassAgnostic, $DetectorPrimary).Where({ $_ }).Count
if ($selectedModeCount -gt 1) {
    throw "Choose only one diagnostic Runtime mode."
}
$expectedVersion = if ($ClassAgnostic -or $DetectorPrimary) { "0.1.11" } else { "0.1.7" }
if ([string]::IsNullOrWhiteSpace($RuntimeDirectory)) {
    $RuntimeDirectory = if ($DetectorPrimary) {
        "artifacts/experiments/detector-primary-0.1.11/runtime-candidate-v2-0.1.11-selective-dino"
    } elseif ($ClassAgnostic) {
        "artifacts/experiments/class-agnostic-detector-0.1.11/runtime-candidate-v6-dense-fallback"
    } elseif ($Ssdlite) {
        "artifacts/experiments/yolo-free-0.1.7/runtime-candidate-v42-ssdlite320-selective-fallback"
    } else {
        "artifacts/experiments/yolo-free-0.1.7/runtime-candidate-v16-primary192-verifier160"
    }
}
if ([string]::IsNullOrWhiteSpace($CatalogDirectory)) {
    $CatalogDirectory = "artifacts/versions/$expectedVersion/staging/catalog"
}

function Assert-SafeChildPath {
    param([string]$Path, [string]$Parent)
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

function Invoke-Native {
    param([scriptblock]$Command, [string]$FailureMessage)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)."
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
    param([string]$Root)
    return @(
        Get-ChildItem -LiteralPath $Root -File -Recurse | Sort-Object FullName |
            ForEach-Object {
                [ordered]@{
                    path = Get-RelativePackagePath -Root $Root -Path $_.FullName
                    size_bytes = $_.Length
                    sha256 = (
                        Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
                    ).Hash.ToLowerInvariant()
                }
            }
    )
}

function Write-JsonFile {
    param([string]$Path, [object]$Value)
    [System.IO.File]::WriteAllText(
        $Path,
        ($Value | ConvertTo-Json -Depth 20) + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$sourceDirectory = Join-Path $repositoryRoot "src"
$resolvedPython = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $PythonExecutable))
$resolvedRuntime = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $RuntimeDirectory))
$resolvedCatalog = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $CatalogDirectory))
$resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$experimentRelative = if ($DetectorPrimary) {
    "artifacts/experiments/detector-primary-0.1.11"
} elseif ($ClassAgnostic) {
    "artifacts/experiments/class-agnostic-detector-0.1.11"
} else {
    "artifacts/experiments/yolo-free-0.1.7"
}
$experimentRoot = Join-Path $repositoryRoot $experimentRelative
$workerOutputName = if ($DetectorPrimary) {
    "openvino-gpu-worker-build-v2"
} elseif ($ClassAgnostic) {
    "openvino-gpu-worker-build-v6"
} elseif ($Ssdlite) {
    "openvino-gpu-worker-build-ssdlite"
} else {
    "openvino-gpu-worker-build"
}
$workerOutput = Join-Path $experimentRoot $workerOutputName
$workerDist = Join-Path $workerOutput "bixolon-worker"
$packageName = if ($DetectorPrimary) {
    "n100-detector-primary-0.1.11-openvino-cpu-gpu-test"
} elseif ($ClassAgnostic) {
    "n100-class-agnostic-0.1.11-openvino-cpu-gpu-test"
} elseif ($Ssdlite) {
    "n100-ssdlite-0.1.7-openvino-cpu-gpu-test"
} else {
    "n100-yolo-free-0.1.7-openvino-cpu-gpu-test"
}
$packageRoot = Join-Path $resolvedOutputRoot $packageName
$zipPath = "$packageRoot.zip"
$zipHashPath = "$zipPath.sha256"
$benchmarkScript = Join-Path $repositoryRoot "scripts/handoff/N100-GPU-BENCHMARK.ps1"
$commandRelativePath = if ($DetectorPrimary) {
    "scripts/handoff/RUN-DETECTOR-PRIMARY-N100-GPU-TEST.cmd"
} elseif ($ClassAgnostic) {
    "scripts/handoff/RUN-CLASS-AGNOSTIC-N100-GPU-TEST.cmd"
} elseif ($Ssdlite) {
    "scripts/handoff/RUN-SSDLITE-N100-GPU-TEST.cmd"
} else {
    "scripts/handoff/RUN-YOLO-FREE-N100-GPU-TEST.cmd"
}
$commandScript = Join-Path $repositoryRoot $commandRelativePath
$readmeRelativePath = if ($DetectorPrimary) {
    "scripts/handoff/README-DETECTOR-PRIMARY-N100-KO.txt"
} elseif ($ClassAgnostic) {
    "scripts/handoff/README-CLASS-AGNOSTIC-N100-KO.txt"
} elseif ($Ssdlite) {
    "scripts/handoff/README-SSDLITE-N100-KO.txt"
} else {
    "scripts/handoff/README-YOLO-FREE-N100-KO.txt"
}
$readmePath = Join-Path $repositoryRoot $readmeRelativePath
$openVinoLibraries = Join-Path (
    Split-Path -Parent (Split-Path -Parent $resolvedPython)
) "Lib/site-packages/openvino/libs"

foreach ($required in @(
    $resolvedPython,
    (Join-Path $resolvedRuntime "metadata.json"),
    (Join-Path $resolvedCatalog "catalog.json"),
    $benchmarkScript,
    $commandScript,
    $readmePath,
    $openVinoLibraries
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required YOLO-free N100 input is missing: $required"
    }
}

$runtimeMetadata = Get-Content -Raw -LiteralPath (
    Join-Path $resolvedRuntime "metadata.json"
) | ConvertFrom-Json
$catalogMetadata = Get-Content -Raw -LiteralPath (
    Join-Path $resolvedCatalog "catalog.json"
) | ConvertFrom-Json
if (
    [string]$runtimeMetadata.worker_version -ne $expectedVersion -or
    [string]$catalogMetadata.catalog_version -ne $expectedVersion -or
    [int]$runtimeMetadata.embedder.input_size[0] -ne 192 -or
    [int]$runtimeMetadata.classifier_resolution_fallback.embedder.input_size[0] -ne 224 -or
    [int]$runtimeMetadata.classifier_verification.independent_embedder.input_size[0] -ne 160
) {
    throw "The selected Runtime/Catalog version or classifier cascade is invalid."
}
if ($DetectorPrimary) {
    $directClasses = @(
        $runtimeMetadata.detector_primary_classifier_routing.direct_approval_class_indices |
            ForEach-Object { [int]$_ }
    )
    $expectedDirectClasses = @(2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 17, 18)
    if (
        (
            $null -ne $runtimeMetadata.detector_class_mode -and
            [string]$runtimeMetadata.detector_class_mode -ne "class_aware"
        ) -or
        [int]$runtimeMetadata.detector_class_count -ne 20 -or
        $null -ne $runtimeMetadata.detector.ensemble -or
        [int]$runtimeMetadata.detector.input_size[0] -ne 320 -or
        $null -ne $runtimeMetadata.count_verifier -or
        $runtimeMetadata.classifier_resolution_fallback.selective_roi_only -ne $true -or
        $directClasses.Count -ne $expectedDirectClasses.Count -or
        (Compare-Object $directClasses $expectedDirectClasses) -or
        [double]$runtimeMetadata.detector_primary_classifier_routing.minimum_detector_score -ne 0.98 -or
        $runtimeMetadata.detector_primary_classifier_routing.require_unique_class_per_image -ne $true
    ) {
        throw "The selected Runtime is not the verified detector-primary 0.1.11 candidate."
    }
}
elseif ($ClassAgnostic) {
    $fallbackRules = @($runtimeMetadata.classifier_resolution_fallback.approval_disagreement_rules)
    if (
        [string]$runtimeMetadata.detector_class_mode -ne "class_agnostic" -or
        [int]$runtimeMetadata.detector_class_count -ne 1 -or
        $null -ne $runtimeMetadata.detector.ensemble -or
        [int]$runtimeMetadata.detector.input_size[0] -ne 320 -or
        $null -ne $runtimeMetadata.count_verifier -or
        $null -ne $runtimeMetadata.quality.detector_classifier_consensus -or
        $runtimeMetadata.classifier_resolution_fallback.selective_roi_only -ne $true -or
        $runtimeMetadata.classifier_resolution_fallback.fuse_unapproved_top3 -ne $true -or
        [double]$runtimeMetadata.classifier_resolution_fallback.minimum_fallback_approval_score -ne 0.08 -or
        $fallbackRules.Count -ne 2 -or
        [int]$fallbackRules[0].minimum_detection_count -ne 6 -or
        [double]$fallbackRules[0].maximum_approval_score -ne 0.76 -or
        $fallbackRules[0].require_detector_disagreement -ne $false -or
        [double]$fallbackRules[1].minimum_box_aspect_ratio -ne 3.0 -or
        [double]$fallbackRules[1].maximum_approval_score_decrease -ne 0.2 -or
        [string]$runtimeMetadata.sources.detector.architecture -notmatch "class-agnostic"
    ) {
        throw "The selected Runtime is not the verified class-agnostic 0.1.11 candidate."
    }
}
elseif ($Ssdlite) {
    if (
        $null -ne $runtimeMetadata.detector.ensemble -or
        [int]$runtimeMetadata.detector.input_size[0] -ne 320 -or
        $null -ne $runtimeMetadata.count_verifier -or
        $runtimeMetadata.classifier_resolution_fallback.selective_roi_only -ne $true -or
        $null -eq $runtimeMetadata.quality.detector_classifier_consensus -or
        [string]$runtimeMetadata.sources.detector.architecture -notmatch "SSDLite320"
    ) {
        throw "The selected Runtime is not the verified SSDLite320 0.1.7 candidate."
    }
}
elseif (
    [string]$runtimeMetadata.detector.ensemble.class_verified_selector.low_resolution_primary_member_filename -ne "detector-production.onnx" -or
    [string]$runtimeMetadata.count_verifier.comparison_mode -ne "exact_count"
) {
    throw "The selected Runtime is not the verified D-FINE 0.1.7 candidate."
}

Invoke-Native -FailureMessage "OpenVINO provider validation failed" -Command {
    & $resolvedPython -c (
        "import onnxruntime as ort; " +
        "p=set(ort.get_available_providers()); " +
        "assert {'OpenVINOExecutionProvider','CPUExecutionProvider'} <= p, p; " +
        "assert p.isdisjoint({'CUDAExecutionProvider','TensorrtExecutionProvider','DmlExecutionProvider'}), p"
    )
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $sourceDirectory
    Invoke-Native -FailureMessage "Runtime/Catalog validation failed" -Command {
        & $resolvedPython -c (
            "from pathlib import Path; " +
            "from bixolon_scanner.contracts.runtime_package_v2 import load_runtime_package_v2; " +
            "from bixolon_scanner.contracts.catalog import load_store_catalog_package; " +
            "load_runtime_package_v2(Path(r'$($resolvedRuntime.Replace("'", "''"))')); " +
            "load_store_catalog_package(Path(r'$($resolvedCatalog.Replace("'", "''"))'), expected_store_id='bread-dev')"
        )
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

if (-not $ReuseWorker) {
    if (Test-Path -LiteralPath $workerOutput) {
        if (-not $Force) {
            throw "Worker output exists; pass -ReuseWorker or -Force: $workerOutput"
        }
        Assert-SafeChildPath -Path $workerOutput -Parent $experimentRoot
        Remove-Item -LiteralPath $workerOutput -Recurse -Force
    }
    Invoke-Native -FailureMessage "OpenVINO GPU Worker build failed" -Command {
        & (Join-Path $PSScriptRoot "build_worker.ps1") `
            -PythonExecutable $resolvedPython `
            -OutputDirectory "$experimentRelative/$workerOutputName" `
            -OpenVinoLibraryDirectory $openVinoLibraries `
            -IncludeOpenVinoGpu
    }
}

$requiredWorkerFiles = @(
    (Join-Path $workerDist "bixolon-worker.exe"),
    (Join-Path $workerDist "_internal/onnxruntime/capi/onnxruntime.dll"),
    (Join-Path $workerDist "_internal/onnxruntime/capi/onnxruntime_providers_openvino.dll"),
    (Join-Path $workerDist "_internal/openvino_intel_cpu_plugin.dll"),
    (Join-Path $workerDist "_internal/openvino_intel_gpu_plugin.dll")
)
foreach ($required in $requiredWorkerFiles) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Packaged Worker dependency is missing: $required"
    }
}
$forbidden = Get-ChildItem -LiteralPath $workerDist -File -Recurse | Where-Object {
    $_.Name -match "(?i)(directml|cuda|cudnn|cublas|cudart|cufft|nvrtc|tensorrt)"
}
if ($forbidden) {
    throw "Worker contains an unintended GPU runtime: $($forbidden[0].FullName)"
}

[System.IO.Directory]::CreateDirectory($resolvedOutputRoot) | Out-Null
foreach ($target in @($packageRoot, $zipPath, $zipHashPath)) {
    if (Test-Path -LiteralPath $target) {
        if (-not $Force) {
            throw "Output exists; pass -Force to replace it: $target"
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
    Copy-Item -LiteralPath $resolvedRuntime -Destination (
        Join-Path $workerTarget "model-package"
    ) -Recurse
    Copy-Item -LiteralPath $resolvedCatalog -Destination (
        Join-Path $workerTarget "store-catalog"
    ) -Recurse
    Copy-Item -LiteralPath $benchmarkScript, $commandScript, $readmePath -Destination $temporaryRoot

    $runtimeRecords = Get-DirectoryRecords -Root $resolvedRuntime
    $catalogRecords = Get-DirectoryRecords -Root $resolvedCatalog
    $manifest = [ordered]@{
        schema_version = "1.0"
        artifact = if ($DetectorPrimary) {
            "detector_primary_0.1.11_n100_openvino_cpu_gpu_diagnostic"
        } elseif ($ClassAgnostic) {
            "class_agnostic_0.1.11_n100_openvino_cpu_gpu_diagnostic"
        } elseif ($Ssdlite) {
            "ssdlite_0.1.7_n100_openvino_cpu_gpu_diagnostic"
        } else {
            "yolo_free_0.1.7_n100_openvino_cpu_gpu_diagnostic"
        }
        product_version = $expectedVersion
        target_hardware = "Intel Processor N100 with Intel UHD Graphics"
        target_full_path_mean_ms = 1000
        target_full_path_p95_ms = 1000
        provider_contract = [ordered]@{
            baseline = if ($Ssdlite -or $ClassAgnostic -or $DetectorPrimary) {
                "OpenVINO CPU detector + CPU classifiers"
            } else {
                "OpenVINO CPU detector + CPU count verifier/classifiers"
            }
            candidate = if ($Ssdlite -or $ClassAgnostic -or $DetectorPrimary) {
                "OpenVINO CPU detector + Intel GPU classifiers"
            } else {
                "OpenVINO CPU detector + Intel GPU count verifier/classifiers"
            }
            silent_cpu_fallback_allowed = $false
        }
        transformation = [ordered]@{
            checkpoint_weights_changed = $false
            model_graph_changed = -not $DetectorPrimary
            runtime_policy_changed = $true
            catalog_payload_changed = $false
            diagnostic_packaging_only = $true
        }
        runtime = [ordered]@{ file_count = $runtimeRecords.Count; files = $runtimeRecords }
        catalog = [ordered]@{ file_count = $catalogRecords.Count; files = $catalogRecords }
        limitation = "Actual N100 measurement is required; this package is not a release artifact."
    }
    Write-JsonFile -Path (Join-Path $temporaryRoot "candidate-manifest.json") -Value $manifest
    $packageRecords = Get-DirectoryRecords -Root $temporaryRoot
    Write-JsonFile -Path (Join-Path $temporaryRoot "package-manifest.json") -Value ([ordered]@{
        schema_version = "1.0"
        artifact = if ($DetectorPrimary) {
            "detector_primary_0.1.11_n100_openvino_cpu_gpu_diagnostic_payload"
        } elseif ($ClassAgnostic) {
            "class_agnostic_0.1.11_n100_openvino_cpu_gpu_diagnostic_payload"
        } elseif ($Ssdlite) {
            "ssdlite_0.1.7_n100_openvino_cpu_gpu_diagnostic_payload"
        } else {
            "yolo_free_0.1.7_n100_openvino_cpu_gpu_diagnostic_payload"
        }
        file_count = $packageRecords.Count
        files = $packageRecords
    })
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

Write-Host "N100 diagnostic package: $packageRoot"
Write-Host "ZIP: $zipPath"
Write-Host "ZIP SHA-256: $zipHash"
