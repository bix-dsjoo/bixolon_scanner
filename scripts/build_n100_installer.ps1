param(
    [string]$Version = "0.1.2",
    [string]$PythonExecutable = "C:/Users/OMEN/AppData/Local/Programs/Python/Python311/python.exe",
    [string]$InnoCompiler = "",
    [string]$VcRedistPath = "",
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
$n100DiagnosticPath = Join-Path $repositoryRoot "docs/diagnostics/n100-worker-$Version.json"
$setupIconPath = Join-Path $repositoryRoot "apps/product_scanner/windows/runner/resources/app_icon.ico"
$resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$versionOutput = Join-Path $resolvedOutputRoot $Version
$payloadRoot = Join-Path $versionOutput "n100-payload"
$setupPath = Join-Path $versionOutput "BixolonScanner-N100-$Version-Setup.exe"
$setupHashPath = "$setupPath.sha256"
$installerManifestPath = Join-Path $versionOutput "installer-manifest.json"

foreach ($requiredPath in @(
    $configPath,
    $openVinoLockPath,
    $installerScript,
    $installerReadme,
    $installerLauncher,
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
if (
    [string]$n100Diagnostic.product_version -ne $Version -or
    [string]$n100Diagnostic.provider -ne "cpu" -or
    -not [bool]$n100Diagnostic.passes -or
    -not [bool]$n100Diagnostic.response_contract_safe -or
    -not [bool]$n100Diagnostic.hardware.target_cpu_detected -or
    -not [bool]$n100Diagnostic.target.mean_within_1_second -or
    -not [bool]$n100Diagnostic.target.p95_within_1_second
) {
    throw (
        "N100 diagnostic must be from an N100 and pass response, error, mean <= 1000ms, " +
        "and p95 <= 1000ms checks."
    )
}
$launcherSource = Get-Content -Raw -LiteralPath $installerLauncher
if (
    $launcherSource -notmatch 'BIXOLON_PROVIDER = "cpu"' -or
    $launcherSource -notmatch 'BIXOLON_CPU_DETECTOR_WORKERS = "\d+"' -or
    $launcherSource -notmatch 'BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "\d+"' -or
    $launcherSource -notmatch 'BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "\d+"'
) {
    throw "N100 launcher is missing CPU provider or one or more thread settings."
}
$recommendedN100Profiles = @(
    $n100Diagnostic.profiles |
        Where-Object { $_.name -eq $n100Diagnostic.recommended_profile.name }
)
if ($recommendedN100Profiles.Count -ne 1) {
    throw "N100 diagnostic must contain exactly one recommended profile result."
}
$recommendedN100Profile = $recommendedN100Profiles[0]
$recommendedDetectorWorkers = [int]$n100Diagnostic.recommended_profile.detector_workers
$recommendedDetectorThreads = [int](
    $n100Diagnostic.recommended_profile.detector_threads_per_session
)
$recommendedEmbedderThreads = [int]$n100Diagnostic.recommended_profile.embedder_threads
if (
    $recommendedDetectorWorkers -lt 1 -or
    $recommendedDetectorThreads -lt 1 -or
    $recommendedEmbedderThreads -lt 1 -or
    [int]$recommendedN100Profile.detector_workers -ne $recommendedDetectorWorkers -or
    [int]$recommendedN100Profile.detector_threads_per_session -ne $recommendedDetectorThreads -or
    [int]$recommendedN100Profile.embedder_threads -ne $recommendedEmbedderThreads -or
    [string]$recommendedN100Profile.provider -ne "cpu" -or
    [string]$n100Diagnostic.recommended_profile.provider -ne "cpu" -or
    [int]$recommendedN100Profile.error_count -ne 0
) {
    throw "N100 recommended profile is invalid or does not match its measured profile."
}
$appBuild = [int]$config.app_build
$versionRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $repositoryRoot ([string]$config.output_root + "/" + $Version))
)
$canonicalBundle = Join-Path $versionRoot "bixolon-scanner-$Version"
$openVinoWorker = Join-Path $versionRoot "openvino-worker-build/bixolon-worker"

if (-not (Test-Path -LiteralPath $canonicalBundle -PathType Container)) {
    throw (
        "Canonical app bundle is missing: $canonicalBundle. " +
        "Run scripts/build_app.ps1 -Version $Version first."
    )
}
if (-not (Test-Path -LiteralPath $openVinoWorker -PathType Container)) {
    throw (
        "OpenVINO Worker build is missing: $openVinoWorker. " +
        "Run scripts/build_worker_handoff.ps1 -Version $Version first."
    )
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
    throw "OpenVINO Worker contains an unsupported GPU provider or CUDA runtime: $paths"
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
foreach ($target in @($payloadRoot, $setupPath, $setupHashPath, $installerManifestPath)) {
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
            provider = "CPUExecutionProvider"
            python_required_on_target = $false
            flutter_required_on_target = $false
            cuda_required_on_target = $false
        }
        default_profile = [ordered]@{
            detector_workers = $recommendedDetectorWorkers
            detector_intra_op_threads = $recommendedDetectorThreads
            embedder_intra_op_threads = $recommendedEmbedderThreads
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
            n100_reference_result_sha256 = (
                Get-FileHash -Algorithm SHA256 -LiteralPath $n100DiagnosticPath
            ).Hash.ToLowerInvariant()
        }
        transformation = [ordered]@{
            model_graph_or_weight_changed = $false
            decision_policy_changed = $false
            runtime_or_catalog_payload_changed = $false
            cuda_runtime_included = $false
            cpu_worker_substituted = $true
        }
        diagnostics = [ordered]@{
            n100_benchmark_status = "MEASURED_DIAGNOSTIC"
            sample_count = [int]$n100Diagnostic.sample_count
            full_path_count = [int]$recommendedN100Profile.full_path_count
            mean_ms = [double]$recommendedN100Profile.latency_ms.mean
            p50_ms = [double]$recommendedN100Profile.latency_ms.p50
            p95_ms = [double]$recommendedN100Profile.latency_ms.p95
            p99_ms = [double]$recommendedN100Profile.latency_ms.p99
            mean_within_1_second = [bool]$n100Diagnostic.target.mean_within_1_second
            p95_within_1_second = [bool]$n100Diagnostic.target.p95_within_1_second
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
        target = "windows-x64-n100-cpu"
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
        throw "N100 installer payload contains GPU runtime files: $paths"
    }

    [System.IO.Directory]::Move($temporaryPayload, $payloadRoot)
}
catch {
    if (Test-Path -LiteralPath $temporaryPayload) {
        Remove-Item -LiteralPath $temporaryPayload -Recurse -Force
    }
    throw
}

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
    target = "windows-x64-n100-cpu"
    setup = [ordered]@{
        filename = $setupFile.Name
        size_bytes = $setupFile.Length
        sha256 = $setupHash
        authenticode = "UNSIGNED"
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
