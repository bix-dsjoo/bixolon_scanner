param(
    [string]$Version = "0.0.2",
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
                Path = [System.IO.Path]::GetRelativePath($Source, $_.FullName).Replace("\", "/")
                Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash
            }
        }
    )
    $targetFiles = @(
        Get-ChildItem -LiteralPath $Target -File -Recurse | Sort-Object FullName | ForEach-Object {
            [ordered]@{
                Path = [System.IO.Path]::GetRelativePath($Target, $_.FullName).Replace("\", "/")
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
$lockPath = Join-Path $repositoryRoot "configs/runtime/requirements-windows-cpu.lock"
$resolvedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputRoot))
$handoffVersionRoot = Join-Path $resolvedOutputRoot $Version
$packageName = "bixolon-worker-$Version-windows-x64-cpu"
$packageRoot = Join-Path $handoffVersionRoot $packageName
$zipPath = Join-Path $handoffVersionRoot "$packageName.zip"
$zipHashPath = "$zipPath.sha256"
$buildEnvironment = Join-Path $repositoryRoot "artifacts/build-envs/worker-cpu-py311"
$workerOutput = "artifacts/versions/$Version/cpu-worker-build"
$workerOutputAbsolute = Join-Path $repositoryRoot $workerOutput

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Version config is missing: $configPath"
}
if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
    throw "CPU dependency lock is missing: $lockPath"
}
$config = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
if ([string]$config.version -ne $Version) {
    throw "Version config identity mismatch: $configPath"
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
    Invoke-Native -FailureMessage "CPU build environment creation failed" -Command {
        & $Python311Executable -m venv $buildEnvironment
    }
}
$buildPython = Join-Path $buildEnvironment "Scripts/python.exe"
Invoke-Native -FailureMessage "CPU dependency installation failed" -Command {
    & $buildPython -m pip install --disable-pip-version-check --no-deps -r $lockPath
}
Invoke-Native -FailureMessage "CPU ONNX Runtime validation failed" -Command {
    & $buildPython -c (
        "import onnxruntime as ort; " +
        "assert ort.__version__ == '1.28.0', ort.__version__; " +
        "providers = ort.get_available_providers(); " +
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
    Invoke-Native -FailureMessage "CPU Worker build failed" -Command {
        & (Join-Path $PSScriptRoot "build_worker.ps1") `
            -PythonExecutable $buildPython `
            -OutputDirectory $workerOutput `
            -SourceDateEpoch $sourceDateEpoch
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
    $provenance | Add-Member -NotePropertyName "worker_handoff" -NotePropertyValue ([ordered]@{
        platform = "windows-x64"
        provider = "CPUExecutionProvider"
        onnxruntime_version = "1.28.0"
        dependency_lock_sha256 = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $lockPath
        ).Hash.ToLowerInvariant()
        default_profile = [ordered]@{
            detector_workers = 1
            detector_intra_op_threads = 4
            embedder_intra_op_threads = 4
        }
        n100_benchmark_status = "PENDING_RECEIVER_MEASUREMENT"
        model_graph_or_weight_changed = $false
        decision_policy_changed = $false
    })
    Write-JsonFile -Path (Join-Path $temporaryRoot "provenance.json") -Value $provenance

    $forbiddenFiles = Get-ChildItem -LiteralPath $temporaryRoot -File -Recurse | Where-Object {
        $_.Name -match "(?i)^(onnxruntime_providers_(cuda|tensorrt)\.dll|cudnn.*\.dll|cublas.*\.dll|cudart.*\.dll|cufft.*\.dll|nvrtc.*\.dll|nvjitlink.*\.dll)$" -or
        $_.FullName -match "(?i)[\\/]cuda-runtime[\\/]"
    }
    if ($forbiddenFiles) {
        $paths = ($forbiddenFiles | ForEach-Object { $_.FullName }) -join ", "
        throw "CPU handoff contains CUDA or GPU provider files: $paths"
    }

    $files = Get-ChildItem -LiteralPath $temporaryRoot -File -Recurse | Sort-Object FullName |
        ForEach-Object {
            [ordered]@{
                path = [System.IO.Path]::GetRelativePath($temporaryRoot, $_.FullName).Replace("\", "/")
                size_bytes = $_.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            }
        }
    $manifest = [ordered]@{
        schema_version = "1.0"
        product_version = $Version
        platform = "windows-x64"
        provider = "CPUExecutionProvider"
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

Write-Host "CPU Worker handoff: $packageRoot"
Write-Host "ZIP: $zipPath"
Write-Host "ZIP SHA-256: $zipHash"
