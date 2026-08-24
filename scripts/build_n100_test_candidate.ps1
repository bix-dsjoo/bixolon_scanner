param(
    [string]$Version = "0.1.2",
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

foreach ($required in @(
    (Join-Path $workerSource "bixolon-worker.exe"),
    (Join-Path $workerSource "_internal"),
    $metadataPath,
    (Join-Path $catalogSource "catalog.json"),
    (Join-Path $repositoryRoot "scripts/handoff/start-worker.ps1"),
    (Join-Path $repositoryRoot "scripts/handoff/benchmark-n100.ps1"),
    (Join-Path $repositoryRoot "scripts/handoff/N100-STAGE-TEST.ps1"),
    (Join-Path $repositoryRoot "scripts/handoff/RUN-N100-TEST.cmd"),
    (Join-Path $repositoryRoot "docs/diagnostics/n100-worker-0.0.2.json"),
    (Join-Path $repositoryRoot "configs/runtime/requirements-windows-openvino.lock"),
    (Join-Path $repositoryRoot "artifacts/evaluations/scanner-0.1.2/development-415-current-pc-cpu.json")
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
    [double]$metadata.detector.score_threshold -ne 0.33 -or
    $null -eq $metadata.count_verifier -or
    [string]$metadata.count_verifier.comparison_mode -ne "object_presence" -or
    [int]$metadata.count_verifier.input_size[0] -ne 192 -or
    [string]$metadata.embedder.embedder_id -ne "dinov3-convnext-tiny-adapted"
) {
    throw "The 0.1.2 N100 candidate does not match the selected single-detector Runtime."
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
    Copy-Item -LiteralPath (Join-Path $repositoryRoot "docs/diagnostics/n100-worker-0.0.2.json") `
        -Destination (Join-Path $temporaryRoot "n100-0.0.2-baseline.json")
    Copy-Item -LiteralPath (
        Join-Path $repositoryRoot "configs/runtime/requirements-windows-openvino.lock"
    ) -Destination $temporaryRoot
    Copy-Item -LiteralPath (
        Join-Path $repositoryRoot "artifacts/evaluations/scanner-0.1.2/development-415-current-pc-cpu.json"
    ) -Destination (Join-Path $temporaryRoot "local-cpu-reference.json")

    $forbidden = Get-ChildItem -LiteralPath $temporaryRoot -File -Recurse | Where-Object {
        $_.Name -match "(?i)(directml|cuda|tensorrt|cudnn|cublas|cudart|cufft|nvrtc)"
    }
    if ($forbidden) {
        throw "CPU test candidate contains a GPU runtime file: $($forbidden[0].FullName)"
    }

    $readme = @"
BIXOLON Scanner $Version N100 CPU 성능 테스트
================================================

1. 이 폴더 전체를 N100 키오스크의 로컬 디스크에 복사합니다.
2. C:\easy 폴더에 실제 촬영 JPEG/PNG 이미지 30장 이상을 넣습니다.
3. RUN-N100-TEST.cmd를 더블클릭합니다.
4. 테스트가 끝나면 n100-$Version-result.json을 USB에 복사합니다.
5. 결과 JSON을 Codex에 전달합니다.

CMD에서 직접 실행:
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\N100-STAGE-TEST.ps1" -ImageDirectory "C:\easy" -OutputPath ".\n100-$Version-result.json"

고정 CPU 1x4 설정으로 평균과 p95 1초 목표를 확인합니다.
결과의 passes, target.mean_within_1_second, target.p95_within_1_second가 모두 true이면 통과입니다.
이미지 경로와 bytes는 결과 JSON에 기록하지 않습니다.
이 후보는 N100 실측용이며 최종 Setup 설치 프로그램이 아닙니다.
"@
    Write-Utf8 -Path (Join-Path $temporaryRoot "README-N100-KO.txt") `
        -Text ($readme + [Environment]::NewLine)

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
        artifact = "n100_cpu_latency_candidate"
        product_version = $Version
        provider = "CPUExecutionProvider"
        detector_filename = [string]$metadata.detector.filename
        count_verifier_filename = [string]$metadata.count_verifier.filename
        embedder_id = [string]$metadata.embedder.embedder_id
        default_profile = [ordered]@{
            provider = "cpu"
            detector_workers = 1
            detector_threads_per_session = 4
            embedder_threads = 4
        }
        target_full_path_latency_ms = 1000
        baseline_reference = "n100-0.0.2-baseline.json"
        current_pc_reference = "local-cpu-reference.json"
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
