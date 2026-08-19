param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$OutputDirectory = "artifacts\portable\rpc200-v18-benchmark",
    [string]$Commit = "e08bd604bdc7f14d2899e7bfe366c274bc47b395",
    [switch]$WithoutCudaRuntime
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repo = [IO.Path]::GetFullPath($RepoRoot)
$dataset = [IO.Path]::GetFullPath($DatasetRoot)
$output = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
    [IO.Path]::GetFullPath($OutputDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $repo $OutputDirectory))
}

$artifactRoot = Join-Path $repo "artifacts\experiments\rpc-data-scale-diverse-worker-gated"
$sourceManifest = Join-Path $artifactRoot "detector\manifest\manifest.jsonl"
$sourcePackage = Join-Path $artifactRoot "validation-candidate-package-small-object-v5"
$sourceContext = Join-Path $artifactRoot "runs\full\seed20260810\context-rejector\logistic.onnx"
$sourceCuda = Join-Path $repo "apps\product_scanner\build\windows\x64\runner\Release\worker\cuda-runtime"

foreach ($required in @($repo, $dataset, $sourceManifest, $sourcePackage, $sourceContext)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required input does not exist: $required"
    }
}
if (-not $WithoutCudaRuntime -and -not (Test-Path -LiteralPath $sourceCuda -PathType Container)) {
    throw "CUDA runtime does not exist: $sourceCuda"
}
if (Test-Path -LiteralPath $output) {
    throw "Output already exists. Choose a new path or remove it explicitly: $output"
}

New-Item -ItemType Directory -Path $output | Out-Null
$sourceArchive = Join-Path $output "source.zip"
& git -C $repo archive --format=zip --output=$sourceArchive $Commit -- pyproject.toml README.md src
if ($LASTEXITCODE -ne 0) {
    throw "git archive failed for commit $Commit"
}
Expand-Archive -LiteralPath $sourceArchive -DestinationPath (Join-Path $output "source")
Remove-Item -LiteralPath $sourceArchive

Copy-Item -LiteralPath $sourcePackage -Destination (Join-Path $output "model-package") -Recurse
New-Item -ItemType Directory -Path (Join-Path $output "context") | Out-Null
Copy-Item -LiteralPath $sourceContext -Destination (Join-Path $output "context\logistic.onnx")

function Get-SelectionKey([object]$Record) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes("rpc-validation-benchmark:$($Record.image_id)")
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

$records = Get-Content -LiteralPath $sourceManifest | ForEach-Object { $_ | ConvertFrom-Json }
$selected = [Collections.Generic.List[object]]::new()
$datasetOutput = Join-Path $output "dataset"
New-Item -ItemType Directory -Path $datasetOutput | Out-Null

foreach ($level in @("easy", "medium", "hard")) {
    $levelRecords = @(
        $records |
            Where-Object { $_.role -eq "selection" -and $_.level -eq $level } |
            Sort-Object @{ Expression = { Get-SelectionKey $_ } } |
            Select-Object -First 200
    )
    if ($levelRecords.Count -ne 200) {
        throw "Expected 200 $level selection images, found $($levelRecords.Count)"
    }
    foreach ($record in $levelRecords) {
        $relative = [string]$record.image_path
        $sourceImage = [IO.Path]::GetFullPath((Join-Path $dataset $relative))
        if (-not $sourceImage.StartsWith($dataset + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Manifest image escapes the dataset root: $relative"
        }
        if (-not (Test-Path -LiteralPath $sourceImage -PathType Leaf)) {
            throw "Manifest image does not exist: $sourceImage"
        }
        $targetImage = Join-Path $datasetOutput $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $targetImage) -Force | Out-Null
        Copy-Item -LiteralPath $sourceImage -Destination $targetImage
        $selected.Add($record)
    }
}

$portableManifest = Join-Path $output "manifest.jsonl"
$manifestLines = @(
    $selected | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 20 }
)
[IO.File]::WriteAllLines($portableManifest, $manifestLines, [Text.UTF8Encoding]::new($false))

if (-not $WithoutCudaRuntime) {
    Copy-Item -LiteralPath $sourceCuda -Destination (Join-Path $output "cuda-runtime") -Recurse
}

$runScript = @'
param(
    [switch]$Cpu,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not $SkipInstall) {
    if (-not (Test-Path -LiteralPath $python)) {
        & py -3.11 -m venv (Join-Path $PSScriptRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "Python 3.11 is required." }
    }
    & $python -m pip install --upgrade pip
    & $python -m pip install -e "$PSScriptRoot\source[cuda]" `
        "onnxruntime-gpu==1.28.0" "numpy==2.4.4" "pillow==12.2.0" `
        "scikit-learn>=1.6,<2"
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment is missing. Run without -SkipInstall first."
}

$provider = if ($Cpu) { "cpu" } else { "cuda" }
if (-not $Cpu -and -not (Test-Path -LiteralPath "$PSScriptRoot\cuda-runtime")) {
    throw "cuda-runtime is missing. Use -Cpu or provide the CUDA runtime folder."
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultDirectory = Join-Path $PSScriptRoot "results\$stamp-$provider"
New-Item -ItemType Directory -Path $resultDirectory -Force | Out-Null

$environment = [ordered]@{
    captured_at = (Get-Date).ToString("o")
    provider = $provider
    os = [Environment]::OSVersion.VersionString
    source_commit = "__SOURCE_COMMIT__"
    nvidia_smi = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1) -join "`n"
    } else { $null }
}
$environment | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultDirectory "environment.json") -Encoding utf8

$common = @(
    "-m", "bixolon_scanner.experiments.rpc200.validation_benchmark",
    "--package-dir", (Join-Path $PSScriptRoot "model-package"),
    "--context-onnx", (Join-Path $PSScriptRoot "context\logistic.onnx"),
    "--manifest", (Join-Path $PSScriptRoot "manifest.jsonl"),
    "--dataset-root", (Join-Path $PSScriptRoot "dataset"),
    "--provider", $provider,
    "--warmup", "30",
    "--images-per-level", "200",
    "--class-aware-nms-threshold", "0.55",
    "--context-threshold", "0.00225",
    "--duplicate-overlap-threshold", "0.452",
    "--duplicate-overlap-max-score", "0.864",
    "--duplicate-overlap-max-quality", "0.194",
    "--duplicate-low-quality-max-quality", "0.0058",
    "--duplicate-low-quality-min-score", "0.93",
    "--duplicate-min-rank", "0.857",
    "--assignment-conflict-top-k", "2",
    "--assignment-mutual-pair", "160:161"
)
if (-not $Cpu) {
    $common += @("--cuda-dll-dir", (Join-Path $PSScriptRoot "cuda-runtime"))
}

foreach ($level in @("easy", "medium", "hard")) {
    Write-Host "Running RPC200 v18 $level benchmark..."
    & $python @common --levels $level --output (Join-Path $resultDirectory "$level.json")
    if ($LASTEXITCODE -ne 0) { throw "$level benchmark failed." }
}

Write-Host ""
Write-Host "Completed: $resultDirectory"
foreach ($level in @("easy", "medium", "hard")) {
    $report = Get-Content -Raw (Join-Path $resultDirectory "$level.json") | ConvertFrom-Json
    $latency = $report.difficulty.$level.all_images
    "{0,-6} mean={1,8:N3} ms  p95={2,8:N3} ms  p99={3,8:N3} ms" -f `
        $level, $latency.mean_ms, $latency.p95_ms, $latency.p99_ms
}
'@
$runScript = $runScript.Replace("__SOURCE_COMMIT__", $Commit)
Set-Content -LiteralPath (Join-Path $output "RUN_BENCHMARK.ps1") -Value $runScript -Encoding utf8

$readme = @"
# RPC200 v18 portable benchmark

This bundle contains the fixed 600-image RPC200 validation benchmark, the ONNX models,
source commit $Commit, and the optional CUDA 13/cuDNN 9 runtime used by the original run.

## Run on another Windows PC

1. Install Python 3.11 and a current NVIDIA driver.
2. Open PowerShell in this directory.
3. Run: ``powershell -ExecutionPolicy Bypass -File .\RUN_BENCHMARK.ps1``

The first run downloads Python dependencies. Results are written below ``results``.
For a CPU-only run use: ``powershell -ExecutionPolicy Bypass -File .\RUN_BENCHMARK.ps1 -Cpu``

Original RTX 5080 reference (all-images mean / p95):

- easy: 55.632 / 70.661 ms
- medium: 65.373 / 80.578 ms
- hard: 77.951 / 96.798 ms
"@
Set-Content -LiteralPath (Join-Path $output "README.md") -Value $readme -Encoding utf8

$checksumLines = Get-ChildItem -LiteralPath $output -Recurse -File |
    Where-Object { $_.Name -ne "checksums.sha256" } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = [IO.Path]::GetRelativePath($output, $_.FullName).Replace("\", "/")
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relative"
    }
Set-Content -LiteralPath (Join-Path $output "checksums.sha256") -Value $checksumLines -Encoding ascii

$totalBytes = (Get-ChildItem -LiteralPath $output -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host "Portable benchmark prepared: $output"
Write-Host ("Files: {0}; Size: {1:N2} GB" -f $checksumLines.Count, ($totalBytes / 1GB))
