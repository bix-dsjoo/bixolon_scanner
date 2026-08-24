param(
    [int]$Port = 8000,
    [ValidateRange(1, 4)]
    [int]$DetectorWorkers = 4,
    [ValidateRange(0, 64)]
    [int]$DetectorThreads = 1
)

$ErrorActionPreference = "Stop"

$workerRoot = Join-Path $PSScriptRoot "worker"
$workerExecutable = Join-Path $workerRoot "bixolon-worker.exe"
$modelPackage = Join-Path $workerRoot "model-package"
$storeCatalog = Join-Path $workerRoot "store-catalog"
$metadataPath = Join-Path $modelPackage "metadata.json"

foreach ($requiredPath in @(
    $workerExecutable,
    $metadataPath,
    (Join-Path $storeCatalog "catalog.json")
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "DirectML Worker package file is missing: $requiredPath"
    }
}

$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
$env:BIXOLON_PACKAGE_DIR = $modelPackage
$env:BIXOLON_CATALOG_DIR = $storeCatalog
$env:BIXOLON_PROVIDER = "cpu"
$env:BIXOLON_EMBEDDER_PROVIDER = "directml"
$env:BIXOLON_HOST = "127.0.0.1"
$env:BIXOLON_PORT = [string]$Port
$env:BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"
$env:BIXOLON_CPU_DETECTOR_WORKERS = [string]$DetectorWorkers
$env:BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = [string]$DetectorThreads
$env:BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "0"
$env:BIXOLON_LOG_TO_STDERR = "1"

Write-Host "BIXOLON Worker $($metadata.worker_version): http://127.0.0.1:$Port"
Write-Host "Detector=CPU $DetectorWorkers x $DetectorThreads, Embedder=DirectML device 0"
Write-Host "Stop the Worker with Ctrl+C."

& $workerExecutable
exit $LASTEXITCODE
