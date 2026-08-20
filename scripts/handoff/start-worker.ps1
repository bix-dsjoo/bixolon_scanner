param(
    [int]$Port = 8000,
    [ValidateRange(1, 4)]
    [int]$DetectorWorkers = 1,
    [ValidateRange(0, 64)]
    [int]$DetectorThreads = 4,
    [ValidateRange(0, 64)]
    [int]$EmbedderThreads = 4
)

$ErrorActionPreference = "Stop"

$workerRoot = Join-Path $PSScriptRoot "worker"
$workerExecutable = Join-Path $workerRoot "bixolon-worker.exe"
$modelPackage = Join-Path $workerRoot "model-package"
$storeCatalog = Join-Path $workerRoot "store-catalog"

foreach ($requiredPath in @(
    $workerExecutable,
    (Join-Path $modelPackage "metadata.json"),
    (Join-Path $storeCatalog "catalog.json")
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "CPU Worker package file is missing: $requiredPath"
    }
}

$env:BIXOLON_PACKAGE_DIR = $modelPackage
$env:BIXOLON_CATALOG_DIR = $storeCatalog
$env:BIXOLON_PROVIDER = "cpu"
$env:BIXOLON_HOST = "127.0.0.1"
$env:BIXOLON_PORT = [string]$Port
$env:BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"
$env:BIXOLON_CPU_DETECTOR_WORKERS = [string]$DetectorWorkers
$env:BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = [string]$DetectorThreads
$env:BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = [string]$EmbedderThreads
$env:BIXOLON_LOG_TO_STDERR = "1"

Write-Host "BIXOLON Worker 0.0.2 CPU: http://127.0.0.1:$Port"
Write-Host (
    "Detector={0} worker(s) x {1} thread(s), Embedder={2} thread(s)" -f `
        $DetectorWorkers, $DetectorThreads, $EmbedderThreads
)
Write-Host "Stop the Worker with Ctrl+C."

& $workerExecutable
exit $LASTEXITCODE
