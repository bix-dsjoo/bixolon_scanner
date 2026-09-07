param(
    [int]$Port = 8000
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
        throw "BIXOLON Worker runtime file is missing: $requiredPath"
    }
}

$metadata = Get-Content -Raw -LiteralPath (Join-Path $modelPackage "metadata.json") |
    ConvertFrom-Json

$env:BIXOLON_PACKAGE_DIR = $modelPackage
$env:BIXOLON_CATALOG_DIR = $storeCatalog
$env:BIXOLON_PROVIDER = "cpu"
$env:BIXOLON_EMBEDDER_PROVIDER = "same"
$env:BIXOLON_EMBEDDER_FALLBACK_PROVIDER = "none"
$env:BIXOLON_HOST = "127.0.0.1"
$env:BIXOLON_PORT = [string]$Port
$env:BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"
$env:BIXOLON_CPU_DETECTOR_WORKERS = "1"
$env:BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"
$env:BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "4"
$env:BIXOLON_LOG_TO_STDERR = "1"

Write-Host "BIXOLON Worker $($metadata.worker_version): http://127.0.0.1:$Port"
Write-Host "Detector/Embedder/Verifier=ONNX Runtime CPU"
Write-Host "종료하려면 Ctrl+C를 누르십시오."

& $workerExecutable
exit $LASTEXITCODE
