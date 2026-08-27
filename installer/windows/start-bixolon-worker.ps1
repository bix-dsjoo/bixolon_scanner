param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$workerRoot = Join-Path $PSScriptRoot "worker"
$workerExecutable = Join-Path $workerRoot "bixolon-worker.exe"
$modelPackage = Join-Path $workerRoot "model-package"
$storeCatalog = Join-Path $workerRoot "store-catalog"
$gpuPlugin = Join-Path $workerRoot "_internal/openvino_intel_gpu_plugin.dll"

foreach ($requiredPath in @(
    $workerExecutable,
    (Join-Path $modelPackage "metadata.json"),
    (Join-Path $storeCatalog "catalog.json"),
    $gpuPlugin
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "BIXOLON Worker hybrid runtime file is missing: $requiredPath"
    }
}

$metadata = Get-Content -Raw -LiteralPath (Join-Path $modelPackage "metadata.json") |
    ConvertFrom-Json
$cacheRoot = Join-Path (
    [Environment]::GetFolderPath("LocalApplicationData")
) "BIXOLON Bakery AI Scanner/openvino-cache/$($metadata.worker_version)"

$env:BIXOLON_PACKAGE_DIR = $modelPackage
$env:BIXOLON_CATALOG_DIR = $storeCatalog
$env:BIXOLON_PROVIDER = "openvino"
$env:BIXOLON_EMBEDDER_PROVIDER = "openvino_gpu"
$env:BIXOLON_EMBEDDER_FALLBACK_PROVIDER = "same"
$env:BIXOLON_HOST = "127.0.0.1"
$env:BIXOLON_PORT = [string]$Port
$env:BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"
$env:BIXOLON_CPU_DETECTOR_WORKERS = "1"
$env:BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"
$env:BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "0"
$env:BIXOLON_OPENVINO_CACHE_DIR = $cacheRoot
$env:BIXOLON_LOG_TO_STDERR = "1"

Write-Host "BIXOLON Worker $($metadata.worker_version): http://127.0.0.1:$Port"
Write-Host "Detector=OpenVINO CPU, Embedder/Verifier=OpenVINO Intel GPU"
Write-Host "Intel GPU 초기화에 실패하면 경고 로그를 남기고 CPU Embedder로 시작합니다."
Write-Host "종료하려면 Ctrl+C를 누르십시오."

& $workerExecutable
exit $LASTEXITCODE
