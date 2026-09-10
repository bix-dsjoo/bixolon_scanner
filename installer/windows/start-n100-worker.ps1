param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$workerRoot = Join-Path $PSScriptRoot 'worker'
$env:BIXOLON_PACKAGE_DIR = Join-Path $workerRoot 'model-package'
$env:BIXOLON_CATALOG_DIR = Join-Path $workerRoot 'store-catalog'
$env:BIXOLON_PROVIDER = 'cpu'
$env:BIXOLON_EMBEDDER_PROVIDER = 'openvino_gpu'
$env:BIXOLON_EMBEDDER_FALLBACK_PROVIDER = 'same'
$env:BIXOLON_PROVIDER_EXECUTION_CPU_FALLBACK = 'true'
$env:BIXOLON_VERIFIER_PROVIDER = 'cpu'
$env:BIXOLON_CPU_DETECTOR_WORKERS = '1'
$env:BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = '4'
$env:BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = '4'
$env:BIXOLON_OPENVINO_GPU_PRECISION = 'f16'
$env:BIXOLON_REUSE_VERIFIER_EMBEDDINGS = 'true'
$env:BIXOLON_PARALLEL_VERIFICATION = 'true'
$env:BIXOLON_HOST = '127.0.0.1'
$env:BIXOLON_PORT = [string]$Port
$env:BIXOLON_REQUEST_TIMEOUT_SECONDS = '120'
$env:BIXOLON_LOG_TO_STDERR = '1'
& (Join-Path $workerRoot 'bixolon-worker.exe')
exit $LASTEXITCODE
