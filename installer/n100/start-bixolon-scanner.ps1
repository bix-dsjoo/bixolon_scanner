$ErrorActionPreference = "Stop"

$productExecutable = Join-Path $PSScriptRoot "product_scanner.exe"
$gpuPlugin = Join-Path $PSScriptRoot "worker/_internal/openvino_intel_gpu_plugin.dll"
$metadataPath = Join-Path $PSScriptRoot "worker/model-package/metadata.json"
foreach ($requiredPath in @($productExecutable, $gpuPlugin, $metadataPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "BIXOLON Scanner hybrid runtime file is missing: $requiredPath"
    }
}
$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
$cacheRoot = Join-Path (
    [Environment]::GetFolderPath("LocalApplicationData")
) "BIXOLON Scanner/openvino-cache/$($metadata.worker_version)"

$env:BIXOLON_PROVIDER = "openvino"
$env:BIXOLON_EMBEDDER_PROVIDER = "openvino_gpu"
$env:BIXOLON_EMBEDDER_FALLBACK_PROVIDER = "same"
$env:BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"
$env:BIXOLON_CPU_DETECTOR_WORKERS = "1"
$env:BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"
$env:BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "0"
$env:BIXOLON_OPENVINO_CACHE_DIR = $cacheRoot

Start-Process -FilePath $productExecutable -WorkingDirectory $PSScriptRoot
