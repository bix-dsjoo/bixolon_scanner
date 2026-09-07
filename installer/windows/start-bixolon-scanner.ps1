$ErrorActionPreference = "Stop"

$productExecutable = Join-Path $PSScriptRoot "product_scanner.exe"
$metadataPath = Join-Path $PSScriptRoot "worker/model-package/metadata.json"
foreach ($requiredPath in @($productExecutable, $metadataPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "BIXOLON Bakery AI Scanner runtime file is missing: $requiredPath"
    }
}
$env:BIXOLON_PROVIDER = "cpu"
$env:BIXOLON_EMBEDDER_PROVIDER = "same"
$env:BIXOLON_EMBEDDER_FALLBACK_PROVIDER = "none"
$env:BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"
$env:BIXOLON_CPU_DETECTOR_WORKERS = "1"
$env:BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"
$env:BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "4"
Start-Process -FilePath $productExecutable -WorkingDirectory $PSScriptRoot
