$ErrorActionPreference = "Stop"

$productExecutable = Join-Path $PSScriptRoot "product_scanner.exe"
if (-not (Test-Path -LiteralPath $productExecutable -PathType Leaf)) {
    throw "BIXOLON Scanner executable is missing: $productExecutable"
}

$env:BIXOLON_PROVIDER = "cpu"
$env:BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"
$env:BIXOLON_CPU_DETECTOR_WORKERS = "1"
$env:BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"
$env:BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = "4"

Start-Process -FilePath $productExecutable -WorkingDirectory $PSScriptRoot
