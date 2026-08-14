param(
    [string]$PythonExecutable = "python",
    [string]$OutputDirectory = "artifacts/worker"
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$resolvedOutput = Join-Path $repositoryRoot $OutputDirectory
$workDirectory = Join-Path $resolvedOutput "build"
$specDirectory = Join-Path $resolvedOutput "spec"
$entryPoint = Join-Path $repositoryRoot "src/bixolon_scanner/worker/__main__.py"
$sourceDirectory = Join-Path $repositoryRoot "src"

& $PythonExecutable -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name bixolon-worker `
    --paths $sourceDirectory `
    --distpath $resolvedOutput `
    --workpath $workDirectory `
    --specpath $specDirectory `
    --exclude-module torch `
    --exclude-module torchvision `
    --exclude-module onnx `
    --exclude-module onnxruntime.quantization `
    --exclude-module onnxruntime.tools `
    --exclude-module onnxruntime.transformers `
    --exclude-module matplotlib `
    --exclude-module pytest `
    --exclude-module scipy `
    --exclude-module pkg_resources `
    --exclude-module setuptools `
    --hidden-import uvicorn.logging `
    --hidden-import uvicorn.loops.auto `
    --hidden-import uvicorn.loops.asyncio `
    --hidden-import uvicorn.protocols.http.auto `
    --hidden-import uvicorn.protocols.http.h11_impl `
    --hidden-import uvicorn.protocols.websockets.auto `
    --hidden-import uvicorn.protocols.websockets.websockets_impl `
    --hidden-import uvicorn.lifespan.on `
    --hidden-import numpy._core._exceptions `
    $entryPoint

if ($LASTEXITCODE -ne 0) {
    throw "BIXOLON Worker packaging failed with exit code $LASTEXITCODE."
}

$workerExecutable = Join-Path $resolvedOutput "bixolon-worker/bixolon-worker.exe"
if (-not (Test-Path -LiteralPath $workerExecutable -PathType Leaf)) {
    throw "Packaged Worker executable was not created: $workerExecutable"
}

Write-Host "Packaged Worker: $workerExecutable"
