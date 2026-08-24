param(
    [string]$PythonExecutable = "python",
    [string]$OutputDirectory = "artifacts/worker",
    [long]$SourceDateEpoch = 0,
    [string]$OpenVinoLibraryDirectory = "",
    [switch]$IncludeOpenVinoGpu
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$resolvedOutput = Join-Path $repositoryRoot $OutputDirectory
$workDirectory = Join-Path $resolvedOutput "build"
$specDirectory = Join-Path $resolvedOutput "spec"
$entryPoint = Join-Path $repositoryRoot "src/bixolon_scanner/worker/__main__.py"
$sourceDirectory = Join-Path $repositoryRoot "src"

$previousSourceDateEpoch = $env:SOURCE_DATE_EPOCH
try {
    if ($SourceDateEpoch -gt 0) {
        $env:SOURCE_DATE_EPOCH = [string]$SourceDateEpoch
    }
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
        --exclude-module psutil `
        --exclude-module rich `
        --exclude-module pygments `
        --exclude-module charset_normalizer `
        --exclude-module websockets `
        --exclude-module watchfiles `
        --exclude-module httptools `
        --exclude-module uvloop `
        --exclude-module yaml `
        --exclude-module markupsafe `
        --exclude-module werkzeug `
        --exclude-module pkg_resources `
        --exclude-module setuptools `
        --hidden-import uvicorn.logging `
        --hidden-import uvicorn.loops.auto `
        --hidden-import uvicorn.loops.asyncio `
        --hidden-import uvicorn.protocols.http.auto `
        --hidden-import uvicorn.protocols.http.h11_impl `
        --hidden-import uvicorn.lifespan.on `
        --hidden-import numpy._core._exceptions `
        $entryPoint
}
finally {
    $env:SOURCE_DATE_EPOCH = $previousSourceDateEpoch
}

if ($LASTEXITCODE -ne 0) {
    throw "BIXOLON Worker packaging failed with exit code $LASTEXITCODE."
}

$workerExecutable = Join-Path $resolvedOutput "bixolon-worker/bixolon-worker.exe"
if (-not (Test-Path -LiteralPath $workerExecutable -PathType Leaf)) {
    throw "Packaged Worker executable was not created: $workerExecutable"
}

if (-not [string]::IsNullOrWhiteSpace($OpenVinoLibraryDirectory)) {
    $resolvedOpenVinoLibraries = [System.IO.Path]::GetFullPath($OpenVinoLibraryDirectory)
    if (-not (Test-Path -LiteralPath $resolvedOpenVinoLibraries -PathType Container)) {
        throw "OpenVINO library directory is missing: $resolvedOpenVinoLibraries"
    }
    $internalDirectory = Join-Path $resolvedOutput "bixolon-worker/_internal"
    $requiredOpenVinoFiles = @(
        "cache.json",
        "openvino.dll",
        "openvino_intel_cpu_plugin.dll",
        "openvino_onnx_frontend.dll",
        "tbb12.dll",
        "tbbbind_2_5.dll",
        "tbbmalloc.dll",
        "tbbmalloc_proxy.dll"
    )
    if ($IncludeOpenVinoGpu) {
        $requiredOpenVinoFiles += "openvino_intel_gpu_plugin.dll"
    }
    foreach ($filename in $requiredOpenVinoFiles) {
        $source = Join-Path $resolvedOpenVinoLibraries $filename
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Required OpenVINO runtime file is missing: $source"
        }
        Copy-Item -LiteralPath $source -Destination $internalDirectory -Force
    }
}

Write-Host "Packaged Worker: $workerExecutable"
