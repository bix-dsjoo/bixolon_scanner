param(
    [Parameter(Mandatory = $true)]
    [string]$ImageDirectory,
    [string]$OutputPath = "",
    [int]$Port = 8188,
    [int]$MinimumImages = 30,
    [int]$MinimumFullPathImages = 10,
    [ValidateRange(0, 20)]
    [int]$WarmupCount = 3
)

$ErrorActionPreference = "Stop"

$manifestPath = Join-Path $PSScriptRoot "candidate-manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Candidate integrity manifest is missing: $manifestPath"
}
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if (
    [string]$manifest.product_version -ne "0.1.14" -or
    [string]$manifest.provider -ne "OpenVINOExecutionProvider:CPU" -or
    [int]$manifest.file_count -ne @($manifest.files).Count
) {
    throw "Candidate integrity manifest contract is invalid."
}

$packageRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)
$packagePrefix = $packageRoot + [System.IO.Path]::DirectorySeparatorChar
foreach ($record in @($manifest.files)) {
    $relativePath = ([string]$record.path).Replace(
        "/",
        [System.IO.Path]::DirectorySeparatorChar
    )
    $resolvedPath = [System.IO.Path]::GetFullPath((Join-Path $packageRoot $relativePath))
    if (-not $resolvedPath.StartsWith(
        $packagePrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Candidate manifest contains an unsafe path."
    }
    if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
        throw "Candidate file is missing: $relativePath"
    }
    $file = Get-Item -LiteralPath $resolvedPath
    $sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedPath).Hash.ToLowerInvariant()
    if ($file.Length -ne [long]$record.size_bytes -or $sha256 -ne [string]$record.sha256) {
        throw "Candidate file integrity check failed: $relativePath"
    }
}
Write-Host "Candidate SHA-256 verification passed ($($manifest.file_count) files)."

$benchmark = Join-Path $PSScriptRoot "benchmark-n100.ps1"
if (-not (Test-Path -LiteralPath $benchmark -PathType Leaf)) {
    throw "N100 benchmark script is missing: $benchmark"
}
& $benchmark @PSBoundParameters
exit $LASTEXITCODE
