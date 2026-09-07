param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeRoot,
    [Parameter(Mandatory = $true)]
    [string]$StoreBundleRoot,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [ValidateSet("same")]
    [string]$EmbedderProvider = "same",
    [ValidateSet("none")]
    [string]$EmbedderFallbackProvider = "none",
    [switch]$WaitUntilReady
)

$ErrorActionPreference = "Stop"

$resolvedRuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$resolvedStoreBundleRoot = [System.IO.Path]::GetFullPath($StoreBundleRoot)
$workerExecutable = Join-Path $resolvedRuntimeRoot "worker/bixolon-worker.exe"
$modelPackage = Join-Path $resolvedStoreBundleRoot "model-package"
$storeCatalog = Join-Path $resolvedStoreBundleRoot "store-catalog"

foreach ($requiredPath in @(
    $workerExecutable,
    (Join-Path $modelPackage "metadata.json"),
    (Join-Path $storeCatalog "catalog.json")
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file is missing: $requiredPath"
    }
}

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $workerExecutable
$startInfo.WorkingDirectory = Split-Path -Parent $workerExecutable
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.Environment["BIXOLON_PACKAGE_DIR"] = $modelPackage
$startInfo.Environment["BIXOLON_CATALOG_DIR"] = $storeCatalog
$startInfo.Environment["BIXOLON_PROVIDER"] = "cpu"
$startInfo.Environment["BIXOLON_EMBEDDER_PROVIDER"] = $EmbedderProvider
$startInfo.Environment["BIXOLON_EMBEDDER_FALLBACK_PROVIDER"] = $EmbedderFallbackProvider
$startInfo.Environment["BIXOLON_HOST"] = $HostAddress
$startInfo.Environment["BIXOLON_PORT"] = [string]$Port
$startInfo.Environment["BIXOLON_REQUEST_TIMEOUT_SECONDS"] = "60"
$startInfo.Environment["BIXOLON_LOG_TO_STDERR"] = "0"

$process = [System.Diagnostics.Process]::Start($startInfo)
if ($null -eq $process) {
    throw "Could not start the Worker."
}
Write-Host "Worker started. PID=$($process.Id) URL=http://${HostAddress}:$Port"

if ($WaitUntilReady) {
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds(180)
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($process.HasExited) {
            throw "Worker exited before readiness. ExitCode=$($process.ExitCode)"
        }
        try {
            $ready = Invoke-RestMethod -Uri "http://${HostAddress}:$Port/health/ready" -TimeoutSec 1
            if ($ready.status -eq "ready") {
                Write-Host "Worker ready. version=$($ready.worker_version) provider=$($ready.provider)"
                break
            }
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if ([DateTimeOffset]::UtcNow -ge $deadline) {
        throw "Worker did not become ready within 180 seconds."
    }
}

$process
