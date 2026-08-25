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
Add-Type -AssemblyName System.Net.Http

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $PSScriptRoot "n100-benchmark-result.json"
}

function Get-Percentile {
    param([double[]]$Values, [double]$Probability)
    if ($Values.Count -eq 0) {
        return $null
    }
    $ordered = @($Values | Sort-Object)
    $index = [int][Math]::Ceiling(($ordered.Count - 1) * $Probability)
    return [Math]::Round([double]$ordered[$index], 3)
}

function Test-VersionContract {
    param($Body)
    $names = @(
        "worker_version",
        "detector_version",
        "classifier_version",
        "embedder_version",
        "detector_policy_version",
        "classifier_policy_version",
        "catalog_version"
    )
    foreach ($name in $names) {
        $value = $Body.$name
        if ($null -ne $value -and [string]$value -ne $script:ExpectedVersion) {
            return $false
        }
    }
    return $true
}

function Invoke-Scan {
    param(
        [System.Net.Http.HttpClient]$Client,
        [System.IO.FileInfo]$Image,
        [string]$BaseUrl
    )
    $multipart = [System.Net.Http.MultipartFormDataContent]::new()
    try {
        $bytes = [System.IO.File]::ReadAllBytes($Image.FullName)
        $fileContent = [System.Net.Http.ByteArrayContent]::new($bytes)
        $extension = $Image.Extension.ToLowerInvariant()
        $mediaType = if ($extension -eq ".png") { "image/png" } else { "image/jpeg" }
        $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new(
            $mediaType
        )
        $multipart.Add($fileContent, "image", $Image.Name)
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = $Client.PostAsync("$BaseUrl/v1/scan", $multipart).GetAwaiter().GetResult()
        $bodyText = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        $stopwatch.Stop()
        return [pscustomobject]@{
            HttpStatus = [int]$response.StatusCode
            ElapsedMs = $stopwatch.Elapsed.TotalMilliseconds
            Body = $bodyText | ConvertFrom-Json
        }
    }
    finally {
        $multipart.Dispose()
    }
}

function Invoke-Profile {
    param(
        $Profile,
        [System.IO.FileInfo[]]$Images,
        [string]$WorkerRoot,
        [int]$WorkerPort,
        [int]$ProfileWarmupCount
    )
    $workerExecutable = Join-Path $WorkerRoot "bixolon-worker.exe"
    $environment = [ordered]@{
        BIXOLON_PACKAGE_DIR = Join-Path $WorkerRoot "model-package"
        BIXOLON_CATALOG_DIR = Join-Path $WorkerRoot "store-catalog"
        BIXOLON_PROVIDER = [string]$Profile.Provider
        BIXOLON_HOST = "127.0.0.1"
        BIXOLON_PORT = [string]$WorkerPort
        BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"
        BIXOLON_CPU_DETECTOR_WORKERS = [string]$Profile.DetectorWorkers
        BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = [string]$Profile.DetectorThreads
        BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = [string]$Profile.EmbedderThreads
        BIXOLON_LOG_TO_STDERR = "0"
    }
    $previous = @{}
    foreach ($entry in $environment.GetEnumerator()) {
        $previous[$entry.Key] = [Environment]::GetEnvironmentVariable(
            $entry.Key,
            [EnvironmentVariableTarget]::Process
        )
        [Environment]::SetEnvironmentVariable(
            $entry.Key,
            $entry.Value,
            [EnvironmentVariableTarget]::Process
        )
    }

    $process = $null
    $client = $null
    try {
        $startup = [System.Diagnostics.Stopwatch]::StartNew()
        $process = Start-Process `
            -FilePath $workerExecutable `
            -WorkingDirectory $WorkerRoot `
            -WindowStyle Hidden `
            -PassThru
        $baseUrl = "http://127.0.0.1:$WorkerPort"
        $ready = $null
        $deadline = [DateTime]::UtcNow.AddSeconds(180)
        while ([DateTime]::UtcNow -lt $deadline) {
            $process.Refresh()
            if ($process.HasExited) {
                throw "Worker exited before readiness for profile $($Profile.Name)."
            }
            try {
                $ready = Invoke-RestMethod -Uri "$baseUrl/health/ready" -TimeoutSec 2
                if ($ready.status -eq "ready") {
                    break
                }
            }
            catch {
                # Model sessions may still be warming.
            }
            Start-Sleep -Milliseconds 200
        }
        $startup.Stop()
        if ($null -eq $ready -or $ready.status -ne "ready") {
            throw "Worker readiness timed out for profile $($Profile.Name)."
        }
        if ($ready.provider -ne $Profile.Provider -or -not (Test-VersionContract $ready)) {
            throw "Worker readiness contract mismatch for profile $($Profile.Name)."
        }

        $client = [System.Net.Http.HttpClient]::new()
        $client.Timeout = [TimeSpan]::FromSeconds(65)
        for ($warmupIndex = 0; $warmupIndex -lt $ProfileWarmupCount; $warmupIndex++) {
            $warmupImage = $Images[$warmupIndex % $Images.Count]
            $warmup = Invoke-Scan -Client $client -Image $warmupImage -BaseUrl $baseUrl
            if (
                $warmup.HttpStatus -lt 200 -or
                $warmup.HttpStatus -ge 300 -or
                $warmup.Body.status -eq "ERROR" -or
                -not (Test-VersionContract $warmup.Body)
            ) {
                throw "Worker warm-up response contract failed for profile $($Profile.Name)."
            }
        }
        $fullPathLatencies = [System.Collections.Generic.List[double]]::new()
        $statusCounts = [ordered]@{
            SEGMENTATION = 0
            IMAGE_RECAPTURE = 0
            ERROR = 0
        }
        $segmentationStatusCounts = [ordered]@{
            APPROVED = 0
            UNKNOWN = 0
            SEGMENT_RECAPTURE = 0
        }
        $errorCount = 0
        foreach ($image in $Images) {
            $scan = Invoke-Scan -Client $client -Image $image -BaseUrl $baseUrl
            $responseContractValid = $statusCounts.Contains([string]$scan.Body.status)
            if ($responseContractValid) {
                $statusCounts[[string]$scan.Body.status]++
                if ([string]$scan.Body.status -eq "SEGMENTATION") {
                    $fullPathLatencies.Add($scan.ElapsedMs)
                }
            }
            foreach ($segmentation in @($scan.Body.segmentations)) {
                $segmentationStatus = [string]$segmentation.status
                if ($segmentationStatusCounts.Contains($segmentationStatus)) {
                    $segmentationStatusCounts[$segmentationStatus]++
                }
                else {
                    $responseContractValid = $false
                }
            }
            if (
                $scan.HttpStatus -lt 200 -or
                $scan.HttpStatus -ge 300 -or
                $scan.Body.status -eq "ERROR" -or
                -not (Test-VersionContract $scan.Body) -or
                -not $responseContractValid
            ) {
                $errorCount++
            }
            $process.Refresh()
        }

        $process.Refresh()
        $values = $fullPathLatencies.ToArray()
        return [pscustomobject]@{
            Name = $Profile.Name
            Provider = $Profile.Provider
            DetectorWorkers = $Profile.DetectorWorkers
            DetectorThreads = $Profile.DetectorThreads
            EmbedderThreads = $Profile.EmbedderThreads
            StartupMs = [Math]::Round($startup.Elapsed.TotalMilliseconds, 3)
            P50Ms = Get-Percentile -Values $values -Probability 0.50
            P95Ms = Get-Percentile -Values $values -Probability 0.95
            P99Ms = Get-Percentile -Values $values -Probability 0.99
            MeanMs = [Math]::Round(($values | Measure-Object -Average).Average, 3)
            PeakWorkingSetBytes = [long]$process.PeakWorkingSet64
            StatusCounts = $statusCounts
            SegmentationStatusCounts = $segmentationStatusCounts
            SegmentationCount = [int](
                ($segmentationStatusCounts.Values | Measure-Object -Sum).Sum
            )
            FullPathCount = [int]$statusCounts.SEGMENTATION
            ErrorCount = $errorCount
        }
    }
    finally {
        if ($null -ne $client) {
            $client.Dispose()
        }
        if ($null -ne $process) {
            $process.Refresh()
            if (-not $process.HasExited) {
                $process.Kill()
                $process.WaitForExit()
            }
            $process.Dispose()
        }
        foreach ($entry in $environment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable(
                $entry.Key,
                $previous[$entry.Key],
                [EnvironmentVariableTarget]::Process
            )
        }
    }
}

$packageRoot = $PSScriptRoot
$workerRoot = Join-Path $packageRoot "worker"
$workerExecutable = Join-Path $workerRoot "bixolon-worker.exe"
$metadataPath = Join-Path $workerRoot "model-package/metadata.json"
$resolvedImageDirectory = [System.IO.Path]::GetFullPath($ImageDirectory)
if (-not (Test-Path -LiteralPath $workerExecutable -PathType Leaf)) {
    throw "CPU Worker executable is missing: $workerExecutable"
}
if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
    throw "Runtime metadata is missing: $metadataPath"
}
$metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
$script:ExpectedVersion = [string]$metadata.worker_version
if (
    $null -ne $metadata.detector.ensemble -or
    [string]$metadata.detector.filename -ne "detector.onnx" -or
    [double]$metadata.detector.score_threshold -ne 0.65 -or
    $null -ne $metadata.count_verifier -or
    $null -eq $metadata.classifier_verification -or
    [double]$metadata.classifier_verification.ambiguity_maximum_approval_score -ne 0.5 -or
    [int]$metadata.classifier_verification.independent_embedder.fixed_batch_size -ne 1
) {
    throw "N100 benchmark Runtime does not match the selected consensus policy."
}
if (-not (Test-Path -LiteralPath $resolvedImageDirectory -PathType Container)) {
    throw "Benchmark image directory is missing: $resolvedImageDirectory"
}
$images = @(
    Get-ChildItem -LiteralPath $resolvedImageDirectory -File |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png") } |
        Sort-Object Name
)
if ($images.Count -lt $MinimumImages) {
    throw "N100 benchmark requires at least $MinimumImages JPEG/PNG images."
}

$profiles = [System.Collections.Generic.List[object]]::new()
$profiles.Add(
    [pscustomobject]@{
        Name = "candidate-openvino-1xauto"
        Provider = "openvino"
        DetectorWorkers = 1
        DetectorThreads = 0
        EmbedderThreads = 0
    }
)
$internalResults = [System.Collections.Generic.List[object]]::new()
foreach ($profile in $profiles) {
    Write-Host "Benchmarking provider profile: $($profile.Name)"
    $result = Invoke-Profile `
        -Profile $profile `
        -Images $images `
        -WorkerRoot $workerRoot `
        -WorkerPort $Port `
        -ProfileWarmupCount $WarmupCount
    $internalResults.Add($result)
}

$candidate = $internalResults[0]
$recommended = $candidate
$selectionResult = "OpenVINO CPU 1xauto is accepted only when errors and the 300ms mean/p95 target all pass."

$processor = Get-CimInstance Win32_Processor | Select-Object -First 1
$computer = Get-CimInstance Win32_ComputerSystem
$targetCpuDetected = ([string]$processor.Name -match "N100")
$publicResults = @(
    foreach ($result in $internalResults) {
        [ordered]@{
            name = $result.Name
            provider = $result.Provider
            detector_workers = $result.DetectorWorkers
            detector_threads_per_session = $result.DetectorThreads
            embedder_threads = $result.EmbedderThreads
            startup_ms = $result.StartupMs
            latency_ms = [ordered]@{
                mean = $result.MeanMs
                p50 = $result.P50Ms
                p95 = $result.P95Ms
                p99 = $result.P99Ms
            }
            peak_working_set_bytes = $result.PeakWorkingSetBytes
            status_counts = $result.StatusCounts
            segmentation_status_counts = $result.SegmentationStatusCounts
            segmentation_count = $result.SegmentationCount
            full_path_count = $result.FullPathCount
            error_count = $result.ErrorCount
        }
    }
)
$responseContractSafe = (
    ($internalResults | Where-Object { $_.ErrorCount -gt 0 }).Count -eq 0
)
$passes = (
    $targetCpuDetected -and
    $responseContractSafe -and
    $candidate.FullPathCount -ge $MinimumFullPathImages -and
    $candidate.MeanMs -le 300 -and
    $candidate.P95Ms -le 300
)
$report = [ordered]@{
    schema_version = "1.0"
    evaluation = "bixolon_worker_n100_openvino_1xauto"
    product_version = $script:ExpectedVersion
    provider = "openvino"
    hardware = [ordered]@{
        cpu_name = $processor.Name
        cores = $processor.NumberOfCores
        logical_processors = $processor.NumberOfLogicalProcessors
        total_physical_memory_bytes = [long]$computer.TotalPhysicalMemory
        target_cpu_detected = $targetCpuDetected
    }
    sample_count = $images.Count
    warmup_count = $WarmupCount
    minimum_full_path_count = $MinimumFullPathImages
    response_contract_safe = $responseContractSafe
    cross_provider_parity_checked = $false
    profiles = $publicResults
    recommended_profile = [ordered]@{
        name = $recommended.Name
        provider = $recommended.Provider
        detector_workers = $recommended.DetectorWorkers
        detector_threads_per_session = $recommended.DetectorThreads
        embedder_threads = $recommended.EmbedderThreads
    }
    target = [ordered]@{
        full_path_latency_ms = 300
        recommended_mean_ms = $recommended.MeanMs
        recommended_p95_ms = $recommended.P95Ms
        mean_within_300ms = ($recommended.MeanMs -le 300)
        p95_within_300ms = ($recommended.P95Ms -le 300)
    }
    selection = "fixed OpenVINO CPU 1xauto"
    selection_result = $selectionResult
    passes = $passes
    privacy = [ordered]@{
        image_paths_recorded = $false
        image_bytes_recorded = $false
    }
    limitation = "Diagnostic measurement only; not an SLA or certification."
}

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $resolvedOutput
if ($outputDirectory) {
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
}
$json = $report | ConvertTo-Json -Depth 12
[System.IO.File]::WriteAllText(
    $resolvedOutput,
    $json + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host $json
Write-Host "N100 benchmark result: $resolvedOutput"
if (-not $passes) {
    throw "N100 benchmark did not satisfy the hardware, full-path, response, and 300ms checks."
}
