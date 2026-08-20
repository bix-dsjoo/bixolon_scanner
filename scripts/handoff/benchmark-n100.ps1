param(
    [Parameter(Mandatory = $true)]
    [string]$ImageDirectory,
    [string]$OutputPath = "",
    [int]$Port = 8188,
    [int]$MinimumImages = 30,
    [int]$MinimumFullPathImages = 10
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

function Get-SemanticPayload {
    param($Body)
    $segmentations = @()
    foreach ($item in @($Body.segmentations)) {
        $prediction = $null
        if ($null -ne $item.prediction) {
            $prediction = [ordered]@{
                class_id = $item.prediction.class_id
                class_name = $item.prediction.class_name
            }
        }
        $top3 = @(
            foreach ($candidate in @($item.top3)) {
                [ordered]@{
                    class_id = $candidate.class_id
                    class_name = $candidate.class_name
                }
            }
        )
        $segmentations += [ordered]@{
            segmentation_id = $item.segmentation_id
            bbox = [ordered]@{
                x = $item.bbox.x
                y = $item.bbox.y
                width = $item.bbox.width
                height = $item.bbox.height
            }
            status = $item.status
            reason_codes = @($item.reason_codes)
            prediction = $prediction
            top3 = $top3
        }
    }
    return ([ordered]@{
        status = $Body.status
        reason_codes = @($Body.reason_codes)
        segmentations = $segmentations
    } | ConvertTo-Json -Depth 12 -Compress)
}

function Get-ConfidenceVector {
    param($Body)
    $values = [System.Collections.Generic.List[double]]::new()
    foreach ($item in @($Body.segmentations)) {
        $values.Add([double]$item.confidence)
        foreach ($candidate in @($item.top3)) {
            $values.Add([double]$candidate.confidence)
        }
    }
    return $values.ToArray()
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
        if ($null -ne $value -and [string]$value -ne "0.0.2") {
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
        $BaselineResponses
    )
    $workerExecutable = Join-Path $WorkerRoot "bixolon-worker.exe"
    $environment = [ordered]@{
        BIXOLON_PACKAGE_DIR = Join-Path $WorkerRoot "model-package"
        BIXOLON_CATALOG_DIR = Join-Path $WorkerRoot "store-catalog"
        BIXOLON_PROVIDER = "cpu"
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
        if ($ready.provider -ne "cpu" -or -not (Test-VersionContract $ready)) {
            throw "Worker readiness contract mismatch for profile $($Profile.Name)."
        }

        $client = [System.Net.Http.HttpClient]::new()
        $client.Timeout = [TimeSpan]::FromSeconds(65)
        $latencies = [System.Collections.Generic.List[double]]::new()
        $responses = [System.Collections.Generic.List[object]]::new()
        $statusCounts = [ordered]@{
            SEGMENTATION = 0
            IMAGE_RECAPTURE = 0
            ERROR = 0
        }
        $errorCount = 0
        foreach ($image in $Images) {
            $scan = Invoke-Scan -Client $client -Image $image -BaseUrl $baseUrl
            $latencies.Add($scan.ElapsedMs)
            $responses.Add($scan.Body)
            if ($statusCounts.Contains($scan.Body.status)) {
                $statusCounts[$scan.Body.status]++
            }
            if (
                $scan.HttpStatus -lt 200 -or
                $scan.HttpStatus -ge 300 -or
                $scan.Body.status -eq "ERROR" -or
                -not (Test-VersionContract $scan.Body)
            ) {
                $errorCount++
            }
            $process.Refresh()
        }

        $paritySafe = $true
        $maximumConfidenceDelta = 0.0
        if ($null -ne $BaselineResponses) {
            for ($index = 0; $index -lt $responses.Count; $index++) {
                $baseline = $BaselineResponses[$index]
                $candidate = $responses[$index]
                if ((Get-SemanticPayload $baseline) -ne (Get-SemanticPayload $candidate)) {
                    $paritySafe = $false
                    continue
                }
                $baselineConfidence = @(Get-ConfidenceVector $baseline)
                $candidateConfidence = @(Get-ConfidenceVector $candidate)
                if ($baselineConfidence.Count -ne $candidateConfidence.Count) {
                    $paritySafe = $false
                    continue
                }
                for ($confidenceIndex = 0; $confidenceIndex -lt $baselineConfidence.Count; $confidenceIndex++) {
                    $delta = [Math]::Abs(
                        [double]$baselineConfidence[$confidenceIndex] -
                        [double]$candidateConfidence[$confidenceIndex]
                    )
                    $maximumConfidenceDelta = [Math]::Max($maximumConfidenceDelta, $delta)
                    if ($delta -gt 0.00001) {
                        $paritySafe = $false
                    }
                }
            }
        }

        $process.Refresh()
        $values = $latencies.ToArray()
        return [pscustomobject]@{
            Name = $Profile.Name
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
            FullPathCount = [int]$statusCounts.SEGMENTATION
            ErrorCount = $errorCount
            ParitySafe = $paritySafe
            MaximumConfidenceDelta = $maximumConfidenceDelta
            Responses = $responses.ToArray()
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
$resolvedImageDirectory = [System.IO.Path]::GetFullPath($ImageDirectory)
if (-not (Test-Path -LiteralPath $workerExecutable -PathType Leaf)) {
    throw "CPU Worker executable is missing: $workerExecutable"
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

$profiles = @(
    [pscustomobject]@{ Name = "sequential-4"; DetectorWorkers = 1; DetectorThreads = 4; EmbedderThreads = 4 },
    [pscustomobject]@{ Name = "parallel-2x2"; DetectorWorkers = 2; DetectorThreads = 2; EmbedderThreads = 4 },
    [pscustomobject]@{ Name = "parallel-4x1"; DetectorWorkers = 4; DetectorThreads = 1; EmbedderThreads = 4 }
)
$internalResults = [System.Collections.Generic.List[object]]::new()
$baselineResponses = $null
foreach ($profile in $profiles) {
    Write-Host "Benchmarking CPU profile: $($profile.Name)"
    $result = Invoke-Profile `
        -Profile $profile `
        -Images $images `
        -WorkerRoot $workerRoot `
        -WorkerPort $Port `
        -BaselineResponses $baselineResponses
    $internalResults.Add($result)
    if ($null -eq $baselineResponses) {
        $baselineResponses = $result.Responses
    }
}

$safeResults = @(
    $internalResults |
        Where-Object { $_.ParitySafe -and $_.ErrorCount -eq 0 }
)
$baselineResult = $internalResults[0]
$beneficialParallelResults = @(
    $safeResults |
        Where-Object {
            $_.DetectorWorkers -gt 1 -and $_.P95Ms -lt $baselineResult.P95Ms
        }
)
if ($beneficialParallelResults.Count -eq 0) {
    $recommended = $internalResults[0]
    $selectionResult = "No parity-safe parallel profile improved p95; use the 1 x 4 fallback."
}
else {
    $selectionCandidates = @($beneficialParallelResults + $baselineResult)
    $minimumP95 = ($selectionCandidates | Measure-Object -Property P95Ms -Minimum).Minimum
    $recommended = $selectionCandidates |
        Where-Object { $_.P95Ms -le $minimumP95 * 1.05 } |
        Sort-Object PeakWorkingSetBytes, DetectorWorkers |
        Select-Object -First 1
    $selectionResult = "Selected the lowest parity-safe p95; within 5 percent preferred lower peak memory."
}

$processor = Get-CimInstance Win32_Processor | Select-Object -First 1
$computer = Get-CimInstance Win32_ComputerSystem
$publicResults = @(
    foreach ($result in $internalResults) {
        [ordered]@{
            name = $result.Name
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
            full_path_count = $result.FullPathCount
            error_count = $result.ErrorCount
            parity_safe = $result.ParitySafe
            maximum_confidence_delta = $result.MaximumConfidenceDelta
        }
    }
)
$passes = (
    $safeResults.Count -gt 0 -and
    $internalResults[0].FullPathCount -ge $MinimumFullPathImages -and
    ($internalResults | Where-Object { $_.ErrorCount -gt 0 }).Count -eq 0
)
$report = [ordered]@{
    schema_version = "1.0"
    evaluation = "bixolon_worker_n100_cpu_profiles"
    product_version = "0.0.2"
    provider = "cpu"
    hardware = [ordered]@{
        cpu_name = $processor.Name
        cores = $processor.NumberOfCores
        logical_processors = $processor.NumberOfLogicalProcessors
        total_physical_memory_bytes = [long]$computer.TotalPhysicalMemory
        target_cpu_detected = ([string]$processor.Name -match "N100")
    }
    sample_count = $images.Count
    minimum_full_path_count = $MinimumFullPathImages
    profiles = $publicResults
    recommended_profile = [ordered]@{
        name = $recommended.Name
        detector_workers = $recommended.DetectorWorkers
        detector_threads_per_session = $recommended.DetectorThreads
        embedder_threads = $recommended.EmbedderThreads
    }
    selection = "lowest parity-safe p95; within 5 percent choose lower peak memory, then fewer detector workers"
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
    throw "N100 benchmark did not satisfy the sample, full-path, error, and parity checks."
}
