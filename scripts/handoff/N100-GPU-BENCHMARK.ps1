param(
    [Parameter(Mandatory = $true)]
    [string]$ImageDirectory,
    [string]$OutputPath = "",
    [int]$Port = 8188,
    [int]$MinimumImages = 30,
    [int]$MinimumFullPathImages = 10,
    [ValidateRange(0, 10)]
    [int]$WarmupImageCount = 3,
    [ValidateRange(1.0, 10.0)]
    [double]$MinimumSpeedupRatio = 1.05,
    [ValidateRange(1, 600000)]
    [int]$MaximumStartupMs = 30000,
    [ValidateRange(1, 8589934592)]
    [long]$MaximumWorkingSetBytes = 2147483648,
    [ValidateRange(1.0, 10.0)]
    [double]$MaximumMemoryIncreaseRatio = 1.35,
    [ValidateRange(1.0, 60000.0)]
    [double]$MaximumFullPathLatencyMs = 300.0,
    [string]$ExpectedVersion = "0.1.3"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $PSScriptRoot "n100-0.1.3-openvino-device-matrix.json"
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

function Get-LatencySummary {
    param([double[]]$Values)
    if ($Values.Count -eq 0) {
        return [ordered]@{
            sample_count = 0
            mean = $null
            p50 = $null
            p95 = $null
            p99 = $null
        }
    }
    return [ordered]@{
        sample_count = $Values.Count
        mean = [Math]::Round(($Values | Measure-Object -Average).Average, 3)
        p50 = Get-Percentile -Values $Values -Probability 0.50
        p95 = Get-Percentile -Values $Values -Probability 0.95
        p99 = Get-Percentile -Values $Values -Probability 0.99
    }
}

function Get-SemanticPayload {
    param($Body)
    $segmentations = @(
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
            [ordered]@{
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
    )
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
        if ($null -ne $item.confidence) {
            $values.Add([double]$item.confidence)
        }
        foreach ($candidate in @($item.top3)) {
            if ($null -ne $candidate.confidence) {
                $values.Add([double]$candidate.confidence)
            }
        }
    }
    return $values.ToArray()
}

function Test-VersionContract {
    param($Body, [string]$Version)
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
        $property = $Body.PSObject.Properties[$name]
        if ($null -eq $property) {
            return $false
        }
        if ($null -ne $property.Value -and [string]$property.Value -ne $Version) {
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
    $response = $null
    try {
        $bytes = [System.IO.File]::ReadAllBytes($Image.FullName)
        $fileContent = [System.Net.Http.ByteArrayContent]::new($bytes)
        $mediaType = if ($Image.Extension.ToLowerInvariant() -eq ".png") {
            "image/png"
        }
        else {
            "image/jpeg"
        }
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
        if ($null -ne $response) {
            $response.Dispose()
        }
        $multipart.Dispose()
    }
}

function Invoke-Profile {
    param(
        $Profile,
        [System.IO.FileInfo[]]$Images,
        [string]$WorkerRoot,
        [int]$WorkerPort,
        [string]$Version,
        [int]$WarmupCount
    )
    $workerExecutable = Join-Path $WorkerRoot "bixolon-worker.exe"
    $environment = [ordered]@{
        BIXOLON_PACKAGE_DIR = Join-Path $WorkerRoot "model-package"
        BIXOLON_CATALOG_DIR = Join-Path $WorkerRoot "store-catalog"
        BIXOLON_PROVIDER = [string]$Profile.DetectorProvider
        BIXOLON_EMBEDDER_PROVIDER = [string]$Profile.EmbedderProvider
        BIXOLON_HOST = "127.0.0.1"
        BIXOLON_PORT = [string]$WorkerPort
        BIXOLON_REQUEST_TIMEOUT_SECONDS = "60"
        BIXOLON_CPU_DETECTOR_WORKERS = "1"
        BIXOLON_CPU_DETECTOR_INTRA_OP_THREADS = "4"
        BIXOLON_CPU_EMBEDDER_INTRA_OP_THREADS = [string]$Profile.EmbedderThreads
        BIXOLON_LOG_TO_STDERR = "1"
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
    $logPath = Join-Path (
        [System.IO.Path]::GetTempPath()
    ) ("bixolon-n100-" + [Guid]::NewGuid().ToString("N") + ".jsonl")
    try {
        $startup = [System.Diagnostics.Stopwatch]::StartNew()
        $startArguments = @{
            FilePath = $workerExecutable
            WorkingDirectory = $WorkerRoot
            WindowStyle = "Hidden"
            PassThru = $true
            RedirectStandardError = $logPath
        }
        $process = Start-Process @startArguments
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
                $ready = $null
            }
            Start-Sleep -Milliseconds 200
        }
        $startup.Stop()
        if ($null -eq $ready -or $ready.status -ne "ready") {
            throw "Worker readiness timed out for profile $($Profile.Name)."
        }
        if ($ready.provider -ne $Profile.ExpectedProvider) {
            throw "Provider mismatch for profile $($Profile.Name): $($ready.provider)"
        }
        if (-not (Test-VersionContract -Body $ready -Version $Version)) {
            throw "Version contract mismatch for profile $($Profile.Name)."
        }

        $client = [System.Net.Http.HttpClient]::new()
        $client.Timeout = [TimeSpan]::FromSeconds(65)
        for ($warmupIndex = 0; $warmupIndex -lt $WarmupCount; $warmupIndex++) {
            $warmup = Invoke-Scan `
                -Client $client `
                -Image $Images[$warmupIndex % $Images.Count] `
                -BaseUrl $baseUrl
            if (
                $warmup.HttpStatus -lt 200 -or
                $warmup.HttpStatus -ge 300 -or
                $warmup.Body.status -eq "ERROR"
            ) {
                throw "Warm-up request failed for profile $($Profile.Name)."
            }
        }
        $clientAll = [System.Collections.Generic.List[double]]::new()
        $clientFullPath = [System.Collections.Generic.List[double]]::new()
        $workerAll = [System.Collections.Generic.List[double]]::new()
        $workerFullPath = [System.Collections.Generic.List[double]]::new()
        $responses = [System.Collections.Generic.List[object]]::new()
        $statusCounts = [ordered]@{
            SEGMENTATION = 0
            IMAGE_RECAPTURE = 0
            ERROR = 0
        }
        $errorCount = 0
        foreach ($image in $Images) {
            $scan = Invoke-Scan -Client $client -Image $image -BaseUrl $baseUrl
            $responses.Add($scan.Body)
            $clientAll.Add([double]$scan.ElapsedMs)
            if ($null -ne $scan.Body.processing_time_ms) {
                $workerAll.Add([double]$scan.Body.processing_time_ms)
            }
            if ($statusCounts.Contains($scan.Body.status)) {
                $statusCounts[$scan.Body.status]++
            }
            if ($scan.Body.status -eq "SEGMENTATION") {
                $clientFullPath.Add([double]$scan.ElapsedMs)
                if ($null -ne $scan.Body.processing_time_ms) {
                    $workerFullPath.Add([double]$scan.Body.processing_time_ms)
                }
            }
            if (
                $scan.HttpStatus -lt 200 -or
                $scan.HttpStatus -ge 300 -or
                $scan.Body.status -eq "ERROR" -or
                -not (Test-VersionContract -Body $scan.Body -Version $Version)
            ) {
                $errorCount++
            }
        }

        $process.Refresh()
        $peakWorkingSetBytes = [long]$process.PeakWorkingSet64
        if (-not $process.HasExited) {
            $process.Kill()
            $process.WaitForExit()
        }
        $process.Dispose()
        $process = $null
        $detectorLatencies = [System.Collections.Generic.List[double]]::new()
        $classifierLatencies = [System.Collections.Generic.List[double]]::new()
        $scanEntries = [System.Collections.Generic.List[object]]::new()
        if (Test-Path -LiteralPath $logPath -PathType Leaf) {
            foreach ($line in Get-Content -LiteralPath $logPath) {
                try {
                    $entry = $line | ConvertFrom-Json
                }
                catch {
                    continue
                }
                if ([string]$entry.message -ne "scan_complete") {
                    continue
                }
                $scanEntries.Add($entry)
            }
            foreach ($entry in @($scanEntries | Select-Object -Last $Images.Count)) {
                if ($null -ne $entry.detector_ms) {
                    $detectorLatencies.Add([double]$entry.detector_ms)
                }
                if ($null -ne $entry.classifier_ms -and [double]$entry.classifier_ms -gt 0) {
                    $classifierLatencies.Add([double]$entry.classifier_ms)
                }
            }
        }
        return [pscustomobject]@{
            Name = $Profile.Name
            DetectorProvider = $Profile.DetectorProvider
            EmbedderProvider = $Profile.EmbedderProvider
            ExpectedProvider = $Profile.ExpectedProvider
            StartupMs = [Math]::Round($startup.Elapsed.TotalMilliseconds, 3)
            ClientAll = Get-LatencySummary -Values $clientAll.ToArray()
            ClientFullPath = Get-LatencySummary -Values $clientFullPath.ToArray()
            WorkerAll = Get-LatencySummary -Values $workerAll.ToArray()
            WorkerFullPath = Get-LatencySummary -Values $workerFullPath.ToArray()
            Detector = Get-LatencySummary -Values $detectorLatencies.ToArray()
            Classifier = Get-LatencySummary -Values $classifierLatencies.ToArray()
            PeakWorkingSetBytes = $peakWorkingSetBytes
            StatusCounts = $statusCounts
            FullPathCount = $clientFullPath.Count
            ErrorCount = $errorCount
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
        if (Test-Path -LiteralPath $logPath -PathType Leaf) {
            Remove-Item -LiteralPath $logPath -Force
        }
    }
}

function Get-PublicProfile {
    param($Result, [double]$MaximumLatencyMs)
    return [ordered]@{
        name = $Result.Name
        completed = ($null -eq $Result.FailureCode)
        failure_code = $Result.FailureCode
        detector_provider = $Result.DetectorProvider
        embedder_provider = $Result.EmbedderProvider
        detector_workers = 1
        detector_threads_per_session = 4
        embedder_threads = if ($Result.EmbedderProvider -eq "openvino_gpu") { 0 } else { 4 }
        startup_ms = $Result.StartupMs
        client_total_ms = [ordered]@{
            all_requests = $Result.ClientAll
            full_path = $Result.ClientFullPath
        }
        worker_processing_ms = [ordered]@{
            all_requests = $Result.WorkerAll
            full_path = $Result.WorkerFullPath
        }
        stage_latency_ms = [ordered]@{
            detector = $Result.Detector
            classifier = $Result.Classifier
        }
        peak_working_set_bytes = $Result.PeakWorkingSetBytes
        status_counts = $Result.StatusCounts
        full_path_count = $Result.FullPathCount
        error_count = $Result.ErrorCount
        target = [ordered]@{
            maximum_full_path_latency_ms = $MaximumLatencyMs
            mean_within_target = (
                $null -ne $Result.ClientFullPath.mean -and
                $Result.ClientFullPath.mean -le $MaximumLatencyMs
            )
            p95_within_target = (
                $null -ne $Result.ClientFullPath.p95 -and
                $Result.ClientFullPath.p95 -le $MaximumLatencyMs
            )
        }
    }
}

$packageRoot = $PSScriptRoot
$workerRoot = Join-Path $packageRoot "worker"
$workerExecutable = Join-Path $workerRoot "bixolon-worker.exe"
$runtimeMetadataPath = Join-Path $workerRoot "model-package/metadata.json"
$catalogChecksumsPath = Join-Path $workerRoot "store-catalog/checksums.json"
$resolvedImageDirectory = [System.IO.Path]::GetFullPath($ImageDirectory)
$resolvedOutputPath = [System.IO.Path]::GetFullPath($OutputPath)

foreach ($requiredPath in @($workerExecutable, $runtimeMetadataPath, $catalogChecksumsPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required test package file is missing: $requiredPath"
    }
}
if (-not (Test-Path -LiteralPath $resolvedImageDirectory -PathType Container)) {
    throw "Benchmark image directory is missing: $resolvedImageDirectory"
}
$runtimeMetadata = Get-Content -Raw -LiteralPath $runtimeMetadataPath | ConvertFrom-Json
if ([string]$runtimeMetadata.worker_version -ne $ExpectedVersion) {
    throw "Expected Runtime $ExpectedVersion but found $($runtimeMetadata.worker_version)."
}
$images = @(
    Get-ChildItem -LiteralPath $resolvedImageDirectory -File |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png") } |
        Sort-Object Name
)
if ($images.Count -lt $MinimumImages) {
    throw "N100 GPU benchmark requires at least $MinimumImages JPEG/PNG images."
}

$profiles = @(
    [pscustomobject]@{
        Name = "openvino-cpu-only"
        DetectorProvider = "openvino"
        EmbedderProvider = "same"
        EmbedderThreads = 4
        ExpectedProvider = "openvino"
    },
    [pscustomobject]@{
        Name = "openvino-cpu-detector-intel-gpu-embedder"
        DetectorProvider = "openvino"
        EmbedderProvider = "openvino_gpu"
        EmbedderThreads = 0
        ExpectedProvider = "openvino+openvino_gpu"
    }
)

$baselineProfile = $profiles[0]
$hybridProfile = $profiles[1]
Write-Host "Testing profile: $($baselineProfile.Name)"
$baselineResult = Invoke-Profile `
    -Profile $baselineProfile `
    -Images $images `
    -WorkerRoot $workerRoot `
    -WorkerPort $Port `
    -Version $ExpectedVersion `
    -WarmupCount $WarmupImageCount

Write-Host "Testing profile: $($hybridProfile.Name)"
try {
    $hybridResult = Invoke-Profile `
        -Profile $hybridProfile `
        -Images $images `
        -WorkerRoot $workerRoot `
        -WorkerPort $Port `
        -Version $ExpectedVersion `
        -WarmupCount $WarmupImageCount
}
catch {
    Write-Warning "OpenVINO Intel GPU Embedder profile failed; the result will recommend the CPU-only profile."
    $emptySummary = Get-LatencySummary -Values ([double[]]@())
    $hybridResult = [pscustomobject]@{
        Name = $hybridProfile.Name
        DetectorProvider = $hybridProfile.DetectorProvider
        EmbedderProvider = $hybridProfile.EmbedderProvider
        ExpectedProvider = $hybridProfile.ExpectedProvider
        StartupMs = $null
        ClientAll = $emptySummary
        ClientFullPath = $emptySummary
        WorkerAll = $emptySummary
        WorkerFullPath = $emptySummary
        Detector = $emptySummary
        Classifier = $emptySummary
        PeakWorkingSetBytes = $null
        StatusCounts = [ordered]@{ SEGMENTATION = 0; IMAGE_RECAPTURE = 0; ERROR = 1 }
        FullPathCount = 0
        ErrorCount = 1
        Responses = @()
        FailureCode = "OPENVINO_GPU_PROFILE_FAILED"
    }
}

$semanticMismatchIndexes = [System.Collections.Generic.List[int]]::new()
$confidenceVectorMismatchIndexes = [System.Collections.Generic.List[int]]::new()
$maximumConfidenceDelta = $null
$parityEvaluated = $null -eq $hybridResult.FailureCode
if ($parityEvaluated) {
    $maximumConfidenceDelta = 0.0
    for ($index = 0; $index -lt $images.Count; $index++) {
        $baselineBody = $baselineResult.Responses[$index]
        $hybridBody = $hybridResult.Responses[$index]
        if ((Get-SemanticPayload $baselineBody) -ne (Get-SemanticPayload $hybridBody)) {
            $semanticMismatchIndexes.Add($index + 1)
        }
        $baselineConfidence = @(Get-ConfidenceVector $baselineBody)
        $hybridConfidence = @(Get-ConfidenceVector $hybridBody)
        if ($baselineConfidence.Count -ne $hybridConfidence.Count) {
            $confidenceVectorMismatchIndexes.Add($index + 1)
            continue
        }
        for (
            $confidenceIndex = 0;
            $confidenceIndex -lt $baselineConfidence.Count;
            $confidenceIndex++
        ) {
            $delta = [Math]::Abs(
                [double]$baselineConfidence[$confidenceIndex] -
                [double]$hybridConfidence[$confidenceIndex]
            )
            $maximumConfidenceDelta = [Math]::Max($maximumConfidenceDelta, $delta)
        }
    }
}

$baselineMean = $baselineResult.ClientFullPath.mean
$baselineP95 = $baselineResult.ClientFullPath.p95
$hybridMean = $hybridResult.ClientFullPath.mean
$hybridP95 = $hybridResult.ClientFullPath.p95
$meanSpeedup = if ($null -ne $hybridMean -and $hybridMean -gt 0) {
    [Math]::Round([double]$baselineMean / [double]$hybridMean, 3)
}
else {
    $null
}
$p95Speedup = if ($null -ne $hybridP95 -and $hybridP95 -gt 0) {
    [Math]::Round([double]$baselineP95 / [double]$hybridP95, 3)
}
else {
    $null
}
$paritySafe = (
    $parityEvaluated -and
    $semanticMismatchIndexes.Count -eq 0 -and
    $confidenceVectorMismatchIndexes.Count -eq 0 -and
    $maximumConfidenceDelta -le 0.00001
)
$hybridBeneficial = (
    $null -ne $meanSpeedup -and
    $null -ne $p95Speedup -and
    $meanSpeedup -ge $MinimumSpeedupRatio -and
    $p95Speedup -ge $MinimumSpeedupRatio
)
$resourceSafe = (
    $null -ne $baselineResult.PeakWorkingSetBytes -and
    $baselineResult.PeakWorkingSetBytes -gt 0 -and
    $null -ne $hybridResult.PeakWorkingSetBytes -and
    $null -ne $hybridResult.StartupMs -and
    $hybridResult.StartupMs -le $MaximumStartupMs -and
    $hybridResult.PeakWorkingSetBytes -le $MaximumWorkingSetBytes -and
    $hybridResult.PeakWorkingSetBytes -le (
        $baselineResult.PeakWorkingSetBytes * $MaximumMemoryIncreaseRatio
    )
)
$baselineTargetMet = (
    $null -ne $baselineMean -and
    $null -ne $baselineP95 -and
    $baselineMean -le $MaximumFullPathLatencyMs -and
    $baselineP95 -le $MaximumFullPathLatencyMs
)
$hybridTargetMet = (
    $null -ne $hybridMean -and
    $null -ne $hybridP95 -and
    $hybridMean -le $MaximumFullPathLatencyMs -and
    $hybridP95 -le $MaximumFullPathLatencyMs
)

$processor = Get-CimInstance Win32_Processor | Select-Object -First 1
$computer = Get-CimInstance Win32_ComputerSystem
$videoControllers = @(
    Get-CimInstance Win32_VideoController | ForEach-Object {
        [ordered]@{
            name = $_.Name
            adapter_ram_bytes = if ($null -eq $_.AdapterRAM) { $null } else { [long]$_.AdapterRAM }
            driver_version = $_.DriverVersion
        }
    }
)
$targetCpuDetected = ([string]$processor.Name -match "N100")
$targetIntelGpuDetected = @(
    $videoControllers | Where-Object { [string]$_.name -match "(?i)Intel.*(UHD|Graphics)" }
).Count -gt 0
$passes = (
    $targetCpuDetected -and
    $targetIntelGpuDetected -and
    $baselineResult.FullPathCount -ge $MinimumFullPathImages -and
    $hybridResult.FullPathCount -ge $MinimumFullPathImages -and
    $baselineResult.ErrorCount -eq 0 -and
    $hybridResult.ErrorCount -eq 0 -and
    $paritySafe
)
$recommendedProvider = if (
    $passes -and $hybridBeneficial -and $resourceSafe -and $hybridTargetMet
) {
    "openvino+openvino_gpu"
}
else {
    "openvino"
}

$report = [ordered]@{
    schema_version = "1.1"
    evaluation = "bixolon_worker_n100_openvino_cpu_vs_intel_gpu_embedder"
    product_version = $ExpectedVersion
    completed = $true
    passes = $passes
    hardware = [ordered]@{
        cpu_name = $processor.Name
        cores = $processor.NumberOfCores
        logical_processors = $processor.NumberOfLogicalProcessors
        total_physical_memory_bytes = [long]$computer.TotalPhysicalMemory
        target_cpu_detected = $targetCpuDetected
        target_intel_gpu_detected = $targetIntelGpuDetected
        video_controllers = $videoControllers
        openvino_gpu_device = "GPU"
    }
    input = [ordered]@{
        sample_count = $images.Count
        minimum_image_count = $MinimumImages
        minimum_full_path_count = $MinimumFullPathImages
        warmup_image_count = $WarmupImageCount
    }
    selection_policy = [ordered]@{
        minimum_mean_speedup_ratio = $MinimumSpeedupRatio
        minimum_p95_speedup_ratio = $MinimumSpeedupRatio
        maximum_startup_ms = $MaximumStartupMs
        maximum_peak_working_set_bytes = $MaximumWorkingSetBytes
        maximum_memory_increase_ratio = $MaximumMemoryIncreaseRatio
        maximum_full_path_mean_ms = $MaximumFullPathLatencyMs
        maximum_full_path_p95_ms = $MaximumFullPathLatencyMs
        require_semantic_and_confidence_parity = $true
    }
    execution_contract = [ordered]@{
        same_worker_executable = $true
        same_runtime_catalog_and_policy = $true
        cpu_only = [ordered]@{
            detector = "OpenVINOExecutionProvider:CPU"
            primary_embedder = "OpenVINOExecutionProvider:CPU"
            rotation_180_embedder = "OpenVINOExecutionProvider:CPU"
            independent_verifier_embedder = "OpenVINOExecutionProvider:CPU"
        }
        cpu_detector_gpu_embedder = [ordered]@{
            detector = "OpenVINOExecutionProvider:CPU"
            primary_embedder = "OpenVINOExecutionProvider:GPU"
            rotation_180_embedder = "OpenVINOExecutionProvider:GPU"
            independent_verifier_embedder = "OpenVINOExecutionProvider:GPU"
            silent_cpu_fallback_allowed = $false
        }
    }
    integrity = [ordered]@{
        runtime_metadata_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeMetadataPath).Hash.ToLowerInvariant()
        catalog_checksums_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $catalogChecksumsPath).Hash.ToLowerInvariant()
        model_graph_or_weight_changed = $false
        decision_policy_changed = $false
    }
    profiles = [ordered]@{
        openvino_cpu_only = Get-PublicProfile `
            -Result $baselineResult `
            -MaximumLatencyMs $MaximumFullPathLatencyMs
        openvino_cpu_detector_intel_gpu_embedder = Get-PublicProfile `
            -Result $hybridResult `
            -MaximumLatencyMs $MaximumFullPathLatencyMs
    }
    parity = [ordered]@{
        evaluated = $parityEvaluated
        safe = $paritySafe
        semantic_mismatch_count = $semanticMismatchIndexes.Count
        semantic_mismatch_indexes = $semanticMismatchIndexes.ToArray()
        confidence_vector_mismatch_count = $confidenceVectorMismatchIndexes.Count
        confidence_vector_mismatch_indexes = $confidenceVectorMismatchIndexes.ToArray()
        maximum_confidence_delta = $maximumConfidenceDelta
        confidence_tolerance = 0.00001
    }
    comparison = [ordered]@{
        provider_initialization_safe = ($null -eq $hybridResult.FailureCode)
        hybrid_mean_improved = ($null -ne $meanSpeedup -and $meanSpeedup -gt 1.0)
        hybrid_p95_improved = ($null -ne $p95Speedup -and $p95Speedup -gt 1.0)
        hybrid_meets_minimum_speedup = $hybridBeneficial
        hybrid_resource_safe = $resourceSafe
        cpu_only_target_met = $baselineTargetMet
        gpu_embedder_target_met = $hybridTargetMet
        target_met_by_any_profile = ($baselineTargetMet -or $hybridTargetMet)
        full_path_mean_speedup_ratio = $meanSpeedup
        full_path_p95_speedup_ratio = $p95Speedup
        recommended_provider = $recommendedProvider
    }
    privacy = [ordered]@{
        image_paths_recorded = $false
        image_bytes_recorded = $false
    }
    limitation = "Diagnostic measurement only; not an SLA, certification, or deployment approval."
}

[System.IO.Directory]::CreateDirectory((Split-Path -Parent $resolvedOutputPath)) | Out-Null
$json = $report | ConvertTo-Json -Depth 20
[System.IO.File]::WriteAllText(
    $resolvedOutputPath,
    $json + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host ""
Write-Host "OpenVINO CPU-only full-path mean/p95: $baselineMean / $baselineP95 ms"
Write-Host "CPU Detector + Intel GPU Embedder full-path mean/p95: $hybridMean / $hybridP95 ms"
Write-Host "GPU Embedder mean/p95 speedup: $meanSpeedup x / $p95Speedup x"
Write-Host "GPU Embedder resource safe: $resourceSafe"
Write-Host "Parity safe: $paritySafe"
Write-Host "Recommended provider: $recommendedProvider"
Write-Host "Result: $resolvedOutputPath"
