param(
    [Parameter(Mandatory=$true)][string]$ImageDirectory,
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [string]$InputManifest = (Join-Path $PSScriptRoot 'log132-image-sha256.json'),
    [string]$OutputDirectory = (Join-Path $env:LOCALAPPDATA ('BIXOLON/Benchmarks/' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http
$files = @(Get-ChildItem -LiteralPath $ImageDirectory -File | Where-Object { $_.Extension.ToLowerInvariant() -in @('.jpg','.jpeg','.png') } | Sort-Object Name)
if ($files.Count -ne 132) { throw 'The frozen log132 measurement requires exactly 132 images.' }
$expected = @((Get-Content -LiteralPath $InputManifest -Raw | ConvertFrom-Json).image_sha256 | Sort-Object)
$inputs = @($files | ForEach-Object {
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
    $hash = [System.Security.Cryptography.SHA256]::Create()
    try { $digest = [BitConverter]::ToString($hash.ComputeHash($bytes)).Replace('-','').ToLowerInvariant() } finally { $hash.Dispose() }
    @{ name=$_.Name; extension=$_.Extension.ToLowerInvariant(); bytes=$bytes; image_sha256=$digest }
})
if (($expected.Count -ne 132) -or ((@($inputs.image_sha256 | Sort-Object) -join ',') -ne ($expected -join ','))) { throw 'Image SHA-256 does not match the frozen log132 manifest.' }
[System.IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds(130)
function Send-Image([hashtable]$InputImage, [bool]$Measure) {
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $multipart = [System.Net.Http.MultipartFormDataContent]::new()
    try {
        $part = [System.Net.Http.ByteArrayContent]::new($InputImage.bytes)
        $part.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new($(if ($InputImage.extension -eq '.png') { 'image/png' } else { 'image/jpeg' }))
        $multipart.Add($part, 'image', $InputImage.name)
        $reply = $client.PostAsync(($BaseUrl.TrimEnd('/') + '/v1/scan'),$multipart).GetAwaiter().GetResult()
        try {
            $body = $reply.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
            $timer.Stop()
            if ($Measure) { return @{ image_sha256=$InputImage.image_sha256; elapsed_ms=$timer.Elapsed.TotalMilliseconds; http_status=[int]$reply.StatusCode; response=$body } }
        } finally { $reply.Dispose() }
    } finally { $multipart.Dispose() }
}
try {
    $ready = $client.GetStringAsync(($BaseUrl.TrimEnd('/') + '/health/ready')).GetAwaiter().GetResult() | ConvertFrom-Json
    $hardware = @{ processor=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors); gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion); ready=$ready; warmup=10; repetitions=3; concurrent_requests=1; image_count=$files.Count }
    $hardware | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'environment.json') -Encoding UTF8
    for ($i=0; $i -lt 10; $i++) { Send-Image $inputs[$i] $false }
    $encoding = [System.Text.UTF8Encoding]::new($false)
    $stream = [System.IO.StreamWriter]::new((Join-Path $OutputDirectory 'responses.jsonl'),$false,$encoding)
    try {
        for ($repeat=0; $repeat -lt 3; $repeat++) {
            foreach ($inputImage in $inputs) {
                $row = Send-Image $inputImage $true
                $row.repetition = $repeat
                $stream.WriteLine(($row | ConvertTo-Json -Depth 24 -Compress))
                $stream.Flush()
            }
        }
    } finally { $stream.Dispose() }
    $client.GetStringAsync(($BaseUrl.TrimEnd('/') + '/health/ready')).GetAwaiter().GetResult() | Set-Content -LiteralPath (Join-Path $OutputDirectory 'ready-end.json') -Encoding UTF8
    Write-Host "Measurement saved: $OutputDirectory"
    Write-Host 'Accuracy requires the frozen GT manifest; no accuracy claim is inferred from APPROVED counts.'
} finally { $client.Dispose() }
