param(
    [Parameter(Mandatory = $true)]
    [string]$BundleZip,
    [string]$ScannerDataRoot = "$env:ProgramData\BIXOLON\Scanner",
    [switch]$Activate
)

$ErrorActionPreference = "Stop"

function Assert-SafeChildPath {
    param([string]$Path, [string]$Parent)
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $prefix = $resolvedParent + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe bundle destination: $resolvedPath"
    }
}

$resolvedZip = [System.IO.Path]::GetFullPath($BundleZip)
if (-not (Test-Path -LiteralPath $resolvedZip -PathType Leaf)) {
    throw "Bundle ZIP is missing: $resolvedZip"
}
$zipHashPath = "$resolvedZip.sha256"
if (Test-Path -LiteralPath $zipHashPath -PathType Leaf) {
    $expectedZipHash = ((Get-Content -Raw -LiteralPath $zipHashPath).Trim() -split "\s+")[0]
    $actualZipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedZip).Hash
    if ($actualZipHash -ne $expectedZipHash) {
        throw "Store Model Bundle ZIP checksum mismatch."
    }
}
$resolvedDataRoot = [System.IO.Path]::GetFullPath($ScannerDataRoot)
[System.IO.Directory]::CreateDirectory($resolvedDataRoot) | Out-Null
$temporaryRoot = Join-Path $resolvedDataRoot (".install-" + [Guid]::NewGuid().ToString("N"))
Assert-SafeChildPath -Path $temporaryRoot -Parent $resolvedDataRoot
[System.IO.Directory]::CreateDirectory($temporaryRoot) | Out-Null

try {
    Expand-Archive -LiteralPath $resolvedZip -DestinationPath $temporaryRoot
    $bundleRoots = @(
        Get-ChildItem -LiteralPath $temporaryRoot -Directory |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "store-bundle.json") }
    )
    if ($bundleRoots.Count -ne 1) {
        throw "ZIP must contain exactly one Store Model Bundle root."
    }
    $bundleRoot = $bundleRoots[0].FullName
    $identity = Get-Content -Raw -LiteralPath (Join-Path $bundleRoot "store-bundle.json") |
        ConvertFrom-Json
    $manifest = Get-Content -Raw -LiteralPath (Join-Path $bundleRoot "bundle-manifest.json") |
        ConvertFrom-Json

    $identitySchema = [string]$identity.schema_version
    $modelVersion = if ($identitySchema -eq "1.1") {
        [string]$identity.model_version
    } else {
        [string]$identity.product_version
    }
    if (
        $identitySchema -notin @("1.0", "1.1") -or
        [string]$identity.bundle_kind -ne "BIXOLON_STORE_MODEL" -or
        [string]$identity.provider -ne "cpu" -or
        [string]::IsNullOrWhiteSpace($modelVersion)
    ) {
        throw "Unsupported Store Model Bundle identity or provider."
    }
    if (Test-Path -LiteralPath (Join-Path $bundleRoot "store-catalog/signature.json")) {
        throw "signature.json is not allowed in an active Catalog."
    }

    $manifestPaths = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    foreach ($entry in $manifest.files) {
        $entryPath = ([string]$entry.path).Replace("\", "/")
        if (-not $manifestPaths.Add($entryPath)) {
            throw "Duplicate manifest path: $entryPath"
        }
        $candidate = Join-Path $bundleRoot $entryPath
        Assert-SafeChildPath -Path $candidate -Parent $bundleRoot
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Manifest file is missing: $entryPath"
        }
        $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
        if ($actualHash -ne ([string]$entry.sha256).ToLowerInvariant()) {
            throw "Checksum mismatch: $entryPath"
        }
    }
    if ($manifestPaths.Count -ne [int]$manifest.file_count) {
        throw "Manifest file_count does not match the file list."
    }
    $actualPaths = @(
        Get-ChildItem -LiteralPath $bundleRoot -File -Recurse |
            ForEach-Object {
                [System.IO.Path]::GetRelativePath($bundleRoot, $_.FullName).Replace("\", "/")
            } |
            Where-Object { $_ -ne "bundle-manifest.json" }
    )
    if ($actualPaths.Count -ne $manifestPaths.Count) {
        throw "Bundle contains missing or unlisted files."
    }
    foreach ($actualPath in $actualPaths) {
        if (-not $manifestPaths.Contains($actualPath)) {
            throw "Bundle contains an unlisted file: $actualPath"
        }
    }

    $destination = Join-Path $resolvedDataRoot (
        "bundles/{0}/{1}" -f [string]$identity.store_id, $modelVersion
    )
    Assert-SafeChildPath -Path $destination -Parent $resolvedDataRoot
    if (Test-Path -LiteralPath $destination) {
        throw "Bundle destination already exists: $destination"
    }
    [System.IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    Move-Item -LiteralPath $bundleRoot -Destination $destination

    if ($Activate) {
        $pointerPath = Join-Path $resolvedDataRoot "active-bundle.json"
        $temporaryPointer = "$pointerPath.$([Guid]::NewGuid().ToString('N')).tmp"
        $pointer = [ordered]@{
            schema_version = "1.1"
            store_id = [string]$identity.store_id
            model_version = $modelVersion
            bundle_path = $destination
        }
        $pointerJson = $pointer | ConvertTo-Json -Depth 4
        [System.IO.File]::WriteAllText(
            $temporaryPointer,
            $pointerJson + [Environment]::NewLine,
            [System.Text.UTF8Encoding]::new($false)
        )
        Move-Item -LiteralPath $temporaryPointer -Destination $pointerPath -Force
    }
    Write-Host "Store Model Bundle installed: $destination"
    if ($Activate) {
        Write-Host "Activated. Restart the Worker to load this bundle."
    }
} finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Assert-SafeChildPath -Path $temporaryRoot -Parent $resolvedDataRoot
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
