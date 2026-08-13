param(
    [string]$Python = "python",
    [string]$Flutter = "flutter",
    [switch]$SkipFlutter
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param(
        [string]$Command,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    Push-Location $WorkingDirectory
    try {
        & $Command @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Command exited with code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

Invoke-Checked $Python @("-m", "ruff", "check", "src", "tests") $root
Invoke-Checked $Python @("-m", "ruff", "format", "--check", "src", "tests") $root
Invoke-Checked $Python @("-m", "pytest", "-q") $root

if (-not $SkipFlutter) {
    $app = Join-Path $root "apps\product_scanner"
    Invoke-Checked $Flutter @("pub", "get") $app
    Invoke-Checked $Flutter @("analyze") $app
    Invoke-Checked $Flutter @("test") $app
}
