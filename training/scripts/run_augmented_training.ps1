$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDirectory = Join-Path $workspace "outputs\prithvi_4band_augmented_refine"
$startedMarker = Join-Path $runDirectory "training.started"
$launcherLog = Join-Path $runDirectory "launcher.log"
$standardOutput = Join-Path $runDirectory "training.stdout.log"
$standardError = Join-Path $runDirectory "training.stderr.log"

try {
    Set-Location -LiteralPath $workspace
    New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
    if (Test-Path -LiteralPath $startedMarker) {
        exit 0
    }
    New-Item -ItemType File -Path $startedMarker -Force | Out-Null
    "Launcher started at $(Get-Date -Format o)" |
        Set-Content -LiteralPath $launcherLog

    $env:HF_HOME = Join-Path $workspace "outputs\cache\huggingface"
    $env:HF_HUB_OFFLINE = "1"
    $env:NO_ALBUMENTATIONS_UPDATE = "1"
    $env:MPLCONFIGDIR = Join-Path $workspace "outputs\cache\matplotlib"
    $env:PYTHONUNBUFFERED = "1"

    $ErrorActionPreference = "Continue"
    & "$workspace\.venv\Scripts\python.exe" `
        -m prithvi_crop.train `
        --config configs/prithvi_4band_augmented_refine.yaml `
        --resume none `
        1> $standardOutput `
        2> $standardError
    $trainingExitCode = $LASTEXITCODE
    "Python exited with code $trainingExitCode at $(Get-Date -Format o)" |
        Add-Content -LiteralPath $launcherLog
    exit $trainingExitCode
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $launcherLog
    exit 1
}
