$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDirectory = Join-Path $workspace "outputs\prithvi_4band_single_frame_binary"
$statusFile = Join-Path $runDirectory "training.status.txt"
$standardOutput = Join-Path $runDirectory "training.stdout.log"
$standardError = Join-Path $runDirectory "training.stderr.log"
$config = Join-Path $workspace "configs\prithvi_4band_single_frame_binary.yaml"

Set-Location -LiteralPath $workspace
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$env:HF_HOME = Join-Path $workspace "outputs\cache\huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:NO_ALBUMENTATIONS_UPDATE = "1"
$env:MPLCONFIGDIR = Join-Path $workspace "outputs\cache\matplotlib"
$env:PYTHONUNBUFFERED = "1"
$ErrorActionPreference = "Continue"

try {
    "training" | Set-Content -LiteralPath $statusFile
    & "$workspace\.venv\Scripts\python.exe" `
        -m prithvi_crop.train `
        --config $config `
        --resume auto `
        1> $standardOutput `
        2> $standardError
    if ($LASTEXITCODE -ne 0) {
        throw "Single-frame binary training failed with code $LASTEXITCODE"
    }

    "complete" | Set-Content -LiteralPath $statusFile
    exit 0
}
catch {
    "failed" | Set-Content -LiteralPath $statusFile
    $_ | Out-String | Add-Content -LiteralPath $standardError
    exit 1
}
