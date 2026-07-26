$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runDirectory = Join-Path $workspace "outputs\prithvi_4band_single_frame"
$statusFile = Join-Path $runDirectory "training.status.txt"
$standardOutput = Join-Path $runDirectory "training.stdout.log"
$standardError = Join-Path $runDirectory "training.stderr.log"
$config = Join-Path $workspace "configs\prithvi_4band_single_frame.yaml"
$validationReport = Join-Path $workspace "outputs\dataset_validation.json"

Set-Location -LiteralPath $workspace
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$env:HF_HOME = Join-Path $workspace "outputs\cache\huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:NO_ALBUMENTATIONS_UPDATE = "1"
$env:MPLCONFIGDIR = Join-Path $workspace "outputs\cache\matplotlib"
$env:PYTHONUNBUFFERED = "1"
# Windows PowerShell surfaces native stderr as NativeCommandError. PyTorch and
# Lightning write non-fatal diagnostics there, so native success is determined
# explicitly from LASTEXITCODE below.
$ErrorActionPreference = "Continue"

try {
    if (-not (Test-Path -LiteralPath $validationReport)) {
        "validating_data" | Set-Content -LiteralPath $statusFile
        & "$workspace\.venv\Scripts\python.exe" `
            -m prithvi_crop.validate_data `
            --root data\multi_temporal_crop `
            --device cuda `
            1> $standardOutput `
            2> $standardError
        if ($LASTEXITCODE -ne 0) {
            throw "Dataset validation failed with code $LASTEXITCODE"
        }
    }

    "preflight" | Set-Content -LiteralPath $statusFile
    & "$workspace\.venv\Scripts\python.exe" `
        -m prithvi_crop.preflight `
        --config $config `
        1>> $standardOutput `
        2>> $standardError
    if ($LASTEXITCODE -ne 0) {
        throw "Single-frame preflight failed with code $LASTEXITCODE"
    }

    "gpu_smoke" | Set-Content -LiteralPath $statusFile
    & "$workspace\.venv\Scripts\python.exe" `
        -m prithvi_crop.smoke `
        --config $config `
        --batch-size 32 `
        1>> $standardOutput `
        2>> $standardError
    if ($LASTEXITCODE -ne 0) {
        throw "Single-frame CUDA smoke test failed with code $LASTEXITCODE"
    }

    "training" | Set-Content -LiteralPath $statusFile
    & "$workspace\.venv\Scripts\python.exe" `
        -m prithvi_crop.train `
        --config $config `
        --resume auto `
        1>> $standardOutput `
        2>> $standardError
    if ($LASTEXITCODE -ne 0) {
        throw "Single-frame training failed with code $LASTEXITCODE"
    }

    "complete" | Set-Content -LiteralPath $statusFile
    exit 0
}
catch {
    "failed" | Set-Content -LiteralPath $statusFile
    $_ | Out-String | Add-Content -LiteralPath $standardError
    exit 1
}
