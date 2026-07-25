$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pipelineDirectory = Join-Path $workspace "outputs\prithvi_4band_europe_replay"
$archive = Join-Path $workspace "data\downloads\PASTIS.zip"
$pastisDirectory = Join-Path $workspace "data\europe\pastis"
$startedMarker = Join-Path $pipelineDirectory "pipeline.started"
$completedMarker = Join-Path $pipelineDirectory "pipeline.completed"
$statusFile = Join-Path $pipelineDirectory "pipeline.status.txt"
$pipelineOutput = Join-Path $pipelineDirectory "pipeline.stdout.log"
$pipelineError = Join-Path $pipelineDirectory "pipeline.stderr.log"
$trainingOutput = Join-Path $pipelineDirectory "training.stdout.log"
$trainingError = Join-Path $pipelineDirectory "training.stderr.log"
$downloadUrl = "https://zenodo.org/records/5012942/files/PASTIS.zip?download=1"
$directMetadata = Join-Path $pastisDirectory "metadata.geojson"
$nestedMetadata = Join-Path $pastisDirectory "PASTIS\metadata.geojson"

try {
    Set-Location -LiteralPath $workspace
    New-Item -ItemType Directory -Path $pipelineDirectory -Force | Out-Null
    New-Item -ItemType Directory -Path (Split-Path $archive) -Force | Out-Null
    New-Item -ItemType Directory -Path $pastisDirectory -Force | Out-Null
    if (Test-Path -LiteralPath $completedMarker) {
        exit 0
    }
    if (Test-Path -LiteralPath $startedMarker) {
        $previousStatus = if (Test-Path -LiteralPath $statusFile) {
            (Get-Content -LiteralPath $statusFile -Raw).Trim()
        } else {
            "unknown"
        }
        if ($previousStatus -ne "failed") {
            exit 0
        }
        Remove-Item -LiteralPath $startedMarker
    }
    New-Item -ItemType File -Path $startedMarker -Force | Out-Null

    $env:HF_HOME = Join-Path $workspace "outputs\cache\huggingface"
    $env:HF_HUB_OFFLINE = "1"
    $env:NO_ALBUMENTATIONS_UPDATE = "1"
    $env:MPLCONFIGDIR = Join-Path $workspace "outputs\cache\matplotlib"
    $env:PYTHONUNBUFFERED = "1"
    $ErrorActionPreference = "Continue"

    if (
        -not (Test-Path -LiteralPath $directMetadata) -and
        -not (Test-Path -LiteralPath $nestedMetadata)
    ) {
        "downloading" | Set-Content -LiteralPath $statusFile
        & curl.exe `
            --location `
            --fail `
            --retry 8 `
            --retry-delay 15 `
            --continue-at - `
            --output $archive `
            $downloadUrl `
            1> $pipelineOutput `
            2> $pipelineError
        if ($LASTEXITCODE -ne 0) {
            throw "PASTIS download failed with code $LASTEXITCODE"
        }

        "extracting" | Set-Content -LiteralPath $statusFile
        & "$workspace\.venv\Scripts\python.exe" `
            scripts\prepare_pastis.py `
            --archive $archive `
            --destination $pastisDirectory `
            1>> $pipelineOutput `
            2>> $pipelineError
        if ($LASTEXITCODE -ne 0) {
            throw "PASTIS preparation failed with code $LASTEXITCODE"
        }
    }

    "validating_data" | Set-Content -LiteralPath $statusFile
    & "$workspace\.venv\Scripts\python.exe" `
        -m prithvi_crop.pastis_validation `
        --root $pastisDirectory `
        1>> $pipelineOutput `
        2>> $pipelineError
    if ($LASTEXITCODE -ne 0) {
        throw "PASTIS validation failed with code $LASTEXITCODE"
    }

    "validating_training" | Set-Content -LiteralPath $statusFile
    & "$workspace\.venv\Scripts\python.exe" `
        -m prithvi_crop.preflight `
        --config configs\prithvi_4band_europe_replay.yaml `
        1>> $pipelineOutput `
        2>> $pipelineError
    if ($LASTEXITCODE -ne 0) {
        throw "European replay preflight failed with code $LASTEXITCODE"
    }
    & "$workspace\.venv\Scripts\python.exe" `
        -m prithvi_crop.smoke `
        --config configs\prithvi_4band_europe_replay.yaml `
        --batch-size 4 `
        1>> $pipelineOutput `
        2>> $pipelineError
    if ($LASTEXITCODE -ne 0) {
        throw "European replay smoke test failed with code $LASTEXITCODE"
    }

    "training" | Set-Content -LiteralPath $statusFile
    & "$workspace\.venv\Scripts\python.exe" `
        -m prithvi_crop.train `
        --config configs\prithvi_4band_europe_replay.yaml `
        --resume auto `
        1> $trainingOutput `
        2> $trainingError
    if ($LASTEXITCODE -ne 0) {
        throw "European replay training failed with code $LASTEXITCODE"
    }

    "complete" | Set-Content -LiteralPath $statusFile
    New-Item -ItemType File -Path $completedMarker -Force | Out-Null
    exit 0
}
catch {
    "failed" | Set-Content -LiteralPath $statusFile
    if (Test-Path -LiteralPath $startedMarker) {
        Remove-Item -LiteralPath $startedMarker
    }
    $_ | Out-String | Add-Content -LiteralPath $pipelineError
    exit 1
}
