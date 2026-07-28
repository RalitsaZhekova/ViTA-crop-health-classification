param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [ValidateSet("sentinel-2", "balkan-1")]
    [string]$Sensor,

    [string]$AcquiredAt,
    [string]$SceneId,
    [string]$Output,
    [ValidateSet("intake", "cloud", "crop", "condition", "downlink")]
    [string]$StopAfter = "cloud",
    [string]$RegionId,
    [int]$ConditionTileSize = 512,
    [int]$DownlinkMaxImageDimension = 1600,
    [int]$DownlinkGridSize = 16,
    [switch]$Overwrite,
    [Nullable[double]]$ReflectanceScale,
    [double]$MaxCropCloudPercentage = 60.0
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
if (-not $Output) {
    $inputStem = [System.IO.Path]::GetFileNameWithoutExtension($resolvedInput)
    $Output = Join-Path $workspace "testing\runs\$inputStem"
}
$payloadSource = Join-Path $workspace "payload\src"
$sharedSource = Join-Path $workspace "shared\src"
$env:PYTHONPATH = "$payloadSource;$sharedSource"
$env:NO_ALBUMENTATIONS_UPDATE = "1"
$env:MPLCONFIGDIR = Join-Path $workspace "outputs\cache\matplotlib"

$pipelineArguments = @(
    "-m", "prithvi_payload.pipeline",
    $resolvedInput,
    "--sensor", $Sensor,
    "--output", $Output,
    "--stop-after", $StopAfter,
    "--max-crop-cloud-percentage", $MaxCropCloudPercentage.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture
    ),
    "--condition-tile-size", $ConditionTileSize.ToString(),
    "--downlink-max-image-dimension", $DownlinkMaxImageDimension.ToString(),
    "--downlink-grid-size", $DownlinkGridSize.ToString()
)
if ($AcquiredAt) {
    $pipelineArguments += @("--acquired-at", $AcquiredAt)
}
if ($SceneId) {
    $pipelineArguments += @("--scene-id", $SceneId)
}
if ($RegionId) {
    $pipelineArguments += @("--region-id", $RegionId)
}
if ($null -ne $ReflectanceScale) {
    $pipelineArguments += @("--reflectance-scale", $ReflectanceScale.ToString())
}
if ($Overwrite) {
    $pipelineArguments += "--overwrite"
}

Set-Location -LiteralPath $workspace
& "$workspace\.venv\Scripts\python.exe" @pipelineArguments
exit $LASTEXITCODE
