param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$AcquiredAt,

    [string]$SceneId,
    [string]$RegionId,
    [string]$Output,
    [string]$GroundStore,
    [Nullable[double]]$ReflectanceScale,
    [double]$MaxCropCloudPercentage = 60.0,
    [int]$ConditionTileSize = 512,
    [int]$DownlinkMaxImageDimension = 1600,
    [int]$DownlinkGridSize = 16,
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Serve,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
if (-not $Output) {
    $inputStem = [System.IO.Path]::GetFileNameWithoutExtension($resolvedInput)
    $Output = Join-Path $workspace "testing\runs\${inputStem}_mvp"
}
if (-not $GroundStore) {
    $GroundStore = Join-Path $workspace "testing\runs\ground_mvp"
}
$integrationSource = Join-Path $workspace "integration\src"
$payloadSource = Join-Path $workspace "payload\src"
$groundSource = Join-Path $workspace "ground\src"
$sharedSource = Join-Path $workspace "shared\src"
$env:PYTHONPATH = "$integrationSource;$payloadSource;$groundSource;$sharedSource"
$env:NO_ALBUMENTATIONS_UPDATE = "1"
$env:MPLCONFIGDIR = Join-Path $workspace "outputs\cache\matplotlib"

$arguments = @(
    "-m", "vita_integration.mvp",
    $resolvedInput,
    "--output", $Output,
    "--ground-store", $GroundStore,
    "--acquired-at", $AcquiredAt,
    "--max-crop-cloud-percentage", $MaxCropCloudPercentage.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture
    ),
    "--condition-tile-size", $ConditionTileSize.ToString(),
    "--downlink-max-image-dimension", $DownlinkMaxImageDimension.ToString(),
    "--downlink-grid-size", $DownlinkGridSize.ToString(),
    "--host", $HostName,
    "--port", $Port.ToString()
)
if ($SceneId) {
    $arguments += @("--scene-id", $SceneId)
}
if ($RegionId) {
    $arguments += @("--region-id", $RegionId)
}
if ($PSBoundParameters.ContainsKey("ReflectanceScale")) {
    $arguments += @(
        "--reflectance-scale",
        ([double]$ReflectanceScale).ToString(
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    )
}
if ($Serve) {
    $arguments += "--serve"
}
if ($Overwrite) {
    $arguments += "--overwrite"
}

Set-Location -LiteralPath $workspace
& "$workspace\.venv\Scripts\python.exe" @arguments
exit $LASTEXITCODE
