param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$AcquiredAt,

    [string]$SceneId,
    [string]$RegionId,
    [string]$Output,
    [Nullable[double]]$ReflectanceScale,
    [double]$MaxCropCloudPercentage = 60.0,
    [int]$GroundTileSize = 512,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
if (-not $Output) {
    $inputStem = [System.IO.Path]::GetFileNameWithoutExtension($resolvedInput)
    $Output = Join-Path $workspace "testing\runs\${inputStem}_end_to_end"
}
$integrationSource = Join-Path $workspace "integration\src"
$payloadSource = Join-Path $workspace "payload\src"
$groundSource = Join-Path $workspace "ground\src"
$sharedSource = Join-Path $workspace "shared\src"
$env:PYTHONPATH = "$integrationSource;$payloadSource;$groundSource;$sharedSource"
$env:NO_ALBUMENTATIONS_UPDATE = "1"
$env:MPLCONFIGDIR = Join-Path $workspace "outputs\cache\matplotlib"

$arguments = @(
    "-m", "vita_integration.pipeline",
    $resolvedInput,
    "--output", $Output,
    "--acquired-at", $AcquiredAt,
    "--max-crop-cloud-percentage", $MaxCropCloudPercentage.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture
    ),
    "--ground-tile-size", $GroundTileSize.ToString()
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
if ($Overwrite) {
    $arguments += "--overwrite"
}

Set-Location -LiteralPath $workspace
& "$workspace\.venv\Scripts\python.exe" @arguments
exit $LASTEXITCODE
