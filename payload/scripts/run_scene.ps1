param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [ValidateSet("sentinel-2", "balkan-1")]
    [string]$Sensor,

    [string]$AcquiredAt,
    [string]$SceneId,
    [string]$Output = "outputs\pipeline",
    [ValidateSet("intake", "cloud")]
    [string]$StopAfter = "cloud",
    [Nullable[double]]$ReflectanceScale
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
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
    "--stop-after", $StopAfter
)
if ($AcquiredAt) {
    $pipelineArguments += @("--acquired-at", $AcquiredAt)
}
if ($SceneId) {
    $pipelineArguments += @("--scene-id", $SceneId)
}
if ($null -ne $ReflectanceScale) {
    $pipelineArguments += @("--reflectance-scale", $ReflectanceScale.ToString())
}

Set-Location -LiteralPath $workspace
& "$workspace\.venv\Scripts\python.exe" @pipelineArguments
exit $LASTEXITCODE
