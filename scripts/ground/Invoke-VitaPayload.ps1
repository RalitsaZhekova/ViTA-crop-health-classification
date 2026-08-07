[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$')]
    [string]$SshTarget,

    [Parameter(Mandatory = $true)]
    [ValidateSet('sentinel-2', 'balkan-1')]
    [string]$Sensor,

    [Parameter(Mandatory = $true)]
    [string]$Input,

    [Parameter(Mandatory = $true)]
    [string]$RegionId,

    [string]$Image,
    [string]$CropCalibration,
    [string]$AcquiredAt,
    [Nullable[double]]$ReflectanceScale,
    [string]$JobId,
    [int]$SshPort = 22,
    [string]$IdentityFile,
    [string]$RemoteProjectRoot = '/data/code/VITA',
    [int]$RemotePayloadPort = 8090,
    [int]$LocalTunnelPort = 18090,
    [string]$GroundStore = 'runtime\ground',
    [switch]$SkipDashboard
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-SafeId([string]$Name, [string]$Value) {
    if ($Value -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$') {
        throw "$Name must contain only letters, digits, dot, underscore or dash (maximum 80 characters)."
    }
}

function Assert-SafeRelativePath([string]$Name, [string]$Value, [bool]$FileNameOnly) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value.Length -gt 512) {
        throw "$Name must be a non-empty relative payload path."
    }
    if ($Value -notmatch '^[A-Za-z0-9][A-Za-z0-9._/-]*$') {
        throw "$Name contains unsupported path characters."
    }
    $parts = $Value -split '[/\\]'
    if ($parts -contains '..' -or $parts -contains '.') {
        throw "$Name cannot contain dot path segments."
    }
    if ($FileNameOnly -and $parts.Count -ne 1) {
        throw "$Name must be a filename, not a path."
    }
}

function ConvertTo-ProcessArgument([string]$Value) {
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

foreach ($command in @('ssh', 'scp')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "$command is required. Install the Windows OpenSSH Client feature."
    }
}

Assert-SafeId 'RegionId' $RegionId
Assert-SafeRelativePath 'Input' $Input $false
if ($Image) { Assert-SafeRelativePath 'Image' $Image $true }
if ($CropCalibration) { Assert-SafeRelativePath 'CropCalibration' $CropCalibration $false }
if ($Sensor -eq 'sentinel-2' -and -not $Image -and $Input -notmatch '\.(tif|tiff)$') {
    throw 'Image is required when the Sentinel Input is a folder.'
}
if ($Sensor -eq 'balkan-1' -and $Image) {
    throw 'Image is only valid for Sentinel folder inputs.'
}
if (-not $JobId) {
    $prefix = if ($Sensor -eq 'sentinel-2') { 'sentinel' } else { 'balkan' }
    $JobId = '{0}-{1}-{2}' -f $prefix, (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'), ([guid]::NewGuid().ToString('N').Substring(0, 8))
}
Assert-SafeId 'JobId' $JobId
if ($SshPort -lt 1 -or $SshPort -gt 65535 -or $LocalTunnelPort -lt 1 -or $LocalTunnelPort -gt 65535) {
    throw 'SSH and tunnel ports must be in the range 1..65535.'
}
if ($RemoteProjectRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteProjectRoot.Contains('..')) {
    throw 'RemoteProjectRoot must be a safe absolute POSIX path.'
}
if ($IdentityFile) {
    $IdentityFile = (Resolve-Path -LiteralPath $IdentityFile).Path
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$groundStorePath = if ([IO.Path]::IsPathRooted($GroundStore)) {
    $GroundStore
} else {
    Join-Path $repositoryRoot $GroundStore
}
$downlinkRoot = Join-Path $repositoryRoot (Join-Path 'runtime\downlink' $JobId)
if (Test-Path -LiteralPath $downlinkRoot) {
    throw "Local downlink directory already exists: $downlinkRoot"
}

$commonSsh = @(
    '-p', $SshPort,
    '-o', 'BatchMode=yes',
    '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=3'
)
if ($IdentityFile) { $commonSsh += @('-i', $IdentityFile) }
$tunnelArgs = $commonSsh + @(
    '-N',
    '-L', "127.0.0.1:${LocalTunnelPort}:127.0.0.1:${RemotePayloadPort}",
    $SshTarget
)
$argumentLine = ($tunnelArgs | ForEach-Object { ConvertTo-ProcessArgument ([string]$_) }) -join ' '

$tunnel = $null
try {
    $tunnel = Start-Process -FilePath 'ssh' -ArgumentList $argumentLine -PassThru -WindowStyle Hidden
    $serviceUri = "http://127.0.0.1:${LocalTunnelPort}"
    $health = $null
    foreach ($attempt in 1..30) {
        if ($tunnel.HasExited) {
            throw "SSH tunnel exited with code $($tunnel.ExitCode). Check SSH access and known_hosts."
        }
        try {
            $health = Invoke-RestMethod -Uri "$serviceUri/healthz" -Method Get -TimeoutSec 3
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $health -or $health.status -ne 'ready') {
        throw 'Payload service is not ready. Run deploy/payload/deploy.sh on the Orin and inspect docker compose logs.'
    }

    $request = [ordered]@{
        sensor = $Sensor
        input = ($Input -replace '\\', '/')
        region_id = $RegionId
        job_id = $JobId
    }
    if ($Image) { $request.image = $Image }
    if ($CropCalibration) { $request.crop_calibration = ($CropCalibration -replace '\\', '/') }
    if ($AcquiredAt) { $request.acquired_at = $AcquiredAt }
    if ($null -ne $ReflectanceScale) { $request.reflectance_scale = $ReflectanceScale.Value }

    Write-Host "Uplink: sending job metadata through the SSH tunnel (no image upload)."
    $response = Invoke-RestMethod `
        -Uri "$serviceUri/v1/jobs" `
        -Method Post `
        -ContentType 'application/json' `
        -Body ($request | ConvertTo-Json -Compress) `
        -TimeoutSec 1800
} finally {
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id
        $tunnel.WaitForExit()
    }
}

if ($response.status -ne 'DOWNLINK_READY' -or $response.job_id -ne $JobId) {
    throw 'Payload returned an invalid completion response.'
}

New-Item -ItemType Directory -Path $downlinkRoot | Out-Null
$scpArgs = @('-P', $SshPort, '-o', 'BatchMode=yes')
if ($IdentityFile) { $scpArgs += @('-i', $IdentityFile) }
$remoteBundle = "$RemoteProjectRoot/runtime/payload/$($response.bundle_relative)"
foreach ($fileName in @('scene.json', 'scene.webp', 'condition.png')) {
    $remoteFile = "${SshTarget}:${remoteBundle}/${fileName}"
    & scp @scpArgs $remoteFile $downlinkRoot
    if ($LASTEXITCODE -ne 0) { throw "SCP failed for $fileName." }
    $expected = $response.files.PSObject.Properties[$fileName].Value.sha256
    $actual = (Get-FileHash -LiteralPath (Join-Path $downlinkRoot $fileName) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "SHA-256 verification failed for $fileName." }
}
Write-Host 'Downlink: all three artifacts passed SHA-256 verification.'

$localIngest = Join-Path $repositoryRoot '.venv\Scripts\vita-ingest.exe'
if (Test-Path -LiteralPath $localIngest) {
    $ingestOutput = & $localIngest $downlinkRoot --store $groundStorePath
} elseif (Get-Command vita-ingest -ErrorAction SilentlyContinue) {
    $ingestOutput = & vita-ingest $downlinkRoot --store $groundStorePath
} else {
    throw 'vita-ingest is unavailable. Run .venv\Scripts\python -m pip install -e . on the ground.'
}
if ($LASTEXITCODE -ne 0) { throw "Ground ingest failed: $ingestOutput" }
$ingest = $ingestOutput | ConvertFrom-Json

if (-not $SkipDashboard) {
    Push-Location $repositoryRoot
    try {
        & docker compose -f deploy\compose.ground.yaml up -d
        if ($LASTEXITCODE -ne 0) { throw 'Ground dashboard container failed to start.' }
    } finally {
        Pop-Location
    }
}

[ordered]@{
    status = 'MVP_READY'
    job_id = $JobId
    payload_seconds = $response.payload_seconds
    under_two_seconds = $response.under_two_seconds
    under_five_seconds = $response.under_five_seconds
    acceleration = $response.stack
    downlink = $downlinkRoot
    ingest = $ingest
    dashboard = if ($SkipDashboard) { $null } else { 'http://127.0.0.1:8000/' }
} | ConvertTo-Json -Depth 8
