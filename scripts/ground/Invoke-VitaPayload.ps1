[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$')]
    [string]$SshTarget,

    [Parameter(Mandatory = $true)]
    [ValidateSet('sentinel-2', 'sentinel-2-live', 'balkan-1', 'balkan-1-raw')]
    [string]$Sensor,

    [Parameter(Mandatory = $true)]
    [Alias('Input')]
    [string]$PayloadInput,

    [Parameter(Mandatory = $true)]
    [string]$RegionId,

    [string]$Image,
    [string]$CropCalibration,
    [string]$AcquiredAt,
    [Nullable[double]]$ReflectanceScale,
    [string]$BboxWgs84,
    [string]$StartDate,
    [string]$EndDate,
    [string]$JobId,
    [int]$SshPort = 22,
    [string]$IdentityFile,
    [string]$RemoteProjectRoot = '/data/code/VITA',
    [string]$RemoteRuntimeRoot = 'runtime/payload',
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
if ($Sensor -ne 'sentinel-2-live') {
    Assert-SafeRelativePath 'Input' $PayloadInput $false
}
if ($Image) { Assert-SafeRelativePath 'Image' $Image $true }
if ($CropCalibration) { Assert-SafeRelativePath 'CropCalibration' $CropCalibration $false }
if ($Sensor -eq 'sentinel-2' -and -not $Image -and $PayloadInput -notmatch '\.(tif|tiff)$') {
    throw 'Image is required when the Sentinel Input is a folder.'
}
if ($Sensor -ne 'sentinel-2' -and $Image) {
    throw 'Image is only valid for Sentinel folder inputs.'
}
$bboxNumbers = $null
if ($Sensor -eq 'sentinel-2-live') {
    $bboxParts = $BboxWgs84 -split ','
    if ($bboxParts.Count -ne 4) {
        throw 'BboxWgs84 must contain west,south,east,north.'
    }
    $bboxNumbers = @($bboxParts | ForEach-Object {
        [double]::Parse(
            $_,
            [Globalization.NumberStyles]::Float,
            [Globalization.CultureInfo]::InvariantCulture
        )
    })
    if ($StartDate -notmatch '^\d{4}-\d{2}-\d{2}$' -or $EndDate -notmatch '^\d{4}-\d{2}-\d{2}$') {
        throw 'StartDate and EndDate must use YYYY-MM-DD.'
    }
} elseif ($BboxWgs84 -or $StartDate -or $EndDate) {
    throw 'Area and date parameters are only valid for live Sentinel analysis.'
}
if (-not $JobId) {
    $prefix = switch ($Sensor) {
        'sentinel-2' { 'sentinel' }
        'sentinel-2-live' { 'live-sentinel' }
        'balkan-1-raw' { 'raw' }
        default { 'balkan' }
    }
    $JobId = '{0}-{1}-{2}' -f $prefix, (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'), ([guid]::NewGuid().ToString('N').Substring(0, 8))
}
Assert-SafeId 'JobId' $JobId
if ($SshPort -lt 1 -or $SshPort -gt 65535 -or $LocalTunnelPort -lt 1 -or $LocalTunnelPort -gt 65535) {
    throw 'SSH and tunnel ports must be in the range 1..65535.'
}
if ($RemoteProjectRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteProjectRoot.Contains('..')) {
    throw 'RemoteProjectRoot must be a safe absolute POSIX path.'
}
if ($RemoteRuntimeRoot -notmatch '^[A-Za-z0-9][A-Za-z0-9._/-]*$' -or $RemoteRuntimeRoot.Contains('..')) {
    throw 'RemoteRuntimeRoot must be a safe relative POSIX path.'
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
        input = ($PayloadInput -replace '\\', '/')
        region_id = $RegionId
        job_id = $JobId
    }
    if ($Image) { $request.image = $Image }
    if ($CropCalibration) { $request.crop_calibration = ($CropCalibration -replace '\\', '/') }
    if ($AcquiredAt) { $request.acquired_at = $AcquiredAt }
    if ($null -ne $ReflectanceScale) { $request.reflectance_scale = $ReflectanceScale.Value }
    if ($Sensor -eq 'sentinel-2-live') {
        $request.bbox_wgs84 = $bboxNumbers
        $request.start_date = $StartDate
        $request.end_date = $EndDate
    }

    Write-Host "Uplink: sending job metadata through the SSH tunnel (no image upload)."
    try {
        $response = Invoke-RestMethod `
            -Uri "$serviceUri/v1/jobs" `
            -Method Post `
            -ContentType 'application/json' `
            -Body ($request | ConvertTo-Json -Compress) `
            -TimeoutSec 1800
    } catch {
        $safeDetail = $null
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            try {
                $errorBody = $_.ErrorDetails.Message | ConvertFrom-Json
                if ($errorBody.detail -is [string]) {
                    $safeDetail = $errorBody.detail
                } elseif ($errorBody.detail.message) {
                    $safeDetail = $errorBody.detail.message
                    if ($errorBody.detail.code) {
                        $safeDetail = "[$($errorBody.detail.code)] $safeDetail"
                    }
                }
            } catch {
                $safeDetail = $null
            }
        }
        if (-not $safeDetail) { $safeDetail = $_.Exception.Message }
        throw "Payload request failed: $safeDetail"
    }
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
$remoteBundle = "$RemoteProjectRoot/$RemoteRuntimeRoot/$($response.bundle_relative)"
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
    & (Join-Path $PSScriptRoot 'Start-VitaDashboard.ps1')
}

Write-Host ('PAYLOAD TOTAL {0:N4} s' -f [double]$response.payload_seconds)

[ordered]@{
    status = 'MVP_READY'
    job_id = $JobId
    payload_seconds = $response.payload_seconds
    under_two_seconds = $response.under_two_seconds
    under_five_seconds = $response.under_five_seconds
    pipeline_timing_seconds = $response.pipeline_timing_seconds
    acceleration = $response.stack
    downlink = $downlinkRoot
    ingest = $ingest
    dashboard = if ($SkipDashboard) { $null } else { 'http://127.0.0.1:8000/' }
} | ConvertTo-Json -Depth 8
