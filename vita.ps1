[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('sentinel', 'balkan', 'raw', 'web', 'health', 'stop', 'help')]
    [string]$Command = 'help',

    [string]$InputPath,
    [string]$Image,
    [string]$RegionId,
    [string]$JobId,
    [int]$PayloadPort = 8090,
    [int]$RawPayloadPort = 8091,
    [int]$RawTunnelPort = 18091,
    [int]$WebPort = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repositoryRoot = $PSScriptRoot
$runtimeRoot = Join-Path $repositoryRoot 'runtime'
$localRoot = Join-Path $runtimeRoot 'payload-local'
$groundStore = Join-Path $runtimeRoot 'ground'
$payloadUri = "http://127.0.0.1:$PayloadPort"

function Get-LocalTool([string]$Name) {
    $local = Join-Path $repositoryRoot ".venv\Scripts\$Name.exe"
    if (Test-Path -LiteralPath $local) { return $local }
    $installed = Get-Command $Name -ErrorAction SilentlyContinue
    if ($installed) { return $installed.Source }
    throw "$Name is unavailable. Run .\.venv\Scripts\python.exe -m pip install -e ."
}

function ConvertTo-NativeArgument([string]$Value) {
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Start-HiddenProcess([string]$FilePath, [string[]]$ArgumentList) {
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $FilePath
    $start.Arguments = ($ArgumentList | ForEach-Object {
        ConvertTo-NativeArgument ([string]$_)
    }) -join ' '
    $start.WorkingDirectory = $repositoryRoot
    $start.UseShellExecute = $true
    $start.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    return [System.Diagnostics.Process]::Start($start)
}

function Get-Health {
    try {
        return Invoke-RestMethod -Uri "$payloadUri/healthz" -Method Get -TimeoutSec 3
    } catch {
        return $null
    }
}

function Get-EnvironmentDefault([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value
}

function Get-JetsonSshTarget {
    return Get-EnvironmentDefault 'VITA_JETSON_SSH_TARGET' ''
}

function Get-ListeningProcessId([int]$Port) {
    $connection = Get-NetTCPConnection `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($connection) { return [int]$connection.OwningProcess }

    # Some Windows configurations hide user-owned sockets from
    # Get-NetTCPConnection but still expose them through netstat.
    foreach ($line in (& netstat.exe -ano -p TCP 2>$null)) {
        if ($line -match "^\s*TCP\s+\S+:$Port\s+\S+\s+LISTENING\s+(\d+)\s*$") {
            return [int]$Matches[1]
        }
    }
    return $null
}

function Start-LocalPayload {
    $health = Get-Health
    if ($health -and $health.status -eq 'ready') { return $health }

    $listenerPid = Get-ListeningProcessId $PayloadPort
    if ($listenerPid) {
        $recordPath = Join-Path $localRoot 'payload-process.json'
        $recordedListenerPid = $null
        if (Test-Path -LiteralPath $recordPath) {
            try {
                $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
                $recordedListenerPid = [int]$record.listener_pid
            } catch {
                $recordedListenerPid = $null
            }
        }
        if ($recordedListenerPid -and $listenerPid -eq $recordedListenerPid) {
            Write-Host 'The recorded payload service failed its active CUDA readiness probe; restarting it.'
            Stop-RecordedProcess $recordPath 'payload service'
            foreach ($attempt in 1..50) {
                if (-not (Get-ListeningProcessId $PayloadPort)) { break }
                Start-Sleep -Milliseconds 100
            }
            if (Get-ListeningProcessId $PayloadPort) {
                throw "The unhealthy ViTA payload did not release port $PayloadPort."
            }
        } else {
            throw "Port $PayloadPort is occupied by a service that is not a healthy ViTA payload."
        }
    }

    $payloadServer = Get-LocalTool 'vita-payload-server'
    $runs = Join-Path $localRoot 'runs'
    $analysisCache = Join-Path $localRoot 'cache\balkan-analysis'
    $engineCache = Join-Path $runtimeRoot 'engines'
    foreach ($directory in @($runs, $analysisCache, $engineCache, $groundStore)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }

    $childEnvironment = [ordered]@{
        CUDA_REQUIRED = Get-EnvironmentDefault 'CUDA_REQUIRED' '1'
        VITA_INPUT_ROOT = Join-Path $repositoryRoot 'data'
        VITA_OUTPUT_ROOT = $runs
        VITA_BALKAN_ANALYSIS_CACHE_DIR = $analysisCache
        VITA_BALKAN_OVERVIEW_FAST_PATH = '1'
        VITA_BALKAN_OVERVIEW_REQUIRED = '1'
        VITA_BALKAN_PREP_MAX_BYTES = '2147483648'
        VITA_MODEL_CACHE_DIR = Join-Path $engineCache 'torch-export'
        VITA_TRT_CACHE_DIR = Join-Path $engineCache 'tensorrt'
        VITA_CROP_BACKEND = Get-EnvironmentDefault 'VITA_CROP_BACKEND' 'pytorch'
        VITA_CROP_BATCH_SIZE = Get-EnvironmentDefault 'VITA_CROP_BATCH_SIZE' '4'
        VITA_CLOUD_BACKEND = Get-EnvironmentDefault 'VITA_CLOUD_BACKEND' 'pytorch'
        VITA_CLOUD_INFERENCE_DTYPE = Get-EnvironmentDefault 'VITA_CLOUD_INFERENCE_DTYPE' 'fp32'
        VITA_CLOUD_BATCH_SIZE = Get-EnvironmentDefault 'VITA_CLOUD_BATCH_SIZE' '2'
        VITA_CLOUD_WARMUP_PATCH_SIZES = Get-EnvironmentDefault 'VITA_CLOUD_WARMUP_PATCH_SIZES' '869'
        VITA_WARMUP = '1'
        VITA_CPU_THREADS = Get-EnvironmentDefault 'VITA_CPU_THREADS' '8'
        VITA_CONDITION_TILE_SIZE = Get-EnvironmentDefault 'VITA_CONDITION_TILE_SIZE' '4096'
        VITA_CONDITION_METRIC_THREADS = Get-EnvironmentDefault 'VITA_CONDITION_METRIC_THREADS' '4'
        VITA_CONDITION_EXACT_PERCENTILES = Get-EnvironmentDefault 'VITA_CONDITION_EXACT_PERCENTILES' '1'
        VITA_CROP_IN_MEMORY = Get-EnvironmentDefault 'VITA_CROP_IN_MEMORY' '1'
        VITA_CROP_IN_MEMORY_MAX_BYTES = Get-EnvironmentDefault 'VITA_CROP_IN_MEMORY_MAX_BYTES' '1073741824'
        VITA_DOWNLINK_GRID_IN_MEMORY = Get-EnvironmentDefault 'VITA_DOWNLINK_GRID_IN_MEMORY' '1'
        VITA_DOWNLINK_GRID_IN_MEMORY_MAX_BYTES = Get-EnvironmentDefault 'VITA_DOWNLINK_GRID_IN_MEMORY_MAX_BYTES' '536870912'
        VITA_FAST_INTERMEDIATE_RASTERS = Get-EnvironmentDefault 'VITA_FAST_INTERMEDIATE_RASTERS' '1'
        VITA_COMPACT_PAYLOAD_PIPELINE = Get-EnvironmentDefault 'VITA_COMPACT_PAYLOAD_PIPELINE' '1'
        PRITHVI_MODEL_DIR = Join-Path $repositoryRoot 'payload\models'
        OMNICLOUDMASK_MODEL_DIR = Join-Path $repositoryRoot 'payload\models\omnicloudmask'
    }
    $balkanInput = Join-Path $repositoryRoot 'data\balkan1\preprocessed\3408_L1ORT.tif'
    if (Test-Path -LiteralPath $balkanInput) {
        $childEnvironment.VITA_BALKAN_PREPARE_INPUT = 'balkan1/preprocessed/3408_L1ORT.tif'
    }

    $previousEnvironment = @{}
    try {
        foreach ($entry in $childEnvironment.GetEnumerator()) {
            $previousEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable(
                $entry.Key,
                'Process'
            )
            [Environment]::SetEnvironmentVariable(
                $entry.Key,
                [string]$entry.Value,
                'Process'
            )
        }
        Write-Host 'Starting the warm local payload service; first startup loads and warms both models.'
        $server = Start-HiddenProcess `
            $payloadServer `
            @('--host', '127.0.0.1', '--port', [string]$PayloadPort)
    } finally {
        foreach ($entry in $previousEnvironment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable(
                $entry.Key,
                $entry.Value,
                'Process'
            )
        }
    }

    foreach ($attempt in 1..900) {
        $health = Get-Health
        if ($health -and $health.status -eq 'ready') {
            $listenerPid = Get-ListeningProcessId $PayloadPort
            [ordered]@{
                listener_pid = $listenerPid
                starter_pid = $server.Id
                starter_start_time = $server.StartTime.ToUniversalTime().ToString('o')
                port = $PayloadPort
            } | ConvertTo-Json | Set-Content `
                -LiteralPath (Join-Path $localRoot 'payload-process.json') `
                -Encoding utf8
            return $health
        }
        if ($server.HasExited) {
            throw "Payload service exited during startup with code $($server.ExitCode)."
        }
        Start-Sleep -Seconds 1
    }
    throw 'Payload service did not become ready within 15 minutes.'
}

function Write-TimingBreakdown($Response) {
    $timing = $Response.pipeline_timing_seconds
    $rows = @(
        @('intake_seconds', 'Scene intake'),
        @('shared_analysis_grid_seconds', 'Balkan shared grid'),
        @('cloud_plan_seconds', 'Cloud planning'),
        @('cloud_stage_seconds', 'Cloud stage'),
        @('cloud_inference_seconds', '  Cloud inference'),
        @('cloud_mask_processing_seconds', '  Cloud mask processing'),
        @('crop_plan_seconds', 'Crop planning'),
        @('crop_stage_seconds', 'Crop stage'),
        @('crop_inference_seconds', '  Crop inference'),
        @('condition_stage_seconds', 'Condition stage'),
        @('downlink_packaging_seconds', 'Downlink packaging'),
        @('orchestration_seconds', 'Orchestration')
    )
    Write-Host ''
    Write-Host 'Execution time breakdown (nested inference rows are already inside stage totals)'
    foreach ($row in $rows) {
        $property = $timing.PSObject.Properties[$row[0]]
        if ($null -ne $property) {
            Write-Host ('{0,-34} {1,9:N4} s' -f $row[1], [double]$property.Value)
        }
    }
    Write-Host ('{0,-34} {1,9:N4} s' -f 'PAYLOAD TOTAL', [double]$Response.payload_seconds)
    Write-Host ('Under two seconds: {0}' -f $Response.under_two_seconds)
    Write-Host ('Under five seconds: {0}' -f $Response.under_five_seconds)
}

function Invoke-LocalPipeline([string]$SensorName) {
    $null = Start-LocalPayload
    $prefix = if ($SensorName -eq 'sentinel-2') { 'local-sentinel' } else { 'local-balkan' }
    $resolvedJobId = if ($JobId) {
        $JobId
    } else {
        '{0}-{1}-{2}' -f `
            $prefix,
            (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'),
            ([guid]::NewGuid().ToString('N').Substring(0, 8))
    }
    $request = [ordered]@{
        sensor = $SensorName
        region_id = if ($RegionId) {
            $RegionId
        } elseif ($SensorName -eq 'sentinel-2') {
            'sentinel-local-cloudy'
        } else {
            'balkan-test-3408'
        }
        job_id = $resolvedJobId
    }
    if ($SensorName -eq 'sentinel-2') {
        $request.input = if ($InputPath) { $InputPath } else { 'sentinel2' }
        $request.image = if ($Image) {
            $Image
        } else {
            'S2_20260712T170851_T14TPL_cloudy.tif'
        }
    } else {
        if ($Image) { throw '-Image is only valid for Sentinel-2.' }
        $request.input = if ($InputPath) {
            $InputPath
        } else {
            'balkan1/preprocessed/3408_L1ORT.tif'
        }
    }

    Write-Host "Running $SensorName as $resolvedJobId ..."
    $response = Invoke-RestMethod `
        -Uri "$payloadUri/v1/jobs" `
        -Method Post `
        -ContentType 'application/json' `
        -Body ($request | ConvertTo-Json -Compress) `
        -TimeoutSec 1800
    if ($response.status -ne 'DOWNLINK_READY') {
        throw "Payload did not produce a downlink bundle: $($response.status)"
    }

    $bundle = Join-Path $localRoot ($response.bundle_relative -replace '/', '\')
    $ingest = Get-LocalTool 'vita-ingest'
    $ingestOutput = & $ingest $bundle --store $groundStore
    if ($LASTEXITCODE -ne 0) { throw "Ground ingest failed: $ingestOutput" }

    Write-TimingBreakdown $response
    Write-Host ''
    Write-Host "Dashboard data: $groundStore"
    Write-Host 'Visualize it with: .\vita.ps1 web'
}

function Invoke-StableJetsonPipeline([string]$SensorName) {
    $sshTarget = Get-JetsonSshTarget
    if ([string]::IsNullOrWhiteSpace($sshTarget)) {
        throw 'Set VITA_JETSON_SSH_TARGET=user@jetson before launching Jetson analysis.'
    }
    if ($SensorName -ne 'sentinel-2' -and $Image) {
        throw '-Image is only valid for Sentinel-2.'
    }
    $payloadInput = if ($InputPath) {
        $InputPath
    } elseif ($SensorName -eq 'sentinel-2') {
        'sentinel2'
    } else {
        'balkan1/preprocessed/3408_L1ORT.tif'
    }
    $resolvedImage = if ($SensorName -eq 'sentinel-2') {
        if ($Image) { $Image } else { 'S2_20260712T170851_T14TPL_cloudy.tif' }
    } else {
        $null
    }
    $prefix = if ($SensorName -eq 'sentinel-2') { 'sentinel' } else { 'balkan' }
    $resolvedRegionId = if ($RegionId) {
        $RegionId
    } elseif ($SensorName -eq 'sentinel-2') {
        'sentinel-local-cloudy'
    } else {
        'balkan-test-3408'
    }
    $resolvedJobId = if ($JobId) {
        $JobId
    } else {
        '{0}-{1}-{2}' -f `
            $prefix,
            (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'),
            ([guid]::NewGuid().ToString('N').Substring(0, 8))
    }
    $remoteProjectRoot = Get-EnvironmentDefault `
        'VITA_JETSON_REMOTE_PROJECT_ROOT' `
        '/data/code/VITA'
    $invoke = Join-Path $repositoryRoot 'scripts\ground\Invoke-VitaPayload.ps1'
    & $invoke `
        -SshTarget $sshTarget `
        -Sensor $SensorName `
        -PayloadInput $payloadInput `
        -Image $resolvedImage `
        -RegionId $resolvedRegionId `
        -JobId $resolvedJobId `
        -RemoteProjectRoot $remoteProjectRoot `
        -RemotePayloadPort $PayloadPort `
        -LocalTunnelPort 18090 `
        -GroundStore $groundStore `
        -SkipDashboard
    if ($LASTEXITCODE -ne 0) { throw "The stable Jetson $SensorName analysis failed." }
}

function Invoke-RemoteRawPipeline {
    if ($Image) { throw '-Image is not used for Balkan-1 raw scenes.' }
    $sshTarget = Get-EnvironmentDefault 'VITA_RAW_SSH_TARGET' ''
    if ([string]::IsNullOrWhiteSpace($sshTarget)) {
        throw 'Set VITA_RAW_SSH_TARGET=user@jetson before launching Balkan-1 raw analysis.'
    }
    $sceneId = if ($InputPath) { $InputPath } else { '3408' }
    $resolvedRegionId = if ($RegionId) { $RegionId } else { "balkan-raw-$sceneId" }
    $resolvedJobId = if ($JobId) {
        $JobId
    } else {
        'raw-{0}-{1}-{2}' -f `
            $sceneId,
            (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'),
            ([guid]::NewGuid().ToString('N').Substring(0, 8))
    }
    $remoteProjectRoot = Get-EnvironmentDefault 'VITA_RAW_REMOTE_PROJECT_ROOT' '/data/code/VITA'
    $invoke = Join-Path $repositoryRoot 'scripts\ground\Invoke-VitaPayload.ps1'
    & $invoke `
        -SshTarget $sshTarget `
        -Sensor 'balkan-1-raw' `
        -PayloadInput $sceneId `
        -RegionId $resolvedRegionId `
        -JobId $resolvedJobId `
        -RemoteProjectRoot $remoteProjectRoot `
        -RemoteRuntimeRoot 'runtime/raw-payload' `
        -RemotePayloadPort $RawPayloadPort `
        -LocalTunnelPort $RawTunnelPort `
        -GroundStore $groundStore `
        -SkipDashboard
    if ($LASTEXITCODE -ne 0) { throw 'The remote Balkan-1 raw analysis failed.' }
}

function Start-LocalWeb {
    $webRuntime = Join-Path $runtimeRoot 'web'
    foreach ($directory in @($groundStore, $webRuntime)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $uri = "http://127.0.0.1:$WebPort/"
    $ready = $false
    try {
        $null = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 3
        $ready = $true
    } catch {
        $listenerPid = Get-ListeningProcessId $WebPort
        if ($listenerPid) { throw "Port $WebPort is occupied by another application." }
    }
    if (-not $ready) {
        $dashboard = Get-LocalTool 'vita-dashboard'
        $web = Start-HiddenProcess `
            $dashboard `
            @('--store', $groundStore, '--host', '127.0.0.1', '--port', [string]$WebPort)
        foreach ($attempt in 1..30) {
            try {
                $null = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 3
                $ready = $true
                break
            } catch {
                if ($web.HasExited) {
                    throw "Dashboard exited during startup with code $($web.ExitCode)."
                }
                Start-Sleep -Milliseconds 500
            }
        }
        if ($ready) {
            $listenerPid = Get-ListeningProcessId $WebPort
            [ordered]@{
                listener_pid = $listenerPid
                starter_pid = $web.Id
                starter_start_time = $web.StartTime.ToUniversalTime().ToString('o')
                port = $WebPort
            } | ConvertTo-Json | Set-Content `
                -LiteralPath (Join-Path $webRuntime 'process.json') `
                -Encoding utf8
        }
    }
    if (-not $ready) { throw 'Dashboard did not become ready within 15 seconds.' }
    Write-Host "ViTA dashboard: $uri"
    if (-not $NoBrowser) { Start-Process $uri }
}

function Stop-RecordedProcess([string]$RecordPath, [string]$Name) {
    if (-not (Test-Path -LiteralPath $RecordPath)) { return }
    $record = Get-Content -LiteralPath $RecordPath -Raw | ConvertFrom-Json
    $listenerPid = Get-ListeningProcessId ([int]$record.port)
    if ($listenerPid -and $listenerPid -eq [int]$record.listener_pid) {
        Stop-Process -Id $listenerPid -Force
        Write-Host "Stopped $Name listener PID $listenerPid."
    }
    if ($record.starter_pid -and $record.starter_pid -ne $record.listener_pid) {
        $starter = Get-Process -Id ([int]$record.starter_pid) -ErrorAction SilentlyContinue
        if (
            $starter -and
            $starter.StartTime.ToUniversalTime().ToString('o') -eq $record.starter_start_time
        ) {
            Stop-Process -Id $starter.Id -Force
        }
    }
    Remove-Item -LiteralPath $RecordPath -Force
}

function Stop-LocalServices {
    Stop-RecordedProcess `
        (Join-Path $localRoot 'payload-process.json') `
        'payload service'
    Stop-RecordedProcess `
        (Join-Path $runtimeRoot 'web\process.json') `
        'web application'
}

switch ($Command) {
    'sentinel' {
        if (Get-JetsonSshTarget) { Invoke-StableJetsonPipeline 'sentinel-2' }
        else { Invoke-LocalPipeline 'sentinel-2' }
    }
    'balkan' {
        if (Get-JetsonSshTarget) { Invoke-StableJetsonPipeline 'balkan-1' }
        else { Invoke-LocalPipeline 'balkan-1' }
    }
    'raw' { Invoke-RemoteRawPipeline }
    'web' { Start-LocalWeb }
    'health' {
        $health = Get-Health
        if (-not $health) { throw "No payload service is ready at $payloadUri." }
        $health | ConvertTo-Json -Depth 12
    }
    'stop' { Stop-LocalServices }
    default {
        Write-Host @'
ViTA local MVP

  .\vita.ps1 sentinel   Run Sentinel-2, print timings, and ingest for the web app
  .\vita.ps1 balkan     Run Balkan-1, print timings, and ingest for the web app
  .\vita.ps1 raw        Run a warm Jetson Balkan-1 raw scene and ingest it
  .\vita.ps1 web        Start the web app and open it in the browser
  .\vita.ps1 health     Show the warm payload acceleration state
  .\vita.ps1 stop       Stop local services started by this script

Optional overrides: -InputPath, -Image, -RegionId, -JobId, -PayloadPort, -RawPayloadPort, -RawTunnelPort, -WebPort
Set VITA_JETSON_SSH_TARGET=user@jetson to use the stable Jetson service on port 8090.
Set VITA_RAW_SSH_TARGET=user@jetson to use the isolated warm raw service on port 8091.
'@
    }
}
