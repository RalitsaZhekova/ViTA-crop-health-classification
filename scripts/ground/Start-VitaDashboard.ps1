[CmdletBinding()]
param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$dashboardUrl = "http://127.0.0.1:$Port/"
$healthUrl = "http://127.0.0.1:$Port/api/v1/health"
$groundStore = Join-Path $repositoryRoot 'runtime\ground'

function Test-VitaDashboardHealth {
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2
        return $health.status -eq 'ok' -and $health.service -eq 'vita-crop-ground'
    } catch {
        return $false
    }
}

if (Test-VitaDashboardHealth) {
    Write-Host "ViTA dashboard is already running: $dashboardUrl"
    exit 0
}

$dockerReady = $false
if (Get-Command docker -ErrorAction SilentlyContinue) {
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        & docker info --format '{{.ServerVersion}}' *> $null
        $dockerReady = $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
}

if ($dockerReady) {
    Push-Location $repositoryRoot
    try {
        $env:VITA_DASHBOARD_PORT = [string]$Port
        & docker compose -f deploy\compose.ground.yaml up -d
        $composeStarted = $LASTEXITCODE -eq 0
    } finally {
        Pop-Location
    }
    if ($composeStarted) {
        foreach ($attempt in 1..60) {
            if (Test-VitaDashboardHealth) {
                Write-Host "ViTA dashboard: $dashboardUrl"
                exit 0
            }
            Start-Sleep -Milliseconds 500
        }
        Write-Warning 'The dashboard container started but did not become healthy; trying the local launcher.'
    } else {
        Write-Warning 'Docker Compose could not start the dashboard; trying the local launcher.'
    }
} else {
    Write-Host 'Docker Desktop is unavailable; starting the installed local dashboard.'
}

$localDashboard = Join-Path $repositoryRoot '.venv\Scripts\vita-dashboard.exe'
if (-not (Test-Path -LiteralPath $localDashboard)) {
    $installedDashboard = Get-Command vita-dashboard -ErrorAction SilentlyContinue
    if (-not $installedDashboard) {
        throw 'vita-dashboard is unavailable. Run .venv\Scripts\python -m pip install -e . on the ground.'
    }
    $localDashboard = $installedDashboard.Source
}

New-Item -ItemType Directory -Path $groundStore -Force | Out-Null
$logRoot = Join-Path $groundStore '.dashboard'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$argumentLine = '--store "{0}" --host 127.0.0.1 --port {1}' -f $groundStore, $Port
$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $localDashboard
$startInfo.Arguments = $argumentLine
$startInfo.UseShellExecute = $true
$startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$dashboardProcess = [System.Diagnostics.Process]::Start($startInfo)

foreach ($attempt in 1..60) {
    if (Test-VitaDashboardHealth) {
        Set-Content -LiteralPath (Join-Path $logRoot 'local-dashboard.pid') -Value $dashboardProcess.Id
        Write-Host "ViTA dashboard (local process $($dashboardProcess.Id)): $dashboardUrl"
        exit 0
    }
    if ($dashboardProcess.HasExited) { break }
    Start-Sleep -Milliseconds 500
}

if (-not $dashboardProcess.HasExited) {
    Stop-Process -Id $dashboardProcess.Id -Force
}
$exitDetail = if ($dashboardProcess.HasExited) {
    "The launcher exited with code $($dashboardProcess.ExitCode)."
} else {
    'The launcher did not become healthy before the timeout.'
}
throw "Ground dashboard failed to start locally. $exitDetail Run $localDashboard in the foreground for details."
