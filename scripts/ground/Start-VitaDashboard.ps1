[CmdletBinding()]
param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location $repositoryRoot
try {
    $env:VITA_DASHBOARD_PORT = [string]$Port
    & docker compose -f deploy\compose.ground.yaml up -d
    if ($LASTEXITCODE -ne 0) { throw 'Ground dashboard container failed to start.' }
    Write-Host "ViTA dashboard: http://127.0.0.1:$Port/"
} finally {
    Pop-Location
}

