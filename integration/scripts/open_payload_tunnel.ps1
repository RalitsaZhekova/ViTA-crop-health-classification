[CmdletBinding()]
param(
    [string]$SshHost = "vita-payload",
    [ValidateRange(1, 65535)]
    [int]$LocalPort = 18081
)

$ErrorActionPreference = "Stop"

ssh `
    -N `
    -L "${LocalPort}:127.0.0.1:8081" `
    $SshHost
