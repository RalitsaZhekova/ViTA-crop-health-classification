[CmdletBinding()]
param(
    [string]$SshTarget,

    [int]$SshPort = 22,
    [string]$IdentityFile,
    [string]$RemoteProjectRoot = '/data/code/VITA',
    [switch]$ValidateOnly,
    [switch]$SkipModels,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($SshPort -lt 1 -or $SshPort -gt 65535) {
    throw 'SshPort must be in the range 1..65535.'
}
if ($RemoteProjectRoot -notmatch '^/[A-Za-z0-9._/-]+$' -or $RemoteProjectRoot.Contains('..')) {
    throw 'RemoteProjectRoot must be a safe absolute POSIX path.'
}
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
function Read-AssetManifest([string]$ManifestName, [int]$ExpectedCount) {
    $manifestPath = Join-Path $repositoryRoot "deploy\payload\$ManifestName"
    $records = @(
        foreach ($line in Get-Content -LiteralPath $manifestPath) {
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith('#')) { continue }
        if ($line -notmatch '^([0-9a-f]{64})  ((?:data|payload/models)/[A-Za-z0-9._/-]+)$') {
            throw "Invalid demo asset manifest line: $line"
        }
        [pscustomobject]@{
            Sha256 = $Matches[1]
            RelativePath = $Matches[2]
        }
        }
    )
    if ($records.Count -ne $ExpectedCount) {
        throw "$ManifestName must contain exactly $ExpectedCount files; found $($records.Count)."
    }
    return $records
}

$dataAssets = @(Read-AssetManifest 'demo-assets.sha256' 6)
$modelAssets = if ($SkipModels) { @() } else { @(Read-AssetManifest 'model-assets.sha256' 3) }
$assets = @($dataAssets) + @($modelAssets)

Write-Host "Validating six demo-data files and $($modelAssets.Count) model files locally..."
foreach ($asset in $assets) {
    $localPath = Join-Path $repositoryRoot ($asset.RelativePath -replace '/', '\')
    if (-not (Test-Path -LiteralPath $localPath -PathType Leaf)) {
        throw "Required local payload asset is missing: $localPath"
    }
    $actual = (Get-FileHash -LiteralPath $localPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $asset.Sha256) {
        throw "Local SHA-256 mismatch for $($asset.RelativePath)"
    }
}
if ($ValidateOnly) {
    Write-Host "Payload asset package is valid: $($assets.Count) files."
    return
}
if ($SshTarget -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$') {
    throw 'SshTarget must use the form user@host.'
}

foreach ($command in @('ssh', 'scp')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "$command is required. Install the Windows OpenSSH Client feature."
    }
}
if ($IdentityFile) {
    $IdentityFile = (Resolve-Path -LiteralPath $IdentityFile).Path
}
$sshArguments = @(
    '-p', [string]$SshPort,
    '-o', 'BatchMode=yes',
    '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=3'
)
$scpArguments = @(
    '-P', [string]$SshPort,
    '-o', 'BatchMode=yes',
    '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=3'
)
if ($IdentityFile) {
    $sshArguments += @('-i', $IdentityFile)
    $scpArguments += @('-i', $IdentityFile)
}

function Invoke-Remote([string]$Command) {
    $output = & ssh @sshArguments $SshTarget $Command
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $Command" }
    return @($output)
}

$remoteDirectories = $assets |
    ForEach-Object { "$RemoteProjectRoot/$($_.RelativePath.Substring(0, $_.RelativePath.LastIndexOf('/')))" } |
    Sort-Object -Unique
$quotedDirectories = ($remoteDirectories | ForEach-Object { "'$_'" }) -join ' '
$null = Invoke-Remote "mkdir -p -- $quotedDirectories"

$copied = 0
$skipped = 0
foreach ($asset in $assets) {
    $localPath = Join-Path $repositoryRoot ($asset.RelativePath -replace '/', '\')
    $remotePath = "$RemoteProjectRoot/$($asset.RelativePath)"
    $remoteOutput = @(
        Invoke-Remote "if [ -f '$remotePath' ]; then sha256sum -- '$remotePath'; fi"
    )
    $remoteHash = if ($remoteOutput.Count) {
        ([string]$remoteOutput[0] -split '\s+')[0].ToLowerInvariant()
    } else {
        $null
    }
    if ($remoteHash -eq $asset.Sha256) {
        Write-Host "Already verified: $($asset.RelativePath)"
        $skipped++
        continue
    }
    if ($remoteHash -and -not $Force) {
        throw "Remote file differs: $remotePath. Re-run with -Force only if replacement is intended."
    }

    $temporaryPath = "$remotePath.partial-$([guid]::NewGuid().ToString('N'))"
    Write-Host "Copying: $($asset.RelativePath)"
    & scp @scpArguments $localPath "${SshTarget}:$temporaryPath"
    if ($LASTEXITCODE -ne 0) {
        $null = Invoke-Remote "rm -f -- '$temporaryPath'"
        throw "SCP failed for $($asset.RelativePath)."
    }
    $temporaryOutput = @(Invoke-Remote "sha256sum -- '$temporaryPath'")
    $temporaryHash = ([string]$temporaryOutput[0] -split '\s+')[0].ToLowerInvariant()
    if ($temporaryHash -ne $asset.Sha256) {
        $null = Invoke-Remote "rm -f -- '$temporaryPath'"
        throw "Transferred SHA-256 mismatch for $($asset.RelativePath)."
    }
    $null = Invoke-Remote "mv -f -- '$temporaryPath' '$remotePath'"
    $copied++
}

Write-Host "Payload assets ready: copied $copied, already present $skipped, total $($assets.Count)."
