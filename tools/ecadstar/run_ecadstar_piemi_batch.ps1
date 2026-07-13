param(
  [Parameter(Mandatory=$true)]
  [string]$ErfPath,

  [Parameter(Mandatory=$true)]
  [string]$PebPath,

  [string]$AhkExe = "$env:LOCALAPPDATA\Programs\AutoHotkey\v2\AutoHotkey64.exe",

  [string]$ScriptPath = "",

  [switch]$SkipOpenErf
)

if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
  $baseDir = $PSScriptRoot
  if ([string]::IsNullOrWhiteSpace($baseDir)) {
    $baseDir = Split-Path -Parent $PSCommandPath
  }
  if ([string]::IsNullOrWhiteSpace($baseDir)) {
    $baseDir = (Get-Location).Path
  }
  $ScriptPath = Join-Path $baseDir "ecadstar_piemi_batch.ahk"
}

if (-not (Test-Path -LiteralPath $ErfPath)) {
  throw "ERF/RIF file not found: $ErfPath"
}

if (-not (Test-Path -LiteralPath $PebPath)) {
  throw "PEB file not found: $PebPath"
}

if (-not (Test-Path -LiteralPath $ScriptPath)) {
  throw "AHK script not found: $ScriptPath"
}

if (-not (Test-Path -LiteralPath $AhkExe)) {
  throw "AutoHotkey v2 not found at: $AhkExe`nInstall AutoHotkey v2 or pass -AhkExe with the correct path."
}

Write-Host "Starting PI/EMI UI automation..."
Write-Host "  ERF: $ErfPath"
Write-Host "  PEB: $PebPath"
Write-Host "  AHK: $ScriptPath"
Write-Host "  Log: $env:TEMP\ecadstar_piemi_batch.log"
if ($SkipOpenErf) {
  Write-Host "  Mode: skipopen (design already open in PI/EMI)"
  & $AhkExe $ScriptPath $ErfPath $PebPath skipopen
} else {
  & $AhkExe $ScriptPath $ErfPath $PebPath
}
Write-Host ""
Write-Host "If something failed, open the log:"
Write-Host "  notepad $env:TEMP\ecadstar_piemi_batch.log"

