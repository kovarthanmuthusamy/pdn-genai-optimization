# One active-learning iteration on Windows (propose is cross-platform; simulate needs Windows + PI/EMI).
# Edit CONFIG in pipelines/active_learning/run.py before running:
#   COMMAND, CONFIG_PATH, SKIP_SIMULATE, SKIP_INGEST, PROPOSE_ONLY

param(
  [switch]$ProposeOnly
)

$ganRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$py = "python"

Set-Location $ganRoot

if ($ProposeOnly) {
  Write-Host "Set PROPOSE_ONLY = True in pipelines/active_learning/run.py, then re-run without -ProposeOnly"
}

& $py pipelines/active_learning/run.py
exit $LASTEXITCODE
