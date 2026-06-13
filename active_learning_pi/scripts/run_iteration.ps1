# One active-learning iteration on Windows (propose is cross-platform; simulate needs Windows + PI/EMI).
param(
  [string]$Config = "",
  [switch]$SkipSimulate,
  [switch]$SkipIngest,
  [switch]$ProposeOnly
)

$ganRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$py = "python"
$cfgArg = @()
if ($Config) { $cfgArg = @("--config", $Config) }

Set-Location $ganRoot

if ($ProposeOnly) {
  & $py active_learning_pi/run_pipeline.py generate @cfgArg
  & $py active_learning_pi/run_pipeline.py infer @cfgArg
  exit $LASTEXITCODE
}

$args = @("active_learning_pi/run_pipeline.py", "cycle") + $cfgArg
if ($SkipSimulate) { $args += "--skip-simulate" }
if ($SkipIngest) { $args += "--skip-ingest" }

& $py @args
exit $LASTEXITCODE
