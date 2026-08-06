# Force-close any running ECADStar PI/EMI Analysis window (manual abort helper).
# Usage (from Windows):  powershell -NoProfile -ExecutionPolicy Bypass -File kill_ecadstar.ps1
# From WSL:              powershell.exe -NoProfile -File "$(wslpath -w tools/ecadstar/kill_ecadstar.ps1)"
$titles = @('*eCADSTAR*PI/EMI*', '*PI/EMI Analysis*')
$procs = Get-Process | Where-Object {
  $mt = $_.MainWindowTitle
  if ([string]::IsNullOrEmpty($mt)) { $false }
  else { ($titles | Where-Object { $mt -like $_ }).Count -gt 0 }
}
if (-not $procs) { Write-Host 'No running ECADStar PI/EMI instance.'; exit 0 }
foreach ($p in $procs) {
  Write-Host ("Killing PID {0}  '{1}'" -f $p.Id, $p.MainWindowTitle)
  try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch { Write-Warning $_ }
}
Write-Host 'Done.'
