param(
  [string]$ErfPath = "",
  [int]$GraceSec = 45,
  [switch]$ForceIfNeeded
)

$ErrorActionPreference = "Stop"
$logPath = Join-Path $env:TEMP "ecadstar_piemi_close.log"

function Write-Log([string]$Message) {
  $line = "$(Get-Date -Format 'yyyyMMdd_HHmmss')  $Message"
  Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
  Write-Host $line
}

function Get-PiemiProcess {
  Get-Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like '*PI/EMI*'
    }
}

function Dismiss-ModalDialogs([int]$MaxRounds = 12) {
  Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32Close {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);
}
"@

  for ($i = 0; $i -lt $MaxRounds; $i++) {
    $dialog = Get-Process -ErrorAction SilentlyContinue |
      Where-Object { $_.MainWindowClassName -eq '#32770' -and $_.MainWindowHandle -ne 0 } |
      Select-Object -First 1
    if (-not $dialog) { return }
    $title = $dialog.MainWindowTitle
    Write-Log "Dismiss modal dialog: $title"
    [Win32Close]::SetForegroundWindow($dialog.MainWindowHandle) | Out-Null
    Start-Sleep -Milliseconds 500
    # Save / unsaved-changes on quit: confirm with Enter (default button).
    Write-Log "Send Enter on close dialog"
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
    Start-Sleep -Milliseconds 700
    if (Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowClassName -eq '#32770' }) {
      Write-Log "Dialog still open — try Discard (Alt+D)"
      [System.Windows.Forms.SendKeys]::SendWait('%d')
      Start-Sleep -Milliseconds 500
    }
    if (Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowClassName -eq '#32770' }) {
      Write-Log "Dialog still open — retry Enter"
      [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
      Start-Sleep -Milliseconds 500
    }
  }
}

function Remove-DesignLock([string]$Path) {
  if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
  $erf = [System.IO.Path]::GetFullPath($Path)
  $dir = [System.IO.Path]::GetDirectoryName($erf)
  $stem = [System.IO.Path]::GetFileNameWithoutExtension($erf)
  $lockPath = Join-Path $dir "$stem.rlk"
  if (-not (Test-Path -LiteralPath $lockPath)) { return $false }
  Remove-Item -LiteralPath $lockPath -Force
  Write-Log "Removed lock file: $lockPath"
  return $true
}

try {
  if (Test-Path -LiteralPath $logPath) { Remove-Item -LiteralPath $logPath -Force }
} catch {}

Add-Type -AssemblyName System.Windows.Forms
Write-Log "=== PI/EMI close start ==="

$proc = Get-PiemiProcess | Select-Object -First 1
if ($proc) {
  Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32CloseMain {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);
}
"@
  Write-Log "Graceful close for PID $($proc.Id): $($proc.MainWindowTitle)"
  [Win32CloseMain]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null
  Start-Sleep -Milliseconds 500
  # WM_CLOSE = 0x0010
  [Win32CloseMain]::PostMessage($proc.MainWindowHandle, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null

  $deadline = (Get-Date).AddSeconds($GraceSec)
  while ((Get-Date) -lt $deadline) {
    Dismiss-ModalDialogs
    if (-not (Get-PiemiProcess)) {
      Write-Log "PI/EMI exited gracefully"
      break
    }
    Start-Sleep -Milliseconds 800
  }

  # Save dialog can appear after the main window closes — keep sending Enter.
  $dialogDeadline = (Get-Date).AddSeconds(15)
  while ((Get-Date) -lt $dialogDeadline) {
    $dialog = Get-Process -ErrorAction SilentlyContinue |
      Where-Object { $_.MainWindowClassName -eq '#32770' -and $_.MainWindowHandle -ne 0 } |
      Select-Object -First 1
    if (-not $dialog) { break }
    Write-Log "Post-close save dialog still open — Enter"
    [Win32CloseMain]::SetForegroundWindow($dialog.MainWindowHandle) | Out-Null
    Start-Sleep -Milliseconds 400
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
    Start-Sleep -Milliseconds 800
  }

  $still = Get-PiemiProcess | Select-Object -First 1
  if ($still -and $ForceIfNeeded) {
    Write-Log "Force-stopping PID $($still.Id) after timeout"
    Stop-Process -Id $still.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
  }
} else {
  Write-Log "No PI/EMI window found"
}

Remove-DesignLock $ErfPath | Out-Null
Write-Log "=== PI/EMI close done ==="
