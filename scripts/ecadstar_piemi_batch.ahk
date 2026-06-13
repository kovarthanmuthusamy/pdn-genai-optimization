#Requires AutoHotkey v2.0
#SingleInstance Force

; eCADSTAR PI/EMI batch runner (GUI automation).
; Usage:
;   AutoHotkey64.exe ecadstar_piemi_batch.ahk "C:\path\analysis.erf" "C:\path\batch.peb"
;   AutoHotkey64.exe ecadstar_piemi_batch.ahk "C:\path\analysis.erf" "C:\path\batch.peb" skipopen
;     skipopen = design already open; only run Load Batch

analysisPath := A_Args.Length >= 1 ? A_Args[1] : ""
pebPath := A_Args.Length >= 2 ? A_Args[2] : ""
skipOpenErf := A_Args.Length >= 3 && StrLower(A_Args[3]) = "skipopen"

if (analysisPath = "") {
  MsgBox 'Missing argument: path to .erf (or .rif) file.`n`nExample:`n  AutoHotkey64.exe ecadstar_piemi_batch.ahk "C:\...\h-design.erf" "C:\...\batch.peb"'
  ExitApp 2
}
if (pebPath = "") {
  MsgBox 'Missing argument: path to .peb batch file.'
  ExitApp 2
}

if !FileExist(analysisPath) {
  MsgBox "ERF file not found:`n" analysisPath
  ExitApp 2
}
if !FileExist(pebPath) {
  MsgBox "PEB file not found:`n" pebPath
  ExitApp 2
}

; --- Settings ---
WinTitle := "ahk_exe "  ; filled after we only use partial title match
WinTitleMatch := "eCADSTAR PI/EMI Analysis"

tLaunch := 60000
tOpenDialog := 2000
; Wait until UI is ready (replaces fixed 30s+8s sleeps — those caused ~38s delay before Load Batch)
tReadyMinShell := 2000       ; minimum after shell/double-click open
tReadyMaxShell := 12000      ; stop waiting early once no modal dialog blocks PI/EMI
tReadyMinMenu := 4000        ; minimum after Ctrl+O open (heavier reload)
tReadyMaxMenu := 45000
tReadyMinSkipOpen := 500     ; design already open
tReadyMaxSkipOpen := 5000
tBetweenKeys := 50
tToolsMenuOpen := 400
tAfterShiftTabs := 200
tBatchDialogOpen := 2500
tAfterPebType := 200
tAfterDown := 150

UseMouseForToolsMenu := true
ToolsMenuClickX := 89
ToolsMenuClickY := 7

; Load Batch file dialog — match your manual steps exactly:
; - Type file name only (not full path) when .peb is in the same folder as .erf
; - One Down arrow before Enter (set false to test if Down arrow changes batch behaviour)
UsePebFileNameOnly := true
UseDownArrowBeforeEnter := true

LogPath := A_Temp "\ecadstar_piemi_batch.log"

SendMode "Input"
SetKeyDelay tBetweenKeys, tBetweenKeys

Log(msg) {
  global LogPath
  line := Format("{}  {}\n", A_Now, msg)
  try FileAppend line, LogPath, "UTF-8"
}

ActivatePiemi() {
  global WinTitleMatch
  if WinExist(WinTitleMatch) {
    WinActivate WinTitleMatch
    if WinWaitActive(WinTitleMatch, , 20)
      return true
  }
  return false
}

WaitForFileDialog(timeoutSec := 20) {
  return WinWait("ahk_class #32770", , timeoutSec)
}

; Wait until PI/EMI is active and no blocking modal dialog (poll, don't sleep 38s fixed).
WaitAnalysisReady(minMs, maxMs) {
  global WinTitleMatch, LogPath
  Log(Format("WaitAnalysisReady min={}ms max={}ms", minMs, maxMs))
  Sleep minMs
  start := A_TickCount
  stableCount := 0
  while (A_TickCount - start < maxMs) {
    if WinExist(WinTitleMatch) && WinActive(WinTitleMatch) && !WinExist("ahk_class #32770") {
      stableCount++
      if (stableCount >= 2) {
        Log(Format("UI ready after {}ms", A_TickCount - start + minMs))
        return true
      }
    } else {
      stableCount := 0
    }
    Sleep 500
  }
  Log(Format("UI wait timeout after {}ms — continuing anyway", maxMs))
  return true
}

PastePathIntoOpenDialog(path) {
  global
  if !WaitForFileDialog(25) {
    Log("ERROR: File Open dialog did not appear after Ctrl+O")
    return false
  }
  WinActivate "ahk_class #32770"
  WinWaitActive "ahk_class #32770", , 10
  Sleep 400

  ; Focus "File name" field (common in Windows dialogs)
  Send "!n"
  Sleep 200
  Send "^a"
  Sleep 100

  saved := A_Clipboard
  A_Clipboard := path
  if !ClipWait(2) {
    Log("ERROR: Could not set clipboard for path")
    return false
  }
  Sleep 100
  Send "^v"
  Sleep 400
  Send "{Enter}"
  Sleep 500
  A_Clipboard := saved
  Log("Submitted Open dialog path: " path)
  return true
}

DismissModalIfAny() {
  if WinExist("ahk_class #32770") {
    WinActivate "ahk_class #32770"
    Sleep 300
    ; Discard / No / Cancel attempts (depends on dialog)
    Send "!d"
    Sleep 400
    if WinExist("ahk_class #32770") {
      Send "!n"
      Sleep 300
    }
  }
}

OpenErfViaMenu() {
  global analysisPath, tReadyMinMenu, tReadyMaxMenu
  if !ActivatePiemi() {
    Log("ERROR: Cannot activate PI/EMI window")
    return false
  }
  Sleep 800
  Log("Sending Ctrl+O to open .erf")
  Send "^o"
  Sleep tOpenDialog

  if !PastePathIntoOpenDialog(analysisPath) {
    return false
  }

  ; Wait for Open dialog to close
  Loop 30 {
    if !WinExist("ahk_class #32770")
      break
    Sleep 1000
  }
  DismissModalIfAny()

  WaitAnalysisReady(tReadyMinMenu, tReadyMaxMenu)
  return true
}

OpenErfViaShell() {
  global analysisPath, WinTitleMatch, tLaunch, tReadyMinShell, tReadyMaxShell
  Log("Launching .erf with shell: " analysisPath)
  try Run analysisPath
  catch as err {
    Log("ERROR Run .erf: " err.Message)
    return false
  }
  if WinWait(WinTitleMatch, , tLaunch / 1000) {
    WinActivate WinTitleMatch
    WinWaitActive WinTitleMatch, , 30
    Log("PI/EMI window appeared after shell open")
    WaitAnalysisReady(tReadyMinShell, tReadyMaxShell)
    return true
  }
  Log("ERROR: PI/EMI window not found after shell open")
  return false
}

LoadBatch() {
  global pebPath, WinTitleMatch, UseMouseForToolsMenu
  global ToolsMenuClickX, ToolsMenuClickY
  global tToolsMenuOpen, tAfterShiftTabs, tBatchDialogOpen
  global tAfterPebType, tAfterDown
  global UsePebFileNameOnly, UseDownArrowBeforeEnter

  pebToType := pebPath
  if (UsePebFileNameOnly)
    pebToType := RegExReplace(pebPath, ".*\\", "")

  if !ActivatePiemi() {
    Log("ERROR: Cannot activate PI/EMI for Load Batch")
    return false
  }
  Sleep 500

  if (UseMouseForToolsMenu) {
    CoordMode "Mouse", "Client"
    Log("Click Tools at " ToolsMenuClickX "," ToolsMenuClickY)
    Click ToolsMenuClickX, ToolsMenuClickY
    Sleep tToolsMenuOpen
  } else {
    Send "!t"
    Sleep tToolsMenuOpen
  }

  Send "+{Tab 3}"
  Sleep tAfterShiftTabs
  Send "{Enter}"
  Sleep tBatchDialogOpen

  if !WaitForFileDialog(20) {
    Log("ERROR: Load Batch file dialog did not appear")
    return false
  }
  WinActivate "ahk_class #32770"
  Sleep 300
  Send "!n"
  Sleep 200
  Send "^a"
  A_Clipboard := pebToType
  ClipWait 2
  Send "^v"
  Sleep tAfterPebType
  if (UseDownArrowBeforeEnter) {
    Send "{Down}"
    Sleep tAfterDown
  }
  Send "{Enter}"
  Log("Load Batch started — typed: " pebToType . " (full path: " pebPath . ")")
  return true
}

; ========== Main ==========
try FileDelete LogPath
Log("=== Start ===")
Log("ERF: " analysisPath)
Log("PEB: " pebPath)
Log("skipopen: " (skipOpenErf ? "yes" : "no"))

opened := false
if (skipOpenErf) {
  if ActivatePiemi() {
    Log("skipopen: using already-open PI/EMI")
    opened := true
    WaitAnalysisReady(tReadyMinSkipOpen, tReadyMaxSkipOpen)
  } else {
    Log("skipopen but no PI/EMI window — trying shell open")
    opened := OpenErfViaShell()
  }
} else if ActivatePiemi() {
  Log("PI/EMI already running — open .erf via Ctrl+O")
  opened := OpenErfViaMenu()
} else {
  Log("PI/EMI not running — open .erf via shell (double-click style)")
  opened := OpenErfViaShell()
  if !opened {
    Log("Shell open failed — waiting 5s then retry Ctrl+O if window appears")
    Sleep 5000
    if ActivatePiemi()
      opened := OpenErfViaMenu()
  }
}

if !opened {
  MsgBox "Could not open the analysis (.erf).`n`nLog file:`n" LogPath . "`n`nTips:`n1) Open PI/EMI Analysis manually first`n2) Re-run with 3rd arg skipopen if design is already open"
  ExitApp 4
}

if !LoadBatch() {
  MsgBox "Opened .erf but Load Batch failed.`n`nLog file:`n" LogPath
  ExitApp 5
}

Log("=== Done — Load Batch triggered; script exits (simulation runs in PI/EMI) ===")
ExitApp 0
