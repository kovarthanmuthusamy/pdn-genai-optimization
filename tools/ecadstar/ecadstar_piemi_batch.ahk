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
tAfterOpenSaveDialog := 3000   ; save dialog always appears after Open Enter — wait then Enter once
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
tBatchStartPoll := 2000        ; poll PI-1 log while keeping PI/EMI on screen
tBatchStartTimeout := 600000   ; 10 min max wait for batch to actually start

UseMouseForToolsMenu := true
ToolsMenuClickX := 89
ToolsMenuClickY := 7

; Load Batch file dialog — paste full path to the .peb file (e.g. C:\...\batch.peb).
; One Down arrow before Enter (set false to test if Down arrow changes batch behaviour)
UsePebFileNameOnly := false
UseDownArrowBeforeEnter := true

LogPath := A_Temp "\ecadstar_piemi_batch.log"

SendMode "Input"
SetKeyDelay tBetweenKeys, tBetweenKeys

Log(msg) {
  global LogPath
  line := Format("{}  {}\n", A_Now, msg)
  try FileAppend line, LogPath, "UTF-8"
}

PiemiExists() {
  global WinTitleMatch
  return WinExist(WinTitleMatch)
}

; Restore + foreground tricks help when the RDP session is visible but not focused.
ActivatePiemi(timeoutSec := 20) {
  global WinTitleMatch
  if !WinExist(WinTitleMatch)
    return false

  hwnd := WinExist(WinTitleMatch)
  try {
    if WinGetMinMax("ahk_id " hwnd) = -1
      WinRestore "ahk_id " hwnd
    WinShow "ahk_id " hwnd
  }

  Loop 3 {
    Send "{Alt down}"
    Sleep 40
    Send "{Alt up}"
    Sleep 80
    WinActivate "ahk_id " hwnd
    if WinWaitActive(WinTitleMatch, , timeoutSec)
      return true
    Sleep 500
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
  if PiemiExists() && !WinActive(WinTitleMatch)
    Log("WARNING: PI/EMI not foreground — keep RDP window visible/focused (automation cannot click minimized sessions)")
  return true
}

PastePathIntoOpenDialog(path) {
  global tAfterOpenSaveDialog
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
  Log("Submitted Open dialog path: " path)
  ; Save dialog always appears after Open — wait 3s then Enter once.
  Sleep tAfterOpenSaveDialog
  Log("Save dialog after open — Enter")
  Send "{Enter}"
  Sleep 500
  A_Clipboard := saved
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
    if WinExist("ahk_class #32770") {
      Send "{Enter}"
      Sleep 300
    }
  }
}

RemoveDesignLock(erfPath) {
  lockPath := RegExReplace(erfPath, "\.[^.\\]+$", ".rlk")
  if FileExist(lockPath) {
    try FileDelete lockPath
    Log("Removed stale design lock: " lockPath)
  }
}

DismissRemoveLockDialog() {
  Loop 3 {
    if !WinExist("ahk_class #32770")
      return
    WinActivate "ahk_class #32770"
    Sleep 300
    txt := WinGetText("ahk_class #32770")
    if RegExMatch(txt, "i)remove\s+lock|lock\s+file|\.rlk") {
      Log("Dismiss remove-lock dialog")
      Send "!y"
      Sleep 400
      if WinExist("ahk_class #32770") {
        Send "{Enter}"
        Sleep 300
      }
    } else {
      DismissModalIfAny()
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

EmcDirFromErf(erfPath) {
  return RegExReplace(erfPath, "\\[^\\]+$", "")
}

; Keep PI/EMI foreground until PI-1 log shows this PEB batch actually started.
WaitBatchSimulationStart(erfPath, pebPath, timeoutMs := 600000) {
  global tBatchStartPoll
  emcDir := EmcDirFromErf(erfPath)
  piLog := emcDir "\PI-1\log.txt"
  pebName := RegExReplace(pebPath, ".*\\", "")
  submitTick := A_TickCount
  Log(Format(
    "WaitBatchStart: keep RDP on PI/EMI until simulation starts (max {}s)",
    timeoutMs // 1000
  ))
  lastLogSec := -1
  Loop {
    elapsed := A_TickCount - submitTick
    if (elapsed > timeoutMs) {
      Log("ERROR: Batch simulation start not confirmed in PI-1/log.txt")
      return false
    }
    ActivatePiemi(2)
    if FileExist(piLog) {
      try txt := FileRead(piLog, "UTF-8")
      if (InStr(txt, pebName)
          && (InStr(txt, "Perform batch step") || InStr(txt, "Batch file"))) {
        Log("Batch simulation started — " pebName " confirmed in PI-1/log.txt")
        Sleep 2000
        return true
      }
    }
    elapsedSec := elapsed // 1000
    if (elapsedSec // 10 != lastLogSec // 10) {
      lastLogSec := elapsedSec
      Log(Format(
        "Waiting for batch start... {}s — keep PI/EMI window visible/focused",
        elapsedSec
      ))
    }
    Sleep tBatchStartPoll
  }
}

LoadBatch() {
  global pebPath, analysisPath, WinTitleMatch, UseMouseForToolsMenu
  global ToolsMenuClickX, ToolsMenuClickY
  global tToolsMenuOpen, tAfterShiftTabs, tBatchDialogOpen
  global tAfterPebType, tAfterDown, tBatchStartTimeout
  global UsePebFileNameOnly, UseDownArrowBeforeEnter

  pebToType := pebPath
  if (UsePebFileNameOnly)
    pebToType := RegExReplace(pebPath, ".*\\", "")

  DismissRemoveLockDialog()
  DismissModalIfAny()
  Sleep 800

  activated := false
  Loop 12 {
    if ActivatePiemi(5) {
      activated := true
      break
    }
    Log(Format("Load Batch: waiting for PI/EMI foreground (attempt {}/12) — RDP must stay visible", A_Index))
    Sleep 5000
  }
  if !activated {
    Log("ERROR: Cannot activate PI/EMI for Load Batch (window missing or RDP session not foreground)")
    return false
  }
  Sleep 500

  methods := ["mouse_shifttab", "alt_t_shifttab", "alt_t_l", "alt_t_down"]
  dialogOpen := false
  for method in methods {
    Log("Load Batch menu attempt: " method)
    if TryInvokeLoadBatchMenu(method) {
      dialogOpen := true
      break
    }
    Log("Load Batch dialog not seen — trying next menu method")
    Send "{Esc}"
    Sleep 300
    DismissModalIfAny()
    ActivatePiemi(3)
    Sleep 400
  }
  if !dialogOpen {
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
  Log("Load Batch submitted — typed: " pebToType . " (full path: " pebPath . ")")
  if !WaitBatchSimulationStart(analysisPath, pebPath, tBatchStartTimeout) {
    Log("ERROR: Load Batch submitted but ECADStar batch did not start")
    return false
  }
  return true
}

TryInvokeLoadBatchMenu(method) {
  global UseMouseForToolsMenu, ToolsMenuClickX, ToolsMenuClickY
  global tToolsMenuOpen, tAfterShiftTabs, tBatchDialogOpen

  Send "{Esc}"
  Sleep 150

  if (method = "mouse_shifttab") {
    if (UseMouseForToolsMenu) {
      CoordMode "Mouse", "Client"
      Log("Click Tools at " ToolsMenuClickX "," ToolsMenuClickY)
      Click ToolsMenuClickX, ToolsMenuClickY
    } else {
      Send "!t"
    }
    Sleep tToolsMenuOpen
    Send "+{Tab 3}"
    Sleep tAfterShiftTabs
    Send "{Enter}"
  } else if (method = "alt_t_shifttab") {
    Log("Keyboard Tools: Alt+T then Shift+Tab x3")
    Send "!t"
    Sleep tToolsMenuOpen
    Send "+{Tab 3}"
    Sleep tAfterShiftTabs
    Send "{Enter}"
  } else if (method = "alt_t_l") {
    Log("Keyboard Tools: Alt+T then L (Load Batch)")
    Send "!t"
    Sleep tToolsMenuOpen
    Send "l"
  } else if (method = "alt_t_down") {
    Log("Keyboard Tools: Alt+T then Down x4 + Enter")
    Send "!t"
    Sleep tToolsMenuOpen
    Send "{Down 4}{Enter}"
  } else {
    return false
  }

  Sleep tBatchDialogOpen
  return WaitForFileDialog(8)
}

; ========== Main ==========
try FileDelete LogPath
Log("=== Start ===")
Log("ERF: " analysisPath)
Log("PEB: " pebPath)
Log("skipopen: " (skipOpenErf ? "yes" : "no"))

RemoveDesignLock(analysisPath)

opened := false
if (skipOpenErf) {
  if PiemiExists() {
    if ActivatePiemi() {
      Log("skipopen: using already-open PI/EMI")
      opened := true
      WaitAnalysisReady(tReadyMinSkipOpen, tReadyMaxSkipOpen)
    } else {
      Log("skipopen: PI/EMI window exists but cannot foreground — retrying (keep RDP visible)")
      Loop 12 {
        Sleep 5000
        if ActivatePiemi(5) {
          Log("skipopen: PI/EMI foreground restored")
          opened := true
          WaitAnalysisReady(tReadyMinSkipOpen, tReadyMaxSkipOpen)
          break
        }
      }
    }
  }
  if !opened {
    Log("skipopen but PI/EMI unavailable — trying shell open")
    opened := OpenErfViaShell()
  }
} else if PiemiExists() {
  if ActivatePiemi() {
    Log("PI/EMI already running — open .erf via Ctrl+O")
    opened := OpenErfViaMenu()
  } else {
    Log("PI/EMI window exists but cannot foreground — waiting (do not minimize RDP)")
    Loop 12 {
      Sleep 5000
      if ActivatePiemi(5) {
        Log("PI/EMI foreground restored — open .erf via Ctrl+O")
        opened := OpenErfViaMenu()
        break
      }
    }
    if !opened {
      Log("ERROR: PI/EMI exists but never became foreground — shell open skipped to avoid duplicate instance")
    }
  }
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

DismissRemoveLockDialog()

DismissModalIfAny()
Sleep 1000

if !LoadBatch() {
  MsgBox "Opened .erf but Load Batch failed.`n`nLog file:`n" LogPath
  ExitApp 5
}

Log("=== Done — batch running in PI/EMI; script exits ===")
ExitApp 0
