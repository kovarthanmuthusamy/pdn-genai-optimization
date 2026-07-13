#Requires AutoHotkey v2.0
#SingleInstance Force

; Helper: capture "Tools" menu click coordinates.
; Steps:
; 1) Make PI/EMI window active.
; 2) Hover mouse exactly on the "Tools" menu text.
; 3) Press F8.
; It copies client-relative X/Y to clipboard and shows a popup.

CoordMode "Mouse", "Client"

F8::{
  MouseGetPos &x, &y, &winId
  if !winId {
    MsgBox "No active window detected."
    return
  }
  title := WinGetTitle("ahk_id " winId)
  text := "WinTitle=" title "`nToolsMenuClickX=" x "`nToolsMenuClickY=" y
  A_Clipboard := text
  MsgBox "Captured and copied to clipboard:`n`n" text
}

