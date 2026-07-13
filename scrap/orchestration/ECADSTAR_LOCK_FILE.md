# ECADSTAR automation notes

## Window close (disabled)

`ECADSTAR_CLOSE_AFTER_BATCH = False` — PI/EMI stays open after batch simulation.

## Save dialog when opening .erf (Ctrl+O)

After pasting the path and pressing Enter in the Open dialog, the save dialog always appears. AHK waits **3 seconds**, then presses **Enter once** (`tAfterOpenSaveDialog := 3000` in `ecadstar_piemi_batch.ahk`).

## Stale lock file (.rlk)

If you still see **Remove lock**, delete `H-shape.rlk` beside the `.erf` or set `ECADSTAR_CLEAR_LOCK_FILE = True` (default).

```powershell
Remove-Item "C:\Users\muthusamy\Desktop\design\H-shape.emc\H-shape.rlk" -Force
```
