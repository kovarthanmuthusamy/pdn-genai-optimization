# Fix: ECADStar AHK crash "Target window not found" (WinActivate)

> **OBSOLETE (2026-08-04).** AutoHotkey has been removed entirely and replaced by
> the native headless CLI (`engineer.exe --batch --batch-auto-exit`). The
> `WinActivate` crash class described here can no longer occur (there is no GUI
> automation). See [`ecadstar_headless_cli.md`](./ecadstar_headless_cli.md). Kept
> for historical context only.

## 📝 Summary of Changes

Hardened `tools/ecadstar/ecadstar_piemi_batch.ahk` so a transient window-handle
change during batch startup no longer crashes the automation script.

- Rewrote `ActivatePiemi()` to **re-resolve** the PI/EMI window handle on every
  attempt and wrap `WinActivate "ahk_id " hwnd` in `try/catch` (soft-fail + retry).
- Added `SafeActivate(target)` helper and routed all remaining bare `WinActivate`
  calls through it (`ahk_class #32770` file/modal dialogs and the shell-open
  `WinTitleMatch` activation).

## 🚀 Implementation Details

### Root cause
The error was **not** a simulation failure — the batch had already started. It was
an uncaught AutoHotkey v2 exception:

```
Error: Target window not found.  (ahk_id 66700)
  098: WinActivate("ahk_id " hwnd)
  ... ActivatePiemi(2) <- WaitBatchSimulationStart <- LoadBatch <- Auto-execute
```

Sequence:
1. `ActivatePiemi()` captured the PI/EMI handle once via `hwnd := WinExist(...)`.
2. It then sent `Alt down/up` and slept ~120 ms per loop.
3. It was being polled from `WaitBatchSimulationStart()` exactly as the batch
   starts, and ECADStar destroys/recreates that top-level PI/EMI window, so the
   captured handle (`66700`) became invalid.
4. In **AHK v2**, `WinActivate "ahk_id " <stale>` throws a fatal `TargetError`.
   The original `try` only wrapped `WinGetMinMax/WinRestore/WinShow`, **not** the
   `WinActivate` — so the exception aborted the whole script.

Because the AHK process exited non-zero, `run_ecadstar_batch()` would report
failure and `_simulate_peb()` would raise `SystemExit`, stopping the multi-phase
pipeline even though ECADStar was simulating fine.

### Fix
`ActivatePiemi()` now:
- re-resolves `hwnd := WinExist(WinTitleMatch)` inside the retry loop (handles a
  recreated window),
- guards `WinActivate` in `try/catch`; a vanished handle is a soft miss → `continue`,
- still confirms foreground via `WinWaitActive(WinTitleMatch, , timeoutSec)`.

The batch-start signal is the PI-1 `log.txt` content (checked in
`WaitBatchSimulationStart`), which does not depend on activation succeeding — so
soft-failing activation lets the loop keep polling and return success once the log
confirms the batch, then AHK exits 0 and the pipeline continues.

`SafeActivate()` applies the same guard to the dialog/window activations that were
previously bare `WinActivate` calls, preventing the same class of `TargetError`
from aborting a long (25-phase) run.

## 🛠️ Verification & Execution Results

- Static review: all `WinActivate` call sites are now guarded — grep shows only
  two bare `WinActivate` statements, both inside `try` blocks
  (`SafeActivate` body and `ActivatePiemi`'s guarded activate). All others are
  `SafeActivate(...)`.
- AHK v2 syntax reviewed (`#Requires AutoHotkey v2.0`): `try { } catch { }` blocks
  and the retry `Loop 3` are valid.
- Runtime execution requires the Windows/ECADStar host (AutoHotkey64.exe) and could
  not be run from WSL. Expected behavior on next run: the "Target window not found"
  abort no longer occurs; if the window flips during batch start, activation retries
  quietly and the run proceeds to output-move and the next MHz phase.

### What to watch on the next real run
- `%TEMP%\ecadstar_piemi_batch.log` should show `Batch simulation started — <peb>
  confirmed in PI-1/log.txt` instead of ending on the WinActivate traceback.
- The Python side (`wait_for_pi_outputs`) continues to gate on filesystem outputs,
  unchanged.
