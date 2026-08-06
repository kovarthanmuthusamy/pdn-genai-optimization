---
title: ecadstar_persistence_and_locking
type: concept
source: docs/ecadstar_persistence_and_locking.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/ecadstar_persistence_and_locking.md` — edit the source file, then re-run `tools/build_vault.py`.

# ECADStar auto-script: persistence, locking, and background operation

> **SUPERSEDED (2026-08-04).** This document describes the old AutoHotkey GUI
> automation, which has been **removed entirely**. ECADStar is now driven by the
> native headless CLI (`engineer.exe --batch --batch-auto-exit`), which needs no
> GUI focus, no persistent instance, and releases its own lock on exit — so the
> persistence/locking/foreground concerns below no longer apply. See
> [[ecadstar_headless_cli|`ecadstar_headless_cli.md`]]. Kept for historical
> context only.

How to run the PI/EMI batch automation across many phases (impedance + 24 MHz)
without lock corruption, and what "background / not focused" can and cannot do.

## 📝 Summary of Changes

- `pipelines/dataset_sim/run_multitype_sim_pipeline.py` — `_simulate_peb()` now
  clears the `.rlk` lock **only on phases that (re)open the ERF** (`not
  skip_open_erf`), never on `skipopen` phases where a live instance holds it.
- `tools/ecadstar/ecadstar_piemi_batch.ahk` — Main clears the lock only when
  opening fresh: `if (!skipOpenErf || !PiemiExists()) RemoveDesignLock(...)`.
- `pipelines/dataset_sim/ecadstar.py` — added `kill_ecadstar_windows()`: finds the
  "eCADSTAR PI/EMI Analysis" window by title and force-closes it (same as the
  manual recovery kill).
- `pipelines/dataset_sim/run_multitype_sim_pipeline.py` — new flag
  `KILL_STALE_ECADSTAR_ON_FRESH_OPEN = True`; `_simulate_peb()` kills any stale
  instance **only on a fresh open** (never on `skipopen`), then clears the orphan
  lock.
- `tools/ecadstar/kill_ecadstar.ps1` — standalone manual abort helper.

## 🔪 Auto-kill on fresh open (why it's safe)

`kill_ecadstar_windows()` runs the same PowerShell the manual recovery used:
match `*eCADSTAR*PI/EMI*` / `*PI/EMI Analysis*` by `MainWindowTitle`, then
`Stop-Process -Force`. In the pipeline it fires **only when `not skip_open_erf`**,
i.e. the impedance phase and the first distribution MHz — the phases that were
going to (re)open the ERF anyway. It never runs on `skipopen` phases, so the
persistent single-instance model is preserved:

| Phase | skipopen? | kill stale? | clear lock? |
|-------|-----------|-------------|-------------|
| Every phase (fresh-open mode) | no | yes | yes |
| Manual abort | — | `python -c "from pipelines.dataset_sim.ecadstar import kill_ecadstar_windows as k; k()"` or `kill_ecadstar.ps1` | — |

## 🔁 Active mode: fresh analysis window EVERY phase (self-healing)

`ECADSTAR_SKIP_OPEN_ERF_AFTER_FIRST = False` → each phase (impedance + every MHz)
opens a **new** analysis window. Combined with
`KILL_STALE_ECADSTAR_ON_FRESH_OPEN = True`, each phase does:

    kill any existing PI/EMI window → clear (orphan) lock → open new analysis
    window (shell open) → Load Batch → wait for outputs → move → next phase

Why this mode: it is **self-healing**. If you accidentally close/exit the
simulation window (or it crashes), the next phase does not depend on a live
instance — it just kills leftovers and opens a fresh window, so the sweep keeps
up on its own. Trade-off vs. the persistent single-instance model: each phase
pays the design-reload cost at open (seconds per phase — "Clean design data",
classification read, "PI/EMI ready"), negligible next to the per-phase
simulation time. Verified: with this flag every phase resolves to
`skip_open=False` (fresh open + kill).

Live test (ECADStar not running): helper printed "No running ECADStar PI/EMI
instance to kill." and returned `[]`; pipeline compiles.

## 🚀 The correct persistence model (keep ONE instance open)

The pipeline is already built to keep a single PI/EMI instance open for the whole
sweep:

- `ECADSTAR_SKIP_OPEN_ERF_AFTER_FIRST = True` → the **first** distribution phase
  re-opens `H-shape.erf`; every later MHz uses `skipopen` (Load Batch only, same
  instance). Impedance (if enabled) is the first phase.
- So between simulations we do **not** close/reopen the design. "Load Batch" is
  issued into the already-open window.

**Do not force-close the window between phases.** ECADStar writes
`<design>.rlk` while a design is open and removes it on a clean close. A
force-close/kill leaves an orphan `.rlk` → next open shows the "Remove lock?"
dialog. Keeping the instance open avoids the lock cycle entirely.

### Lock handling now (after the fix)
| Phase | Opens ERF? | Clears `.rlk`? |
|-------|-----------|----------------|
| Impedance / first MHz | yes (fresh) | yes (safe — nothing holds it yet) |
| Later MHz (`skipopen`) | no (reuse) | **no** (live instance holds it) |
| `skipopen` but window died | yes (shell fallback) | yes (orphan → correct) |

Two safety layers remain for genuine orphans: Python `clear_ecadstar_lock()` on
fresh opens, AHK `RemoveDesignLock()` + `DismissRemoveLockDialog()` (auto Alt+Y).

### If you must close between runs
Close via ECADStar's own File→Close/Exit (clean removal of `.rlk`). Never kill the
process — that is what creates the orphan lock.

## 🖥️ "Run in background even if we're not in that window"

This is the hard constraint. The script drives the GUI with `WinActivate` + `Send`
(and `Click`) — synthetic input that **requires the Windows session desktop to be
active/composited**. Consequences:

- A **minimized** PI/EMI window or a **disconnected/locked RDP session** cannot
  receive these keystrokes/menu clicks reliably. That is why the script logs
  "keep RDP visible/focused" everywhere.
- You can run it while *you* are doing other things **on the same visible
  desktop** (the script foregrounds PI/EMI itself), but you cannot minimize the
  session or disconnect RDP and expect menu automation to work.

### Options to make it truly unattended
1. **Keep the session's console active (recommended).** Configure the Windows
   host so the session stays composited when you disconnect:
   - Use `tscon <sessionId> /dest:console` to redirect your RDP session to the
     physical console on disconnect (session stays "logged in and active"), or
   - Set the host not to lock on disconnect (group policy / registry), or
   - Run on a VM/host whose **console** session stays logged in.
   With an active console session, AHK input works while you are "not looking".
2. **Prefer `ControlSend` / `ControlClick` over `WinActivate`+`Send`.** These
   target a specific control by handle and can deliver input to a background
   (non-minimized) window without stealing foreground. Menu navigation
   (Tools→Load Batch) is the tricky part and may still need the window
   non-minimized. This is a larger refactor of `LoadBatch()` /
   `TryInvokeLoadBatchMenu()` and should be validated on the real host.
3. **Supervisor/watchdog.** A long-lived AHK (using `SetTimer`) that (re)launches
   PI/EMI if missing, auto-dismisses lock/modal dialogs, and retries Load Batch,
   instead of one one-shot process per phase.

The Python side is already background-friendly: it never depends on GUI focus —
`wait_for_batch_started()` and `wait_for_pi_outputs()` gate purely on
`PI-*/log.txt` and output `.map`/`.csv` files on disk.

## 🛠️ Verification & Execution Results

- Static review of the lock-clear gating:
  - Python: `if ECADSTAR_CLEAR_LOCK_FILE and not skip_open_erf:` — `skip_open_erf`
    is resolved from config just above, so impedance/first-MHz clear, later MHz
    do not.
  - AHK: `if (!skipOpenErf || !PiemiExists()) RemoveDesignLock(...)` — fresh opens
    and dead-instance fallback clear; live `skipopen` does not.
- Runtime requires the Windows/ECADStar host; not runnable from WSL. Expected
  effect: no "Remove lock?" churn between MHz phases, and no accidental deletion
  of the live instance's lock.

### Recommended config for a persistent sweep
- `ECADSTAR_SKIP_OPEN_ERF_AFTER_FIRST = True` (keep one instance open).
- `RESUME_FROM_PROGRESS = True` (survive interruptions; resume by MHz).
- Keep an **active console session** (see options above) for unattended runs.
