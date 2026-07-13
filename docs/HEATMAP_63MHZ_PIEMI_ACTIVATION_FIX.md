# heatmap_63MHz PI/EMI activation failure — diagnosis & fix

## 📝 Summary of Changes

- Diagnosed **63 MHz failure** as RDP foreground / window-activation issue (not a 63 MHz–specific PEB bug).
- Hardened `tools/ecadstar/ecadstar_piemi_batch.ahk`:
  - Separate **window exists** vs **can foreground**
  - Avoid shell-opening a second PI/EMI when the window already exists
  - Restore minimized PI/EMI, Alt-key foreground trick, retries before Load Batch
  - Clearer log messages when RDP is minimized
- Updated `pipelines/dataset_sim/run_combinations_sim_pipeline.py`:
  - `ECADSTAR_SKIP_OPEN_ERF_AFTER_FIRST = True` — later MHz phases use `skipopen` (Load Batch only)

## 🚀 Implementation Details

### What the log shows

```
PI/EMI not running — open .erf via shell
PI/EMI window appeared after shell open
WaitAnalysisReady … UI wait timeout after 12000ms — continuing anyway
ERROR: Cannot activate PI/EMI for Load Batch
```

Sequence:

1. **10 MHz completed** — pipeline moved to `combinations_dist_63MHz.peb`.
2. AutoHotkey could not **foreground** the PI/EMI window (`WinWaitActive` failed).
3. Script treated that as “PI/EMI not running” and launched `.erf` via shell again.
4. Window existed but stayed **not active** (typical when **RDP is minimized** and you work on your local PC).
5. `LoadBatch()` failed because GUI automation requires the RDP desktop in the foreground.

Windows blocks `SetForegroundWindow` from background/minimized sessions. This is environmental, not a 63 MHz simulation issue.

### Immediate recovery (on Windows RDP)

1. **Restore and focus the RDP window** (full screen or at least visible — do not minimize during automation).
2. Open PI/EMI manually if needed; confirm **H-shape.erf** is loaded and no modal dialog is blocking (lock file, save, etc.).
3. Re-run:
   ```bash
   python pipelines/dataset_sim/run_combinations_sim_pipeline.py
   ```
4. With `RESUME_FROM_PROGRESS = True`, **10 MHz is skipped**; it retries **63 MHz** only.

### Code behavior after fix

| Run | `skipopen` | Behavior |
|-----|------------|----------|
| First (impedance or 10 MHz) | `no` | Open `.erf` if needed |
| Later MHz (63, 80, …) | `yes` | Load Batch only — faster, fewer dialogs |
| RDP not foreground | — | AHK retries up to ~60s before failing; logs explicit RDP warning |

## 🛠️ Verification & Execution Results

- **AHK / pipeline changes**: applied in repo (WSL workspace).
- **Live ECADStar run**: not executed here — requires your Windows RDP host with PI/EMI + AutoHotkey v2.
- **Expected on re-run**: log shows `skipopen: yes` for 63 MHz; PI/EMI foreground succeeds with RDP visible; Load Batch pastes `combinations_dist_63MHz.peb`.

### Operational rule

Keep the **RDP session visible and focused** for the entire multi-day sweep. Minimizing RDP or switching to local desktop while AHK runs will reproduce this error on any MHz step.
