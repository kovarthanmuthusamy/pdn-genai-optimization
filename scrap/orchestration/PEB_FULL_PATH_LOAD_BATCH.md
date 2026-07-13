# Load Batch: full PEB path paste

## Change

ECADStar Load Batch automation now pastes the **full Windows path** to the `.peb` file (e.g. `C:\Users\...\pi_distribution_K30_freq_sweep.peb`) instead of the filename only.

## Files updated

| File | Change |
|------|--------|
| `tools/ecadstar/ecadstar_piemi_batch.ahk` | `UsePebFileNameOnly := false` — clipboard gets full `pebPath` |
| `scrap/orchestration/run_multifreq_sweep_pipeline.py` | Log messages and docstrings aligned with full-path behavior |

## Flow

1. Pipeline stages the generated PEB to `PEB_COPY_DEST` (if set) and beside the `.erf`.
2. `run_ecadstar_batch()` passes `-PebPath` with the Windows path to `run_ecadstar_piemi_batch.ps1`.
3. AutoHotkey `LoadBatch()` pastes that full path into the Load Batch file dialog.

## Verify

After running the simulate step, check `%TEMP%\ecadstar_piemi_batch.log` on Windows. You should see a line like:

```
Load Batch started — typed: C:\Users\...\pi_distribution_K30_freq_sweep.peb (full path: ...)
```

## Revert to filename-only

If your ECADStar dialog only accepts a basename when the `.peb` sits next to the `.erf`, set in `ecadstar_piemi_batch.ahk`:

```ahk
UsePebFileNameOnly := true
```
