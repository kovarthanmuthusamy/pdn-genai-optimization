---
title: sim-pipeline-auto-append (archived)
type: archive
source: docs/_archive/sim-pipeline-auto-append.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# Sim pipeline: CLI append + ECADStar Load Batch fix (490 MHz)

## Summary of Changes

- **Append**: already uses CLI (`--mhz`, `--append-tag`, `--raw-root`); trigger passes args only — no script patching.
- **`ECADSTAR_SKIP_OPEN_ERF_AFTER_FIRST = True`**: after the first distribution MHz, AHK uses `skipopen` (Load Batch only) instead of Ctrl+O re-open every MHz.
- **AHK Load Batch**: retries Tools menu with 4 methods (mouse, Alt+T variants) before failing.

## Implementation Details

### Append trigger

```bash
python pipelines/data/append_merged_combinations_multifreq.py --mhz 490 --append-tag merged_490 --raw-root "C:\Users\...\Raw"
```

Sim pipeline fires this via `run_append_locked.py` in the background (flock serializes overlapping appends).

### 490 MHz failure root cause

Log showed `skipopen: no` → PI/EMI already running → **Ctrl+O re-open .erf** → Tools click → **Load Batch dialog never opened**.

Re-opening .erf every MHz (170→590) is fragile after long runs. With `ECADSTAR_SKIP_OPEN_ERF_AFTER_FIRST`:

| Step | skipopen |
|------|----------|
| First pending MHz (e.g. 170) | no — open .erf |
| All later MHz (180, …, 490) | yes — Load Batch only |
| Resume at 490 (170–470 done) | yes |

### Recovery for 490 MHz

1. Keep **RDP visible/focused** on PI/EMI.
2. Confirm H-shape.erf is loaded; dismiss any modal dialogs.
3. Re-run sim pipeline (`RESUME_FROM_PROGRESS=True` skips completed MHz):

```bash
python pipelines/dataset_sim/run_combinations_sim_pipeline.py
```

Expected log for 490 MHz:

```text
ECADStar mode: skipopen (Load Batch only)
skipopen: yes
Load Batch menu attempt: mouse_shifttab
```

## Verification & Execution Results

- `append_merged_combinations_multifreq.py --help` — CLI args present.
- Pipeline `_skip_open_erf_for_distribution(done_mhz={170,...470}, pending_index=0)` → `True` (skipopen on resume).
- AHK `TryInvokeLoadBatchMenu` — 4 fallback methods added.
