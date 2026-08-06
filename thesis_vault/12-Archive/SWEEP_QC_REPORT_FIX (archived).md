---
title: SWEEP_QC_REPORT_FIX (archived)
type: archive
source: docs/_archive/SWEEP_QC_REPORT_FIX.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# Sweep QC report deduplication

### 📝 Summary of Changes

- Fixed duplicate `=== SWEEP QC (copy to agent) ===` output when sweeping multiple K values (e.g. K=7–17 printed 11 full reports).
- Consolidated QC into **one** combined report with a **K column** for multi-K sweeps.
- Anchor GT eval still runs per K, but prints one progress line per K instead of a full copy block each time.
- Safe metric formatting (`n/a` instead of crash/truncation on non-finite values).
- Set `QC_SWEEP = True` in orchestration when using QC inference modes.

### 🚀 Implementation Details

**Root cause:** `run_generate()` called `run_sweep_qc_eval()` inside `for k in k_values`, so each K rewrote `sweep_qc_report.md` / `sweep_qc_metrics.json` and printed the full agent copy block.

**Fix:**
- `run_sweep_qc_eval(..., k_values=[...])` collects anchor metrics for all K, then calls `write_sweep_qc_report()` once.
- `format_agent_copy_block` / markdown tables add a `K` column when `len(k_values) > 1`.
- `_collect_anchor_rows` logs compact per-K pearson summaries during anchor eval.

**Files changed:**
- `scrap/generation/sweep_qc_eval.py`
- `scrap/generation/run_multifreq_heatmap_sweep.py`
- `scrap/orchestration/run_multifreq_sweep_pipeline.py`

### 🛠️ Verification & Execution Results

```text
python -m py_compile scrap/generation/sweep_qc_eval.py scrap/generation/run_multifreq_heatmap_sweep.py
# OK

Smoke test: multi-K format_agent_copy_block → single block, K column present
# OK: single consolidated block with K column
```

After re-running the pipeline, expect **one** copy block at the end of STEP 1, with rows like:

```text
  K | MHz | anchor | gen_max | ...
  7 |     10 | yes    |    0.29 | ...
 13 |    270 | yes    |    5.13 | ...
```
