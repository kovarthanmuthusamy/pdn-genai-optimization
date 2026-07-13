# K_VALUE Val Layout Fix

### 📝 Summary of Changes

- **Root cause**: `layout_qc` mode loads real validation layouts per K. `data_multi_norm_unbounded` has **no K=1** samples (minimum K is **2**). Pipeline `K_VALUE` included `1`, causing `SweepQCError`.
- **Pipeline config**: `K_VALUE` in `run_multifreq_sweep_pipeline.py` now starts at **2** instead of **1**.
- **Auto-filter**: Added `filter_k_for_layout_qc()` in `sweep_latent_opt_rules.py` — drops K not present in the val split with a clear warning instead of failing mid-run.
- **Bugfix**: Added missing `resolve_repo_path` import in `multifreq_layout_store.py`.

### 🚀 Implementation Details

- `k_values_in_val_loader(val_ld)` scans the val `DataLoader` once and returns the set of K values present.
- `filter_k_for_layout_qc(k_values, val_ld)` is called at the start of `run_generate()` when `INFERENCE_MODE == "layout_qc"`, before loading layouts.
- `k_output_tag` / sweep print now use the **filtered** K list so logs match what is actually generated.

### 🛠️ Verification & Execution Results

- Dataset K cache: K range **2–50**, **0** samples at K=1.
- `filter_k_for_layout_qc(range(1,21), val_ld)` → drops `[1]`, keeps K **2–20** (19 values).
- `load_val_layout_samples(k_value=2)` → success: `occ [1,52]`, `imp [1,1,231]`, `K=2`.

Re-run the pipeline:

```bash
python scrap/orchestration/run_multifreq_sweep_pipeline.py
```
