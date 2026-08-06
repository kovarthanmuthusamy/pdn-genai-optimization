# Multifreq Sweep Pipeline — Multi-K Support

### 📝 Summary of Changes

- Extended `K_VALUE` in `scrap/orchestration/run_multifreq_sweep_pipeline.py` to accept **either a single int or a list of ints** (e.g. `30` or `[10, 20, 30]`).
- Updated `scrap/generation/run_multifreq_heatmap_sweep.py` to generate samples, PEB entries, and manifests for all listed K values in one run.
- Updated PI move/compare/report steps (`multifreq_move_and_compare.py`, `compare.py`, `build_comparison_report.py`) to route ECADStar outputs and build reports across multiple K folders.

### 🚀 Implementation Details

**Config (pipeline or generator):**

```python
K_VALUE = 30              # single K (unchanged behavior)
K_VALUE = [10, 20, 30]    # multiple K values in one run
```

**Output layout:**

- Folder: `experiments/exp050/multifreq_heatmap_sweep_10_20_30/` (tag = K values joined with `_`)
- Per-frequency paths: `freq_10MHz/K10/`, `freq_10MHz/K20/`, …
- Combined PEB: `pi_distribution_K10_20_30_freq_sweep.peb`

**PEB order** (matches ECADStar PI numbering):

```
for each MHz in SWEEP:
  for each K in K_VALUE:
    for each sample:
      one PI entry
```

**Helpers added** in `run_multifreq_heatmap_sweep.py`:

- `normalize_k_values()` — scalar or list → sorted unique K list
- `k_output_tag()` — folder/label suffix
- `peb_basename_for_k()` — combined PEB filename
- `exported_k_values()` — reads `k_values` from sweep manifest or falls back to config

Single-K runs remain backward compatible (same folder names, PEB names, and manifest shape with added `k_values` field).

### 🛠️ Verification & Execution Results

Terminal smoke tests passed:

- `normalize_k_values`, `k_output_tag`, `peb_basename_for_k` assertions
- Multi-K PI slot ordering (`_build_pi_slots`) — 2 freqs × 2 K × 2 samples = 8 slots
- Pipeline `_sync_output_root()` for `K_VALUE=30` and `K_VALUE=[10, 30]`
- `py_compile` on all modified modules — no errors

Full end-to-end generate + ECADStar simulate was not run in this session (requires checkpoint + Windows ECADStar).
