# Script refinement (agent-friendly entry points)

## Modular folder layout

Scripts live under **`pipelines/`** and **`libs/`** only. Legacy shims were **removed** — use canonical paths (see [`pipelines/README.md`](../pipelines/README.md)).

---

Refactored runnable scripts for **less duplication**, **clear module order**, and **unchanged behavior** where noted.

## Shared modules added

| Module | Role |
|--------|------|
| `repo_paths.py` | `REPO_ROOT`, `setup_path()`, `repo_path(*parts)` |
| `New_heatmaps/peb_frequency.py` | Shared PEB `EditPIDistribution` frequency regex |
| `scrap/comparison/batch_over_k.py` | K-loop runner for compare scripts |
| `Latent_opm/_optimization_loader.py` | Lazy load `latent_optimization_impedance` + `resolve_run_dir()` |

## Scripts slimmed / reorganized

### `scripts/`
- `impedance_visuals.py` — fixed tuple bug (`np.load(...),`); renamed `plot_impedance_profile`; uses `plt.close()`
- `check_mask_shape.py` — compact `main()`, `repo_path`
- `precompute_decap_profiles.py` — 138 → ~55 lines; same RLC math
- `compute_latent_stats.py` — loop over modalities; same output JSON
- `dis_con.py` — `main()` + `process_channels()`; pathlib glob
- `quick_traversal_test.py` — fixed repo root; config-driven traversal values

### `New_heatmaps/`
- `change_frequency.py` — delegates to `peb_frequency.write_peb_at_mhz`
- `regenerate_mhz_pebs.py` — same shared module

### `scrap/comparison/`
- `compare_generated_vs_real_all_k.py` — 74 → ~35 lines via `batch_over_k`
- `compare_generated_vs_real_occupancy_all_k.py` — same

### `Latent_opm/`
- `latent_run_export_peb.py` — removed duplicate importlib; ~90 → ~45 lines
- `latent_run_compare_report.py` — shared loader; clearer mode flow

### `datasets/`
- `remove_freq_from_multifreq.py`, `build_multifreq_gmax_dataset.py` — `repo_paths` bootstrap

### Other
- `Data_Creation/verify_layout_store.py` — tighter verify/prune loop
- `active_learning_pi/run_pipeline.py` — trimmed comments; same CONFIG API

## Conventions (unchanged)

1. Module docstring (Purpose, Run, Agent notes, Config keys)
2. Imports → CONFIG block → helpers → `main()` → `if __name__`
3. Run via `python path/script.py` — edit the CONFIG block first; entry scripts do not take flags.

## Not refactored in this pass (large / separate effort)

These remain functionally correct but are candidates for a follow-up:

- `scrap/comparison/move_and_compare.py` (~1300 lines) — plot dedup with `compare.py`
- `Data_Creation/Data_processing*.py` — shared pipeline core
- `Latent_opm/latent_optimization_impedance.py` — split losses/runner
- `scripts/latent_traversal.py` — needs modern checkpoint paths + possible merge with quick test

## Verify

```bash
python3 -m py_compile repo_paths.py New_heatmaps/peb_frequency.py scripts/*.py
```
