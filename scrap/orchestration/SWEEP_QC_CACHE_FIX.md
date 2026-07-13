# Sweep QC — shared val loader + ceiling fix

## Problem

`run_multifreq_sweep_pipeline.py` (generate step) rebuilt the full 311k-sample RAM cache **6 times**:

1. `load_val_layout_samples()` — pick val layouts for `layout_qc`
2–6. `eval_layout_vs_real_anchors()` — one `load_val_anchor_samples()` per training anchor in the sweep (e.g. 10, 63, 300, 330, 400 MHz)

Each call created a new `VAEDataset(cache_in_ram=True)` (~85–400s each).

A separate bug crashed QC report writing: `NameError: ceiling` in `write_sweep_qc_report`.

## Fix

### Shared val `DataLoader`

**`scrap/generation/sweep_latent_opt_rules.py`**

- `get_sweep_val_loader()` — builds val loader once per `(data_dir, split config)` and caches it module-wide.
- `load_val_layout_samples(..., val_ld=None)` and `load_val_anchor_samples(..., val_ld=None)` accept an optional pre-built loader.

**`scrap/generation/run_multifreq_heatmap_sweep.py`**

- `run_generate()` creates `sweep_val_ld` once for `layout_qc` and passes it to layout load + `run_sweep_qc_eval`.

**`scrap/generation/sweep_qc_eval.py`**

- `eval_layout_vs_real_anchors(..., val_ld=None)` and `run_sweep_qc_eval(..., val_ld=None)` reuse the same loader across all anchor MHz.

**Expected runtime:** one RAM cache build per pipeline run (~90s with `cache_in_ram: true`), not per anchor.

### Per-MHz ceiling in QC report

**`scrap/generation/sweep_qc_eval.py`**

- `write_sweep_qc_report` uses `clip_ceiling_ohm(engine, mhz=...)` per row instead of undefined global `ceiling`.

## Re-run

```bash
cd /home/ubuntu/genai_pdn
python scrap/orchestration/run_multifreq_sweep_pipeline.py
```

Or generate-only (skip ECADStar):

Set in `run_multifreq_sweep_pipeline.py`: `SKIP_SIMULATE = True`, then run as above.

## Optional: skip RAM cache entirely for sweeps

In `experiments/exp048/config.yaml`, set `"cache_in_ram": false` if you prefer mmap-on-demand (~slower per batch, no ~90s upfront load). Training can keep `true`.
