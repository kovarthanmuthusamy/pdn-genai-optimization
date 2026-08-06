---
title: diagnose_sweep_vs_dataset
type: code
path: scratch/diagnose_sweep_vs_dataset.py
group: scratch
loc: 568
tags: [code, scratch]
---

# diagnose_sweep_vs_dataset

> Compare sweep inference vs dataset-ground-truth paths (encode / layout).

**Source:** `scratch/diagnose_sweep_vs_dataset.py` · 568 lines

## Purpose

```text
Compare sweep inference vs dataset-ground-truth paths (encode / layout).

Isolates why multifreq sweep compare plots look "all blue" while training
metrics look good: sweep uses marginal z -> generated occ/imp -> layout-z decode;
dataset eval uses true occ/imp (or full encode).

Run:
    .venv/bin/python scratch/diagnose_sweep_vs_dataset.py

Outputs:
    scratch/sweep_vs_dataset_report.json
    scratch/sweep_vs_dataset_summary.txt
    scratch/sweep_vs_dataset_plots/sample_*.png  (optional quick viz)
```

## Constants

| Name | Value |
|------|-------|
| `EXPERIMENT_DIR` | `REPO_ROOT / 'experiments/exp043'` |
| `CHECKPOINT_PATH` | `REPO_ROOT / 'experiments/exp043/runs/run_20260616T165326Z/checkpoints/last_model.pt'` |
| `DATA_DIR` | `REPO_ROOT / 'datasets/data_multifreq_gmax'` |
| `OUT_JSON` | `REPO_ROOT / 'scratch/sweep_vs_dataset_report.json'` |
| `OUT_TXT` | `REPO_ROOT / 'scratch/sweep_vs_dataset_summary.txt'` |
| `OUT_PLOT_DIR` | `REPO_ROOT / 'scratch/sweep_vs_dataset_plots'` |
| `K_VALUE` | `30` |
| `TARGET_MHZ` | `300.0` |
| `PI_REF_MHZ` | `200.0` |
| `SHARED_TEMP` | `1.5` |
| `SEED` | `42` |
| `NUM_DATASET_SAMPLES` | `5` |
| `MHZ_TOL` | `2.0` |
| `SAVE_PLOTS` | `True` |
| `SWEEP_NPY` | `PROJECT_ROOT / 'experiments/exp043/multifreq_heatmap_sweep_30/freq_300MHz/K30/data_sample…` |

## Classes

- **`PathMetrics`**

## Functions

- **`_config_ns()`**
- **`_batch_to_device(batch: dict, device: torch.device)`**
- **`_pi_mhz_from_norm(pi_norm: float)`** — Invert log10 norm → MHz (approx for filtering).
- **`_find_val_samples(val_loader, *, k: int, mhz: float, max_n: int, mhz_tol: float)`** — Collect up to max_n val batches items matching K and MHz.
- **`_fg_mask(hm_train: torch.Tensor, fg_thr: float)`**
- **`_fg_mse(recon: torch.Tensor, target: torch.Tensor, fg_thr: float)`**
- **`_to_phys(hm_train: torch.Tensor, engine: VAEInference)`**
- **`_gt_phys_from_disk(hm_lin: torch.Tensor, gmax: float)`**
- **`_phys_stats(phys: np.ndarray, mask: np.ndarray)`**
- **`_pearson_r(a: np.ndarray, b: np.ndarray, mask: np.ndarray)`**
- **`_occ_match_frac(gen_occ: torch.Tensor, gt_occ: torch.Tensor, k: int)`** — Fraction of top-K decap indices that match between gen and GT.
- **`_imp_mse_log(gen_imp: torch.Tensor, gt_imp: torch.Tensor, log_mean: float, log_std: float)`**
- **`_run_path(engine: VAEInference, model, *, path: str, occ: torch.Tensor, imp: torch.Tensor, hm_gt_train: torch.Tensor | None, k: int, mhz: float, device: torch.device, fg_thr: float, mask: np.ndarray, gt_phys: np.ndarray, gt_occ: torch.Tensor | None=None, gt_imp: torch.Tensor | None=None)`**
- **`_metrics_from_npy(npy_path: Path, gt_phys: np.ndarray, mask: np.ndarray)`**
- **`_save_plot(gt_phys: np.ndarray, preds: dict[str, np.ndarray], mask: np.ndarray, out_path: Path, title: str)`**
- **`_format_row(m: PathMetrics)`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp043.codes.inference_vae]]
- [[gmax_training_patch]]
- [[heatmap_gmax_norm]]
- [[pi_freq_utils]]
- [[repo_paths]]

## External dependencies

`experiments`, `matplotlib`, `numpy`, `repo_paths`, `src_vae`, `torch`, `types`
