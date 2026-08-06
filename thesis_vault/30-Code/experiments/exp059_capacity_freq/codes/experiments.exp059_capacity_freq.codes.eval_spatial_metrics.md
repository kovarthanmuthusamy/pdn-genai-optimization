---
title: eval_spatial_metrics
type: code
path: experiments/exp059_capacity_freq/codes/eval_spatial_metrics.py
group: experiments/exp059_capacity_freq/codes
experiment: exp059_capacity_freq
loc: 143
tags: [code, exp059_capacity_freq, uncommitted]
---

# eval_spatial_metrics

> Off-anchor eval — spatial metrics, append rows to one CSV (epoch column).

**Source:** `experiments/exp059_capacity_freq/codes/eval_spatial_metrics.py` · 143 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp059_capacity_freq]]

## Constants

| Name | Value |
|------|-------|
| `OFF_ANCHOR_CSV_NAME` | `'off_anchor_eval.csv'` |
| `CSV_FIELDS` | `('epoch', 'mhz', 'kind', 'n', 'hm_fg_mse_mean', 'pearson_fg_mean', 'peak_loc_err_mean')` |

## Functions

- **`_fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float, margin: float=0.5)`**
- **`epoch_from_off_anchor_path(out_csv: Path | str | None)`**
- **`append_off_anchor_rows(path: Path, epoch: int, rows: list[dict])`**
- **`run_off_anchor_eval_spatial(model: torch.nn.Module, val_loader, *, bg: float, off_anchor_mhz: tuple[float, ...]=(100.0, 270.0, 400.0), max_batches: int=12, device: str | torch.device='cuda', out_csv: Path | str | None=None, epoch: int=0, use_binary_occupancy: bool=True, use_occ_only_layout: bool=False)`**

## Imports

- [[experiments.exp059_capacity_freq.codes.distributed_train]]
- [[experiments.exp059_capacity_freq.codes.occupancy_binary]]
- [[experiments.exp059_capacity_freq.codes.spatial_metrics]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp059_capacity_freq.codes.eval_off_anchor]]

## External dependencies

`src_vae`, `torch`
