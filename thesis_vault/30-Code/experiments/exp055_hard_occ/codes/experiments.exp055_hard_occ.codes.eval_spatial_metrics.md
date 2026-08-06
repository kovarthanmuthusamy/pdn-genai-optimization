---
title: eval_spatial_metrics
type: code
path: experiments/exp055_hard_occ/codes/eval_spatial_metrics.py
group: experiments/exp055_hard_occ/codes
experiment: exp055_hard_occ
loc: 142
tags: [code, exp055_hard_occ]
---

# eval_spatial_metrics

> Off-anchor eval — spatial metrics, append rows to one CSV (epoch column).

**Source:** `experiments/exp055_hard_occ/codes/eval_spatial_metrics.py` · 142 lines
**Experiment:** [[exp055_hard_occ]]

## Constants

| Name | Value |
|------|-------|
| `OFF_ANCHOR_CSV_NAME` | `'off_anchor_eval.csv'` |
| `CSV_FIELDS` | `('epoch', 'mhz', 'kind', 'n', 'hm_fg_mse_mean', 'pearson_fg_mean', 'peak_loc_err_mean')` |

## Functions

- **`_fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float, margin: float=0.5)`**
- **`epoch_from_off_anchor_path(out_csv: Path | str | None)`**
- **`append_off_anchor_rows(path: Path, epoch: int, rows: list[dict])`**
- **`run_off_anchor_eval_spatial(model: torch.nn.Module, val_loader, *, bg: float, off_anchor_mhz: tuple[float, ...]=(100.0, 270.0, 400.0), max_batches: int=12, device: str | torch.device='cuda', out_csv: Path | str | None=None, epoch: int=0, use_encode_skips: bool=False, use_binary_occupancy: bool=True)`**

## Imports

- [[experiments.exp055_hard_occ.codes.occupancy_binary]]
- [[experiments.exp055_hard_occ.codes.spatial_metrics]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp055_hard_occ.codes.eval_off_anchor]]

## External dependencies

`src_vae`, `torch`
