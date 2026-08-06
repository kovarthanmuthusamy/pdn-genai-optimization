---
title: eval_spatial_metrics
type: code
path: experiments/exp048/codes/eval_spatial_metrics.py
group: experiments/exp048/codes
experiment: exp048
loc: 182
tags: [code, exp048]
---

# eval_spatial_metrics

> Off-anchor eval with spatial metrics (Pearson r, peak location) + early-stop hook.

**Source:** `experiments/exp048/codes/eval_spatial_metrics.py` · 182 lines
**Experiment:** [[exp048]]

## Functions

- **`_fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float, margin: float=0.5)`**
- **`run_off_anchor_eval_spatial(model: torch.nn.Module, val_loader, *, bg: float, off_anchor_mhz: tuple[float, ...]=(100.0, 175.0, 330.0, 400.0), max_batches: int=32, device: str | torch.device='cuda', out_csv: Path | str | None=None, ckpt_dir: Path | str | None=None, early_stop_state: dict | None=None, early_stop_mhz: float=330.0, early_stop_patience: int=3, early_stop_min_epoch: int=150)`** — Encode path @ off-anchor MHz with spatial metrics per MHz/kind.
- **`_epoch_from_csv_path(out_csv: Path | str)`**
- **`_maybe_early_stop(rows: list[dict], state: dict, *, epoch: int, mhz: float, patience: int, min_epoch: int=150, ckpt_dir: Path | str | None, model: torch.nn.Module)`**

## Imports

- [[experiments.exp048.codes.spatial_metrics]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp048.codes.eval_train_vs_val]]
- [[experiments.exp048.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
