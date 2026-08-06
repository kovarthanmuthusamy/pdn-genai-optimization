---
title: sweep_qc_eval
type: code
path: scrap/generation/sweep_qc_eval.py
group: scrap/generation
loc: 600
tags: [code, scrap]
---

# sweep_qc_eval

> Lightweight sweep QC metrics + agent-copy report.

**Source:** `scrap/generation/sweep_qc_eval.py` · 600 lines

## Purpose

```text
Lightweight sweep QC metrics + agent-copy report.

Runs after ``run_generate`` — only metrics needed to judge whether to proceed training.
```

## Functions

- **`_spatial_metrics(experiment_dir: str)`**
- **`_fg_mse(recon: torch.Tensor, target: torch.Tensor, fg_thr: float)`**
- **`clip_ceiling_ohm(engine: Any, mhz: float | None=None)`** — Physical Ω reference ceiling for QC (p99.5 soft ref when unbounded).
- **`_is_near_clip(gen_max: float, ceiling: float | None, *, frac: float=0.9)`**
- **`_sort_gen_rows(rows: list[dict[str, Any]])`**
- **`_k_label(k_values: list[int])`**
- **`_multi_k(k_values: list[int])`**
- **`_fmt_metric(value: Any, *, width: int=7, prec: int=2)`**
- **`_fmt_metric_plain(value: Any, *, prec: int=2)`**
- **`_row_k_prefix(r: dict[str, Any], multi_k: bool)`**
- **`eval_layout_vs_real_anchors(engine: Any, device: torch.device, *, data_dir: Path, experiment_cfg: dict[str, Any], experiment_dir: str, k_value: int, anchor_mhz_list: list[float], num_samples: int, seed: int, fg_thr: float, mhz_tol: float=2.0, val_ld: DataLoader | None=None)`** — Layout path vs real val heatmaps at training-anchor MHz only.
- **`_flags(gen_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]], engine: Any, *, multi_k: bool=False)`**
- **`format_agent_copy_block(*, experiment_dir: str, checkpoint_path: str, qc_cfg: SweepQCConfig, gen_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]], flags: list[str], engine: Any, k_values: list[int])`** — Single block to paste into an agent chat.
- **`write_sweep_qc_report(out_root: Path, *, experiment_dir: str, checkpoint_path: str, qc_cfg: SweepQCConfig, gen_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]], engine: Any, k_values: list[int], print_copy_block: bool=True)`** — Write ``sweep_qc_report.md``, ``sweep_qc_metrics.json``, optionally print copy block.
- **`_anchors_in_sweep(sweep_mhz: list[float])`**
- **`_collect_anchor_rows(engine: Any, device: torch.device, *, data_dir: Path, experiment_cfg: dict[str, Any], experiment_dir: str, k_values: list[int], anchor_mhz_list: list[float], qc_cfg: SweepQCConfig, val_ld: DataLoader | None)`** — Anchor GT eval for each K — one progress line per K, no per-K report spam.
- **`run_sweep_qc_eval(engine: Any, device: torch.device, *, out_root: Path, experiment_dir: str, checkpoint_path: str, data_dir: Path, experiment_cfg: dict[str, Any], qc_cfg: SweepQCConfig, gen_rows: list[dict[str, Any]], sweep_mhz: list[float], k_values: list[int] | None=None, val_ld: DataLoader | None=None)`** — Run anchor GT eval (layout_qc only) and write one combined agent report.

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp050.codes.__init__]]
- [[experiments.exp050.codes.spatial_metrics]]
- [[heatmap_z_clip]]
- [[pi_freq_utils]]
- [[sweep_latent_opt_rules]]

## Imported by

- [[run_multifreq_heatmap_sweep]]

## External dependencies

`experiments`, `importlib`, `numpy`, `src_vae`, `torch`
