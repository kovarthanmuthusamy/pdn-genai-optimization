---
title: gmax_training_patch
type: code
path: experiments/exp043/codes/gmax_training_patch.py
group: experiments/exp043/codes
experiment: exp043
loc: 185
tags: [code, exp043]
---

# gmax_training_patch

> Monkey-patch exp038 trainer for global-max heatmap normalization (exp043).

**Source:** `experiments/exp043/codes/gmax_training_patch.py` · 185 lines
**Experiment:** [[exp043]]

## Purpose

```text
Monkey-patch exp038 trainer for global-max heatmap normalization (exp043).

Run:
    Called automatically from ``train_vae_simple._on_stats_loaded`` — not run directly.
```

## Functions

- **`load_heatmap_stats(data_dir: str | Path)`**
- **`apply_gmax_config(c, raw: dict[str, Any] | None=None)`** — If dataset uses global-max heatmaps, set clip/bg/phys fields on ``c``. Returns True if applied.
- **`_prepare_batch_gmax(batch: dict, c, orig_prepare)`** — Synthetic blend + disk→train remap + FG mask in train space.
- **`transform_disk_heatmap(hm: torch.Tensor, c)`** — Linear on-disk heatmap batch → train space (for cross-freq alt / eval).
- **`patch_trainer(_tr, c)`** — Monkey-patch exp038 trainer losses when ``c`` uses global-max heatmaps.

## Imports

- [[experiments.exp043.codes.synthetic_freq_blend]]
- [[gmax_heatmap_loss]]
- [[heatmap_gmax_norm]]
- [[pi_freq_utils]]

## Imported by

- [[_audit_decap_locality]]
- [[_audit_physics]]
- [[_bench_physics_overhead]]
- [[_bench_train_step]]
- [[_breakdown_hm_loss]]
- [[_breakdown_hm_real]]
- [[check_log1p_train_space]]
- [[diagnose_sweep_vs_dataset]]
- [[eval_cross_freq_gmax]]
- [[experiments.exp043.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
