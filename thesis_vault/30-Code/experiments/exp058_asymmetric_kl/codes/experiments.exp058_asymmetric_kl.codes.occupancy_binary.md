---
title: occupancy_binary
type: code
path: experiments/exp058_asymmetric_kl/codes/occupancy_binary.py
group: experiments/exp058_asymmetric_kl/codes
experiment: exp058_asymmetric_kl
loc: 66
tags: [code, exp058_asymmetric_kl, uncommitted]
---

# occupancy_binary

> Binary occupancy for heatmap decode — matches CAD / ECADStar discrete layouts.

**Source:** `experiments/exp058_asymmetric_kl/codes/occupancy_binary.py` · 66 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp058_asymmetric_kl]]

## Purpose

```text
Binary occupancy for heatmap decode — matches CAD / ECADStar discrete layouts.

Fully vectorized (no Python loops / no GPU->CPU syncs) so it is safe to call on the
training decode hot path. On already-binary GT occupancy it is a no-op (returns the
same 0/1 grid with exactly K ones).
```

## Functions

- **`topk_occ_binary(occ_prob: torch.Tensor, k: 'int | torch.Tensor')`** — Hard top-K mask on 52-d occupancy (B, 52). Matches latent-opt / PEB binarization.
- **`is_binary_occupancy(occ: torch.Tensor, *, atol: float=0.0001)`** — Cheap CPU-syncing check — for diagnostics only, NOT the decode hot path.
- **`occupancy_for_heatmap_decode(occ: 'torch.Tensor | None', k: 'int | torch.Tensor', *, force_binary: bool=True, ste: bool=False)`** — Binarize occupancy before decoder spatial conditioning (CAD-aligned).

## Imported by

- [[experiments.exp058_asymmetric_kl.codes.eval_spatial_metrics]]
- [[experiments.exp058_asymmetric_kl.codes.vae_multi_input_simple]]

## External dependencies

`torch`
