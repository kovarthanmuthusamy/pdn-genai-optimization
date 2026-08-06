---
title: occupancy_binary
type: code
path: experiments/exp060_multitype_occ/codes/occupancy_binary.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 61
tags: [code, exp060_multitype_occ, uncommitted]
---

# occupancy_binary

> Occupancy harden helpers (legacy binary + multi-type one-hot).

**Source:** `experiments/exp060_multitype_occ/codes/occupancy_binary.py` · 61 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

## Purpose

```text
Occupancy harden helpers (legacy binary + multi-type one-hot).

Heatmap spatial paths use per-slot presence. Multi-type layouts use
``occupancy_types`` one-hot (B, 52, C); this module keeps the binary top-K
utilities and a compatible ``occupancy_for_heatmap_decode`` entry point.
```

## Functions

- **`topk_occ_binary(occ_prob: torch.Tensor, k: 'int | torch.Tensor')`** — Hard top-K mask on 52-d occupancy (B, 52).
- **`is_binary_occupancy(occ: torch.Tensor, *, atol: float=0.0001)`**
- **`occupancy_for_heatmap_decode(occ: 'torch.Tensor | None', k: 'int | torch.Tensor', *, force_binary: bool=True, force_hard: bool | None=None, ste: bool=False)`** — Return presence (B, 52) for heatmap spatial conditioning.

## Imports

- [[occupancy_types]]

## Imported by

- [[experiments.exp060_multitype_occ.codes.eval_spatial_metrics]]
- [[occupancy_types]]

## External dependencies

`torch`
