---
title: occupancy_types
type: code
path: experiments/exp060_multitype_occ/codes/occupancy_types.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 204
tags: [code, exp060_multitype_occ, uncommitted]
---

# occupancy_types

> Multi-type decap occupancy: one-hot over catalog types only (no empty channel).

**Source:** `experiments/exp060_multitype_occ/codes/occupancy_types.py` · 204 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

## Purpose

```text
Multi-type decap occupancy: one-hot over catalog types only (no empty channel).

Canonical layout tensor for exp060:

    occ ∈ {0,1}^{N × T} with N = 52
    empty slot     = all zeros
    type t (1..T)  = one-hot on channel (t-1)

On disk prefer ``type.npy`` with integer codes in {0,...,T} (0 = empty).
Legacy binary ``occ.npy`` (0/1) maps to empty / type-1.
```

## Constants

| Name | Value |
|------|-------|
| `NUM_SLOTS` | `52` |

## Functions

- **`n_occ_classes(n_decap_types: int)`** — Number of type channels (= T). Empty is the zero vector, not a channel.
- **`type_ids_to_onehot(type_ids: IntTensor, n_decap_types: int, *, dtype: torch.dtype=torch.float32)`** — Convert type codes → (..., N, T) one-hot; empty codes stay all-zero rows.
- **`_codes_to_onehot(codes: torch.Tensor, n_types: int, *, dtype: torch.dtype)`** — codes in {0,...,T}; 0 → zero row; k≥1 → channel k-1.
- **`onehot_to_type_ids(occ: torch.Tensor)`** — (B, N, T) or (N, T) → integer codes (..., N); empty rows → 0.
- **`presence_from_onehot(occ: torch.Tensor)`** — Occupied mask: any type channel on. Shape (..., N).
- **`k_from_onehot(occ: torch.Tensor)`** — K = number of non-empty slots.
- **`logits_to_probs(logits: torch.Tensor)`** — (B, N, T) logits → independent sigmoid type probs (empty ≈ all low).
- **`hard_onehot_from_logits(logits: torch.Tensor, k: 'int | torch.Tensor')`** — CAD-aligned hard layout from per-slot type logits (B, N, T).
- **`hard_onehot_from_probs(occ: torch.Tensor, k: 'int | torch.Tensor')`** — Harden (B, N, T) one-hot/probs/logits with budget K.
- **`occupancy_for_heatmap_decode(occ: 'torch.Tensor | None', k: 'int | torch.Tensor', *, force_hard: bool=True, ste: bool=False)`** — Return presence (B, N) for heatmap spatial grids after optional hard budget.
- **`ensure_batch_onehot(occ: torch.Tensor, n_decap_types: int)`** — Normalize batch occupancy to (B, N, T) float one-hot.

## Imports

- [[experiments.exp060_multitype_occ.codes.occupancy_binary]]

## Imported by

- [[experiments.exp060_multitype_occ.codes.dataloader_base]]
- [[experiments.exp060_multitype_occ.codes.eval_spatial_metrics]]
- [[experiments.exp060_multitype_occ.codes.inference_vae]]
- [[experiments.exp060_multitype_occ.codes.occupancy_binary]]
- [[experiments.exp060_multitype_occ.codes.train_core]]
- [[experiments.exp060_multitype_occ.codes.vae_multi_input_simple]]

## External dependencies

`numpy`, `torch`
