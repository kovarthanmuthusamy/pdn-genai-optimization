---
title: graph_imp
type: code
path: experiments/exp058_asymmetric_kl/codes/graph_imp.py
group: experiments/exp058_asymmetric_kl/codes
experiment: exp058_asymmetric_kl
loc: 124
tags: [code, exp058_asymmetric_kl, uncommitted]
---

# graph_imp

> 1D spectrum GNN for PI impedance (231 bins) — exp058.

**Source:** `experiments/exp058_asymmetric_kl/codes/graph_imp.py` · 124 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp058_asymmetric_kl]]

## Constants

| Name | Value |
|------|-------|
| `NUM_BINS` | `231` |

## Classes

- **`ImpSpectrumGraphEncoder(nn.Module)`** — (B, 1, 231) or (B, 231) -> pooled feature (B, out_dim).
- **`ImpSpectrumGraphDecoder(nn.Module)`** — Global cond + optional occ context -> 231-bin spectrum.

## Functions

- **`build_spectrum_adjacency(*, self_loops: bool=True)`** — Chain graph along log-frequency bins (231 nodes).
- **`_bin_positions_norm()`**

## Imports

- [[experiments.exp058_asymmetric_kl.codes.graph_occ]]

## Imported by

- [[experiments.exp058_asymmetric_kl.codes.vae_multi_input_simple]]

## External dependencies

`torch`
