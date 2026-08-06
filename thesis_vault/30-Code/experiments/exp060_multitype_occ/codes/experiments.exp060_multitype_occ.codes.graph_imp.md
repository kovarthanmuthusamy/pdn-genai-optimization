---
title: graph_imp
type: code
path: experiments/exp060_multitype_occ/codes/graph_imp.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 124
tags: [code, exp060_multitype_occ, uncommitted]
---

# graph_imp

> 1D spectrum GNN for PI impedance (231 bins) — exp060.

**Source:** `experiments/exp060_multitype_occ/codes/graph_imp.py` · 124 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

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

- [[experiments.exp060_multitype_occ.codes.graph_occ]]

## Imported by

- [[experiments.exp060_multitype_occ.codes.vae_multi_input_simple]]

## External dependencies

`torch`
