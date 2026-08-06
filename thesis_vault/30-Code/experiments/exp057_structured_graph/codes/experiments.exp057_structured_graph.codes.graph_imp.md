---
title: graph_imp
type: code
path: experiments/exp057_structured_graph/codes/graph_imp.py
group: experiments/exp057_structured_graph/codes
experiment: exp057_structured_graph
loc: 124
tags: [code, exp057_structured_graph]
---

# graph_imp

> 1D spectrum GNN for PI impedance (231 bins) — exp057.

**Source:** `experiments/exp057_structured_graph/codes/graph_imp.py` · 124 lines
**Experiment:** [[exp057_structured_graph]]

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

- [[experiments.exp057_structured_graph.codes.graph_occ]]

## Imported by

- [[experiments.exp057_structured_graph.codes.vae_multi_input_simple]]

## External dependencies

`torch`
