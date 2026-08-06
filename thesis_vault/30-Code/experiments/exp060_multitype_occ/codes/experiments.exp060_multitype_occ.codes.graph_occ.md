---
title: graph_occ
type: code
path: experiments/exp060_multitype_occ/codes/graph_occ.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 197
tags: [code, exp060_multitype_occ, uncommitted]
---

# graph_occ

> Graph message-passing occupancy encoder/decoder for multi-type occ (exp060).

**Source:** `experiments/exp060_multitype_occ/codes/graph_occ.py` · 197 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

## Purpose

```text
Graph message-passing occupancy encoder/decoder for multi-type occ (exp060).

Slots are nodes on the 7x8 PCB grid (52 valid decap positions); edges connect
4-neighbors on the board. Pure PyTorch — no torch_geometric dependency.
```

## Constants

| Name | Value |
|------|-------|
| `NUM_SLOTS` | `len(LABELS_ORDERED)` |

## Classes

- **`GraphConvLayer(nn.Module)`**
- **`OccGraphEncoder(nn.Module)`** — occupancy (B, 52, T) one-hot -> graph pool -> (B, out_dim).
- **`OccGraphDecoder(nn.Module)`** — Global latent cond (B, dec_in) -> per-node class logits (B, 52, C).

## Functions

- **`_grid_neighbors(h: int, w: int)`**
- **`build_slot_adjacency(*, self_loops: bool=True)`** — Symmetric normalized adjacency (N, N) for 52 decap slots.
- **`_slot_positions_norm()`** — (N, 2) row/col normalized to [0, 1].

## Imports

- [[occupancy]]

## Imported by

- [[experiments.exp060_multitype_occ.codes.graph_imp]]
- [[experiments.exp060_multitype_occ.codes.vae_multi_input_simple]]

## External dependencies

`libs`, `torch`
