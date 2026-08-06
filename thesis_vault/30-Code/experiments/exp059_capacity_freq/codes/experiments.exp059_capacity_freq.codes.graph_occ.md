---
title: graph_occ
type: code
path: experiments/exp059_capacity_freq/codes/graph_occ.py
group: experiments/exp059_capacity_freq/codes
experiment: exp059_capacity_freq
loc: 179
tags: [code, exp059_capacity_freq, uncommitted]
---

# graph_occ

> Graph message-passing occupancy encoder/decoder for exp056 Graph VAE.

**Source:** `experiments/exp059_capacity_freq/codes/graph_occ.py` · 179 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp059_capacity_freq]]

## Purpose

```text
Graph message-passing occupancy encoder/decoder for exp056 Graph VAE.

Slots are nodes on the 7x8 PCB grid (52 valid decap positions); edges connect
4-neighbors on the board. Pure PyTorch — no torch_geometric dependency.
```

## Constants

| Name | Value |
|------|-------|
| `NUM_SLOTS` | `len(LABELS_ORDERED)` |

## Classes

- **`GraphConvLayer(nn.Module)`**
- **`OccGraphEncoder(nn.Module)`** — occupancy (B, 52) -> graph pool -> (B, out_dim).
- **`OccGraphDecoder(nn.Module)`** — Global latent cond (B, dec_in) -> per-node logits (B, 52).

## Functions

- **`_grid_neighbors(h: int, w: int)`**
- **`build_slot_adjacency(*, self_loops: bool=True)`** — Symmetric normalized adjacency (N, N) for 52 decap slots.
- **`_slot_positions_norm()`** — (N, 2) row/col normalized to [0, 1].

## Imports

- [[occupancy]]

## Imported by

- [[experiments.exp059_capacity_freq.codes.graph_imp]]
- [[experiments.exp059_capacity_freq.codes.vae_multi_input_simple]]

## External dependencies

`libs`, `torch`
