---
title: graph_occ
type: code
path: experiments/exp056_graph_vae/codes/graph_occ.py
group: experiments/exp056_graph_vae/codes
experiment: exp056_graph_vae
loc: 179
tags: [code, exp056_graph_vae]
---

# graph_occ

> Graph message-passing occupancy encoder/decoder for exp056 Graph VAE.

**Source:** `experiments/exp056_graph_vae/codes/graph_occ.py` · 179 lines
**Experiment:** [[exp056_graph_vae]]

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

- [[experiments.exp056_graph_vae.codes.vae_multi_input_simple]]

## External dependencies

`libs`, `torch`
