---
title: Zhao2024 - PDNNet GNN-CNN
type: literature
tier: core
year: 2024
tags: [literature, core, pi-spatial-maps]
---

# PDNNet: PDN-Aware GNN-CNN Heterogeneous Network for Dynamic IR Drop Prediction

**Authors:** Yuxiang Zhao, Zhuomin Chai, Xun Jiang, Yibo Lin, Runsheng Wang, Ru Huang  
**Year:** 2024 · **Venue:** IEEE Trans. CAD (TCAD)  
**Link:** https://arxiv.org/abs/2403.18569  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The strongest published justification for your central architectural bet — combining a graph view of the PDN with a spatial/convolutional view. They introduce PDNGraph and a dual-branch GNN-CNN network, and state plainly that prior CNN-only IR-drop work *overlooked PDN structure*; 545x speedup over the commercial tool. Cite this wherever you defend graph encoders over a flat MLP on the 52-slot vector: someone else independently concluded the graph structure carries information the image alone does not.

## Where it goes in the thesis

Ch.4 GNN + spatial decoder (the strongest external support for gnn-rationale).

## Links

- [[gnn-rationale]]
- [[model-architecture]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
