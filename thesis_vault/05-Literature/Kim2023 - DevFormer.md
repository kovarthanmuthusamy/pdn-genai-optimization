---
title: Kim2023 - DevFormer
type: literature
tier: core
year: 2023
tags: [literature, core, pdn-domain]
---

# DevFormer: A Symmetric Transformer for Context-Aware Device Placement

**Authors:** Haeyeon Kim, Minsu Kim, Federico Berto, Joungho Kim, Jinkyoo Park  
**Year:** 2023 · **Venue:** ICML 2023  
**Link:** https://arxiv.org/abs/2205.13225  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The strongest directly-competing method: decap placement solved by an offline-trained symmetric transformer with permutation-symmetry and relative-position inductive biases, reporting >30% fewer components than prior art on real hardware. This is your primary point of contrast — they learn a placement *policy* over a discrete action sequence, you learn a differentiable *surrogate* and descend gradients in a latent space. State the trade-off explicitly: their policy needs no per-target re-optimization, yours retargets to any new Z_target without retraining.

## Where it goes in the thesis

Ch.2 related work (primary baseline); Ch.9 comparison of paradigms.

## Links

- [[framework-overview]]
- [[latent-optimization]]
- [[gnn-rationale]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
