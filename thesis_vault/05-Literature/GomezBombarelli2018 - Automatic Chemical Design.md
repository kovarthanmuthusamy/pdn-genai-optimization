---
title: GomezBombarelli2018 - Automatic Chemical Design
type: literature
tier: core
year: 2018
tags: [literature, core, inverse-design]
---

# Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules

**Authors:** Rafael Gómez-Bombarelli, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling et al.  
**Year:** 2018 · **Venue:** ACS Central Science 4(2) / arXiv:1610.02415  
**Link:** https://arxiv.org/abs/1610.02415  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The canonical template for your entire method: encode discrete objects into a continuous latent space, attach a property predictor, then do gradient-based optimization in that space and decode. Your decap vector is their molecule; your impedance surrogate is their property predictor. They also hit the same failure you guard against with prior/boundary terms — optimizing off the data manifold decodes to invalid objects. Cite this as the intellectual lineage of Stage 2.

## Where it goes in the thesis

Ch.6 opening — the latent-optimization paradigm and the off-manifold hazard.

## Links

- [[latent-optimization]]
- [[framework-overview]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
