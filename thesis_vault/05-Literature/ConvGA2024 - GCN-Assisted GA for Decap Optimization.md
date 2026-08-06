---
title: ConvGA2024 - GCN-Assisted GA for Decap Optimization
type: literature
tier: core
year: 2024
tags: [literature, core, pdn-domain]
---

# Graph Convolutional Neural Network Assisted Genetic Algorithm for PDN Decap Optimization

**Authors:** (IEEE conference paper 10705608)  
**Year:** 2024 · **Venue:** IEEE Xplore 10705608  
**Link:** https://ieeexplore.ieee.org/document/10705608/  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The nearest complete competitor to your whole pipeline: a graph-CNN impedance surrogate trained on BEM, ~500x faster than BEM, used inside a genetic algorithm that minimises both capacitor count and deviation from target impedance — reported as 15x faster optimization with 30% fewer capacitors. Your differentiator is the optimizer, not the surrogate: they search discretely with a GA that only queries the surrogate's value, you differentiate *through* the surrogate. That is the comparison your Chapter 6 has to win, and an equal-budget GA baseline on your own board would be the strongest experiment you could still run.

## Where it goes in the thesis

Ch.2 related work; Ch.6 the baseline to beat (consider implementing it).

## Links

- [[latent-optimization]]
- [[impedance-surrogate]]
- [[gnn-rationale]]
- [[Literature Review]]

---

⚠️ Title/venue from search results, not fetched in full — **verify author list, year and page numbers before citing**.
