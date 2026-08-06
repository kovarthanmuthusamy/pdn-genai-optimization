---
title: Withoft2026 - Amortized Neural Optimization
type: literature
tier: core
year: 2026
tags: [literature, core, si-pi-surrogates]
---

# Amortized Neural Optimization for Pre-Layout Signal Integrity Design Space Exploration using Differentiable Surrogates

**Authors:** Julian Withöft, Werner John, Emre Ecik, Ralf Brüning, Jürgen Götze  
**Year:** 2026 · **Venue:** arXiv:2606.07463 (Jun 2026)  
**Link:** https://arxiv.org/abs/2606.07463  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The methodological twin of your Stage 2 and the single most important paper here. They build fully differentiable NN surrogates, extract analytical gradients, and train a policy that maps context to near-optimal parameters in one forward pass — 3-4 orders of magnitude speedup for ~10% optimality loss. Differences you must state: they optimize continuous equalizer/routing parameters in *input* space; you optimize a *latent* vector and must cross a discrete top-K read-out, which is exactly why you need the STE. Note also Brüning is at Zuken's EMC Technology Center Paderborn — the ECADSTAR vendor your pipeline drives.

## Where it goes in the thesis

Ch.2 related work; Ch.6 positioning of latent vs. input-space gradient design.

## Links

- [[framework-overview]]
- [[latent-optimization]]
- [[impedance-surrogate]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
