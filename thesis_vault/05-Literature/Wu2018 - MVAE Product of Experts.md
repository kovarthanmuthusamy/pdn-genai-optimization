---
title: Wu2018 - MVAE Product of Experts
type: literature
tier: core
year: 2018
tags: [literature, core, generative-models]
---

# Multimodal Generative Models for Scalable Weakly-Supervised Learning

**Authors:** Mike Wu, Noah Goodman  
**Year:** 2018 · **Venue:** NeurIPS 2018 / arXiv:1802.05335  
**Link:** https://arxiv.org/abs/1802.05335  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The Product-of-Experts multimodal VAE your architecture implements, including the sub-sampled training paradigm that is the ancestor of your modality dropout. Their whole point — robustness to missing modalities — is what makes your occupancy-only encode path work at inference, when spectrum and heatmaps are unavailable. Cite this for the PoE precision-weighted fusion equations.

## Where it goes in the thesis

Ch.4 PoE fusion and modality dropout (the load-bearing citation).

## Links

- [[model-architecture]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
