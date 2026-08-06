---
title: Gal2016 - Dropout as Bayesian Approximation
type: literature
tier: core
year: 2016
tags: [literature, core, active-learning]
---

# Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning

**Authors:** Yarin Gal, Zoubin Ghahramani  
**Year:** 2016 · **Venue:** ICML 2016 / arXiv:1506.02142  
**Link:** https://arxiv.org/abs/1506.02142  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The theoretical licence for your MC-dropout acquisition scores (mc_decoder_dropout, mc_passes). Important caveat for your write-up: this justifies dropout sampling as *approximate* Bayesian inference, which is consistent with your finding that MC self-uncertainty underperformed the residual GP — cite it when explaining why you moved from MC to GP scoring rather than treating MC as broken.

## Where it goes in the thesis

Ch.7 MC-dropout baseline and why it was superseded.

## Links

- [[active-learning]]
- [[gp-error-surrogate]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
