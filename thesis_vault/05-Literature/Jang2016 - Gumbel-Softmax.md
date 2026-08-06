---
title: Jang2016 - Gumbel-Softmax
type: literature
tier: supporting
year: 2016
tags: [literature, supporting, discrete-gradients]
---

# Categorical Reparameterization with Gumbel-Softmax

**Authors:** Eric Jang, Shixiang Gu, Ben Poole  
**Year:** 2016 · **Venue:** ICLR 2017 / arXiv:1611.01144  
**Link:** https://arxiv.org/abs/1611.01144  
**Tier:** Supporting — cite in passing

## Why it matters here

The main alternative to a hard STE: a temperature-controlled continuous relaxation with a bias/variance trade-off. You should say why you chose hard top-K + identity gradient instead — your surrogate must see an in-distribution binary vector with exactly K ones, which a soft relaxation would violate.

## Where it goes in the thesis

Ch.6 justifying STE over a relaxation.

## Links

- [[latent-optimization]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
