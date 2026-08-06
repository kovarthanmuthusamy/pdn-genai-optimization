---
title: Zhang2021 - Fast PDN Impedance Prediction
type: literature
tier: core
year: 2021
tags: [literature, core, pi-impedance-surrogates]
---

# Fast PDN Impedance Prediction Using Deep Learning

**Authors:** Ling Zhang, Jack Juang, Zurab Kiguradze, Bo Pu, Shuai Jin, Songping Wu, Zhiping Yang, Chulsoon Hwang  
**Year:** 2021 · **Venue:** arXiv:2106.10693 / Int. J. Numerical Modelling 2022  
**Link:** https://arxiv.org/abs/2106.10693  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The direct precedent for your occupancy→impedance surrogate, and the single best comparison point for your data strategy. A DNN predicts PDN impedance for arbitrary board shape, stackup, IC location **and decap placement**, trained on >1M BEM-generated boards, at 0.1 s — 100x faster than BEM and 5000x faster than full-wave. Two things to draw out: (1) they generalize across *boards* where you train one board deeply, which is the honest framing of your single-design limitation; (2) their supervision is BEM, yours is ECADSTAR PI simulation, so your labels are costlier and fewer — which is exactly what motivates your active-learning loop and theirs did not need.

## Where it goes in the thesis

Ch.4/6 impedance surrogate; Ch.3 dataset scale comparison; Ch.9 single-board limitation.

## Links

- [[impedance-surrogate]]
- [[dataset]]
- [[framework-overview]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
