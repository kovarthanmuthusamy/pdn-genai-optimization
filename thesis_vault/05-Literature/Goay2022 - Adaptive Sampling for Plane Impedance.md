---
title: Goay2022 - Adaptive Sampling for Plane Impedance
type: literature
tier: supporting
year: 2022
tags: [literature, supporting, pi-impedance-surrogates]
---

# Power-Ground Plane Impedance Modeling Using Deep Neural Networks and an Adaptive Sampling Process

**Authors:** Goay et al.  
**Year:** 2022 · **Venue:** Int. J. Electronics and Telecommunications  
**Link:** https://ijet.pl/index.php/ijet/article/view/10.24425-ijet.2022.143887  
**Tier:** Supporting — cite in passing

## Why it matters here

Active learning for PI, before anyone called it that: start from few samples, fit a cheap surrogate, and concentrate new samples where that surrogate predicts poorly — structurally identical to your residual-GP acquisition. Also solves your normalization problem the same way, with a second network predicting the min/max of the un-normalized Z so accuracy survives large dynamic range across designs. Closest PI-domain precedent for both your AL loop and your robust normalization.

## Where it goes in the thesis

Ch.7 AL precedent in PI; Ch.5 normalization design.

## Links

- [[active-learning]]
- [[gp-error-surrogate]]
- [[normalization-and-losses]]
- [[Literature Review]]

---

⚠️ Title/venue from search results, not fetched in full — **verify author list, year and page numbers before citing**.
