---
title: Srinivas2010 - GP-UCB
type: literature
tier: core
year: 2010
tags: [literature, core, active-learning]
---

# Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design

**Authors:** Niranjan Srinivas, Andreas Krause, Sham Kakade, Matthias Seeger  
**Year:** 2010 · **Venue:** ICML 2010 / arXiv:0912.3995  
**Link:** https://arxiv.org/abs/0912.3995  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The regret analysis behind the UCB rule a(x)=mu(x)+kappa*sigma(x) your acquisition implements. Note honestly that your production setting uses score_mode: mu (kappa effectively 0), i.e. pure exploitation of the residual GP — the theory here motivates the form but does not by itself justify dropping exploration; your equal-budget A/B does.

## Where it goes in the thesis

Ch.7 acquisition function; be precise that you run mu, not full UCB.

## Links

- [[gp-error-surrogate]]
- [[active-learning]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
