---
title: Literature Review
type: moc
tags: [moc, literature, thesis]
---

# Literature Review

40 papers selected against what this vault actually contains — the PoE VAE, the graph encoders, the STE top-K read-out, the residual-GP acquisition loop, and the target-impedance feasibility rule. Curated, not exhaustive: everything here has a specific job in a specific chapter.

**21 are CORE** — a literature review missing any of these has a hole in it. The other 19 are supporting citations.

## Read these first

| Paper | Why it is unavoidable |
|-------|----------------------|
| [[Withoft2026 - Amortized Neural Optimization]] | Same idea as your Stage 2 (differentiable surrogate + analytical gradients), from the Zuken/TU Dortmund group behind your ECADSTAR toolchain |
| [[Kim2023 - DevFormer]] | Strongest competing decap-placement method; the paradigm you must contrast against |
| [[GomezBombarelli2018 - Automatic Chemical Design]] | The intellectual template for latent-space inverse design, including the off-manifold hazard |
| [[Wu2018 - MVAE Product of Experts]] | The PoE fusion and modality-dropout mechanism your architecture implements |
| [[Bengio2013 - Straight-Through Estimator]] | The gradient trick without which your optimizer is random search |
| [[ConvGA2024 - GCN-Assisted GA for Decap Optimization]] | Nearest complete competitor: graph surrogate + GA, 30% fewer capacitors — the baseline Ch.6 must beat |
| [[Zhang2021 - Fast PDN Impedance Prediction]] | Board-level PDN impedance DNN with decap placement as input; the precedent for your impedance surrogate |
| [[GIF2026 - Conditional Multimodal Generative IR Drop Imaging]] | Conditional multimodal generative model producing spatial PI maps — closest analogue to your heatmap decoder |
| [[Chhabria2021 - IREDGe ThermEDGe]] | The paper that legitimises decoding a PI field as an image |
| [[Zhao2024 - PDNNet GNN-CNN]] | Independent evidence that graph structure carries PI information a spatial view alone misses |
| [[Srinivas2010 - GP-UCB]] | The acquisition rule your residual-GP loop is derived from |

## By theme

### PDN domain

- [[Smith1999 - Target Impedance Methodology]] (1999) — **core**
- [[Park2022 - Transformer RL for HBM PDN]] (2022) — **core**
- [[Kim2023 - DevFormer]] (2023) — **core**
- [[ARS2023 - AI Modules for Decap Placement]] (2023) — **core**
- [[ConvGA2024 - GCN-Assisted GA for Decap Optimization]] (2024) — **core**
- [[Electronics2019 - Nature-Inspired Decap Placement]] (2019) — supporting
- [[GA2020 - Minimum Decap Count for Arbitrary Target]] (2020) — supporting
- [[Electronics2020 - Iterative GA plus ML Decoupling]] (2020) — supporting
- [[Duan2024 - Hierarchical Decap DRL 2.5D]] (2024) — supporting

### SI/PI surrogates

- [[Swaminathan2020 - Demystifying ML for SI PI]] (2020) — **core**
- [[Withoft2026 - Amortized Neural Optimization]] (2026) — **core**
- [[TODAES2022 - Worst-Case PI CNN]] (2022) — supporting
- [[TraceFormer2024 - Graph Transformer S-parameters]] (2024) — supporting
- [[Withoft2026 - Buffer-Parameterized Surrogates]] (2026) — supporting
- [[Ecik2026 - EMD SI-Compliant Design]] (2026) — supporting

### Generative models

- [[Kingma2013 - Auto-Encoding Variational Bayes]] (2013) — **core**
- [[Sohn2015 - Conditional VAE]] (2015) — **core**
- [[Wu2018 - MVAE Product of Experts]] (2018) — **core**
- [[Perez2018 - FiLM]] (2018) — **core**
- [[Higgins2017 - beta-VAE]] (2017) — supporting
- [[Shi2019 - MMVAE Mixture of Experts]] (2019) — supporting
- [[Kutuzova2021 - In Defense of Product of Experts]] (2021) — supporting

### Discrete gradients

- [[Bengio2013 - Straight-Through Estimator]] (2013) — **core**
- [[Jang2016 - Gumbel-Softmax]] (2016) — supporting
- [[Maddison2016 - Concrete Distribution]] (2016) — supporting

### Inverse design

- [[GomezBombarelli2018 - Automatic Chemical Design]] (2018) — **core**
- [[Mirhoseini2021 - Chip Placement RL]] (2021) — supporting

### Graph networks

- [[Kipf2016 - Graph Convolutional Networks]] (2016) — **core**
- [[Gilmer2017 - Message Passing Neural Networks]] (2017) — **core**

### Active learning

- [[Srinivas2010 - GP-UCB]] (2010) — **core**
- [[Gal2016 - Dropout as Bayesian Approximation]] (2016) — **core**
- [[Settles2009 - Active Learning Survey]] (2009) — supporting
- [[Lakshminarayanan2016 - Deep Ensembles]] (2016) — supporting
- [[BatchAL2022 - Batch Deep AL for Regression]] (2022) — supporting

### PI impedance surrogates

- [[Zhang2021 - Fast PDN Impedance Prediction]] (2021) — **core**
- [[Goay2022 - Adaptive Sampling for Plane Impedance]] (2022) — supporting

### PI spatial maps

- [[Chhabria2021 - IREDGe ThermEDGe]] (2021) — **core**
- [[Zhao2024 - PDNNet GNN-CNN]] (2024) — **core**
- [[GIF2026 - Conditional Multimodal Generative IR Drop Imaging]] (2026) — **core**
- [[Huang2021 - ML for EDA Survey]] (2021) — supporting

## Gaps this list does not cover

Stated plainly so they do not surprise you at review time:

- **No paper does what you do.** Nothing found combines a multimodal PoE VAE over (occupancy, spectrum, spatial PI field) with latent-space inverse design for decap placement. That is the novelty claim — but it also means there is no direct baseline to benchmark against, and you should say so rather than implying one exists.
- **But you now have a beatable baseline.** [[ConvGA2024 - GCN-Assisted GA for Decap Optimization]] pairs a graph impedance surrogate with a genetic algorithm on the same problem. The distinction is the optimizer — GA queries the surrogate's *value*, you differentiate *through* it. An equal-budget GA run on your own board is the single strongest experiment still available to you.
- **No shared benchmark.** Every PDN paper here uses its own board and its own target mask, so reported numbers are not comparable across papers or to yours.
- **The spatial-PI gap is narrower than it looked, but real.** The IR-drop map literature ([[Chhabria2021 - IREDGe ThermEDGe]], [[Zhao2024 - PDNNet GNN-CNN]], [[GIF2026 - Conditional Multimodal Generative IR Drop Imaging]]) establishes generative spatial PI prediction as a recognised task, so your heatmap decoder is no longer unsupported. What remains genuinely unaddressed: those works predict **on-chip voltage drop** maps conditioned on layout, whereas you generate **board-level impedance-distribution** maps conditioned on **frequency**. State the distinction precisely — it is a narrower and more defensible novelty claim than 'nobody does spatial PI'.
- **Scope note.** This list is deliberately PI-only. Signal-integrity work appears solely where the *method* transfers (the Zuken/TU Dortmund differentiable-surrogate papers); no EM/EMC field solving or emissions literature is included.
- **Citation hygiene.** Entries marked ⚠️ were taken from search results rather than fetched in full; verify author lists and page numbers before they enter the bibliography. The two IEEE decap-GA entries and ConvGA are paywalled — you will need library access to confirm them.

## Related

- [[Thesis Outline]] — where each of these lands
- [[Claim Ledger]] — what your own numbers can and cannot support
