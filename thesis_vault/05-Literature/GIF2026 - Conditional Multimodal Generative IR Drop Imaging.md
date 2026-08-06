---
title: GIF2026 - Conditional Multimodal Generative IR Drop Imaging
type: literature
tier: core
year: 2026
tags: [literature, core, pi-spatial-maps]
---

# GIF: A Conditional Multimodal Generative Framework for IR Drop Imaging in Chip Layouts

**Authors:** Kiran Thorat, Nicole Meng, Mostafa Karami, Caiwen Ding, Yingjie Lao, Zhijie Jerry Shi  
**Year:** 2026 · **Venue:** arXiv:2604.09999 (Apr 2026)  
**Link:** https://arxiv.org/abs/2604.09999  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The closest architectural analogue to your heatmap decoder that exists: a *conditional multimodal generative* model that fuses image (layout geometry) and graph (circuit topology) features to drive a conditional diffusion process producing spatial IR-drop maps, scored with SSIM/PSNR/correlation. Same three ingredients as you — multimodal fusion, conditioning, generative spatial PI output — with two differences to state: they use diffusion where you use a VAE decoder, and they condition on layout where you condition on frequency. Their metric set is also worth adopting alongside your Pearson / pattern-MAE. Published four months ago; check for a newer version before submitting.

## Where it goes in the thesis

Ch.4 heatmap decoder related work (the key citation); Ch.8 metric selection.

## Links

- [[model-architecture]]
- [[evaluation-metrics]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
