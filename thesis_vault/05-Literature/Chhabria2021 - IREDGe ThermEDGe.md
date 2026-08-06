---
title: Chhabria2021 - IREDGe ThermEDGe
type: literature
tier: core
year: 2021
tags: [literature, core, pi-spatial-maps]
---

# Encoder-Decoder Networks for Analyzing Thermal and Power Delivery Networks

**Authors:** Vidya A. Chhabria, Vipul Ahuja, Ashwath Prabhu, Nikhil Patil, Palkesh Jain, Sachin S. Sapatnekar  
**Year:** 2021 · **Venue:** ACM TODAES 2022 / arXiv:2110.14197  
**Link:** https://arxiv.org/abs/2110.14197  
**Tier:** CORE — belongs in the literature review

## Why it matters here

The canonical 'PI analysis as image-to-image translation' paper: IREDGe predicts full-chip static and dynamic IR-drop maps, ThermEDGe temperature maps, via an encoder-decoder generative (EDGe) architecture, in milliseconds against hours for commercial tools. This is the reference that legitimises treating a PI field as an image and decoding it — the core assumption of your 64x64 heatmap head. Their transferability-within-technology claim is also the template for arguing your model transfers across layouts of one board.

## Where it goes in the thesis

Ch.4 justifying the spatial decoder (essential); Ch.9 transferability framing.

## Links

- [[model-architecture]]
- [[evaluation-metrics]]
- [[Literature Review]]

---

Bibliographic details confirmed against the source page.
