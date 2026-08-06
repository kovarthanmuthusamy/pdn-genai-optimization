---
title: evaluation (code index)
type: index
tags: [index, code, evaluation]
---

# evaluation/ — code index

8 modules.

## `evaluation/novelty/runs/legacy_generated_samples/`

- [[build_generated_vs_real_all_K_md]] — Generate a single Markdown file embedding per-K comparison plots.

## `evaluation/novelty/scripts/`

- [[run_vae_novelty_sweep]] *(runnable)* — Generate + score novelty for many K values.
- [[run_vae_novelty_test]] *(runnable)* — End-to-end novelty test: generate N samples + score vs dataset.
- [[summarize_novelty_sweep]] *(runnable)* — Post-process a novelty sweep folder into a more detailed report.
- [[vae_novelty_report]] *(runnable)* — Nearest-neighbor novelty / memorization check for this repo's multi-modal VAE.

## `evaluation/vae/`

- [[evaluation.vae.latent_traversal]] *(runnable)* — Latent Traversal Analysis for Multi-Input VAE
- [[run_vae_eval]] — Evaluate a trained VAE checkpoint (reconstruction + latent diagnostics).
- [[run_vae_eval_heldout]] *(runnable)* — Full test evaluation on the held-out (non-training) dataset.
