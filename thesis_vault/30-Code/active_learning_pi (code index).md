---
title: active_learning_pi (code index)
type: index
tags: [index, code, active_learning_pi]
---

# active_learning_pi/ — code index

26 modules.

## `active_learning_pi/al/`

- [[active_learning_pi.al.__init__]] — Active learning package for PI-distribution heatmaps (``active_learning_pi.al``).
- [[acquisition]] — Acquisition functions — select worst/uncertain candidates for ECADSTAR simulation.
- [[build_overlay]] — Build layout_store overlay dataset from AL iterations (Option B: heatmap-only labels).
- [[candidates]] — Random decap-layout × PI-frequency candidate pool generation.
- [[config]] — Load/save active-learning pipeline configuration (JSON or YAML).
- [[decision_report]] — Decision ledger: collect numerical evals → PASS/FAIL/UNCERTAIN claims.
- [[active_learning_pi.al.ecadstar]] — ECADSTAR batch simulation bridge (native headless CLI).
- [[evaluate_cycle]] — Full-cycle evaluation: pre/post fine-tune ECAD labels + training off-anchor metrics.
- [[evaluate_hole_finding]] *(runnable)* — Solid hole-finding evaluation for residual-GP active learning.
- [[evaluate_off_anchor]] — Evaluate model predictions vs ECADSTAR labels at off-anchor MHz frequencies.
- [[evaluate_report]] — Markdown cycle evaluation report (pre/post fine-tune + training metrics).
- [[finetune_run]] — Resolve AL fine-tune schedule from ``last_model.pt`` checkpoint epoch.
- [[gp_error_surrogate]] — Latent-space error-GP surrogate for active-learning acquisition.
- [[gp_error_surrogate_test]] — Falsification test for the GP error-surrogate AL idea.
- [[inference_pool]] — VAE Monte-Carlo inference over the candidate pool (uncertainty scoring).
- [[ingest_labels]] — Ingest ECADSTAR PI-Distribution outputs into per-sample label directories.
- [[k_config]] — Resolve decap-count K settings for active-learning candidate pools.
- [[mhz_strata]] — Stratified MHz quotas for per-K AL candidate pools and ECAD selection.
- [[normalize_labels]] — Package and normalize ingested ECADSTAR labels for VAE fine-tuning.
- [[active_learning_pi.al.paths]] — Project and run-directory path resolution for active learning.
- [[peb_batch]] — Build a single combined ECADSTAR .peb for all selected worst-case candidates.
- [[per_k_acquire]] — Per-K candidate pools: N candidates and worst-M selection for each K value.
- [[pipeline]] — Active-learning orchestrator — full generate→simulate→ingest→normalize loop.
- [[robust_normalize]] — Robust per-MHz heatmap normalization for active-learning labels.
- [[robust_stats]] — Robust foreground peak statistics on physical heatmaps.
- [[validate_acquisition_ab]] *(runnable)* — Equal-budget acquisition A/B: GP-UCB (+novelty) vs random vs optional MC.
