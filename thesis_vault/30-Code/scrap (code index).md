---
title: scrap (code index)
type: index
tags: [index, code, scrap]
---

# scrap/ — code index

30 modules.

## `scrap/`

- [[audit_dataset_stats]]
- [[debug_clip_spatial]] — How much does z-clip flatten spatial structure per MHz?
- [[debug_inspect_robust_stats]] — Inspect robust per-MHz stats and clip ceilings.
- [[peb_copy]] — Copy PEB Files (WSL-Safe).
- [[tmp_compare_clip]]

## `scrap/comparison/`

- [[scrap.comparison.__init__]]
- [[batch_over_k]] — Run a single-K comparison module for K_MIN..K_MAX.
- [[compare]] *(runnable)* — Compare Generated vs Real (Workflow-Aware).
- [[compare_generated_vs_real]] *(runnable)* — Compare generated vs real for one K.
- [[compare_generated_vs_real_all_k]] *(runnable)* — Generated vs real comparison for all K.
- [[compare_generated_vs_real_occupancy]] *(runnable)* — Occupancy checkbox plot for one K.
- [[compare_generated_vs_real_occupancy_all_k]] *(runnable)* — Occupancy comparison for all K.
- [[heatmap_sim_metrics]] — Metrics for generated vs ECADStar-simulated real heatmaps (post-move).
- [[move_and_compare]] *(runnable)* — Move PI Outputs and Compare (run_all_k).

## `scrap/generation/`

- [[scrap.generation.__init__]]
- [[generate_peb]] — Generate ECADStar Batch PEB.
- [[generate_peb_multitype]] — Generate ECADStar Batch PEB for multi-type decap layouts.
- [[generate_samples_and_peb]] *(runnable)* — Generate VAE samples and PEB for one K.
- [[generate_samples_and_peb_all_k]] *(runnable)* — Generate samples and combined PEB for all K (legacy wrapper).
- [[run_all_k]] *(runnable)* — Generate VAE Samples for K Sweep.
- [[run_multifreq_heatmap_sweep]] *(runnable)* — Multifreq Heatmap Sweep at Fixed K (or explicit K list).
- [[sweep_latent_opt_rules]] — Hard rules aligning multifreq sweep QC with ``pipelines/latent/optimize.py``.
- [[sweep_qc_eval]] — Lightweight sweep QC metrics + agent-copy report.

## `scrap/orchestration/`

- [[scrap.orchestration.__init__]] — Package: scrap.orchestration
- [[build_comparison_report]] *(runnable)* — Build Comparison HTML Report.
- [[compare_multifreq_datasets]] — Compare Multifreq Dataset Layouts.
- [[move_pi_to_real]] *(runnable)* — Move ECADStar PI Outputs to Real/ Folders.
- [[multifreq_move_and_compare]] *(runnable)* — Multifreq Move, Compare, and Report.
- [[plot_sim_compare_metrics]] — Plot sim_compare_metrics.json from a multifreq sweep run.
- [[run_multifreq_sweep_pipeline]] *(runnable)* — Multifreq heatmap sweep pipeline (generate → simulate → compare).
