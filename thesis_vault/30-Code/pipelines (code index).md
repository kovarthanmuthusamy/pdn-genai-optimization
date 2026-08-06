---
title: pipelines (code index)
type: index
tags: [index, code, pipelines]
---

# pipelines/ — code index

79 modules.

## `pipelines/`

- [[pipelines.__init__]] — Package: pipelines
- [[_bootstrap]] — Bootstrap repo root for all ``pipelines.*`` modules.

## `pipelines/active_learning/`

- [[pipelines.active_learning.__init__]] — Package: pipelines.active_learning
- [[finetune_exp057]] *(runnable)* — Fine-tune exp057 after active-learning (Option B: heatmap-only overlay).
- [[finetune_exp058]] *(runnable)* — Fine-tune exp058 after active-learning (asymmetric: near AL path + distill).
- [[finetune_occ_only_exp057]] *(runnable)* — Pre-AL occupancy-only warm-up for exp057 (no overlay).
- [[finetune_occ_only_exp058]] *(runnable)* — Pre-AL occupancy-only warm-up for exp058 (no overlay).
- [[run]] *(runnable)* — Active-learning PI pipeline (exp059 capacity/freq by default).

## `pipelines/analysis/`

- [[pipelines.analysis.__init__]] — Package: pipelines.analysis
- [[check_mask]] *(runnable)* — Binary PCB mask shape inspector.
- [[dis_con]] *(runnable)* — Sigmoid-smooth 3-channel occupancy heatmap batch processor.
- [[impedance_decade_diversity]] *(runnable)* — Per-frequency-bin impedance diversity across layouts (231 PI-spectrum bins).
- [[latent_stats]] *(runnable)* — Per-modality latent statistics from a trained VAE checkpoint.
- [[pipelines.analysis.latent_traversal]] *(runnable)* — Systematic latent-space traversal for Multi-Input VAE.
- [[quick_traversal]] *(runnable)* — Quick latent-dimension traversal smoke test.

## `pipelines/data/`

- [[pipelines.data.__init__]] — Package: pipelines.data
- [[append_merged_combinations_multifreq]] *(runnable)* — Append multifreq rows from a **merged** ECADStar Raw run (PI-1..PI-N) into data_multifreq_train.
- [[append_peb_batch_raw]] *(runnable)* — Append multifreq rows from ECADStar PEB-batch Raw (PI-1..PI-N) into an existing dataset.
- [[append_restore_49k_legacy_multifreq]] *(runnable)* — Append missing legacy layout heatmaps from the 49k ECADStar restore into data_multifreq_train.
- [[copy_multifreq_k_subset]] *(runnable)* — Copy K<=5 subset of raw multifreq train (real files, no symlinks).
- [[data_check_stats]] *(runnable)* — Comprehensive dataset statistics and quality scoring.
- [[delete_combinations_append]] *(runnable)* — Delete combinations-appended samples from a multifreq dataset.
- [[occ_grid]] — Legacy script: convert Decaps*.csv rows to 7×8 Occ_map/sample_*.npy grids.
- [[processing_eval]] *(runnable)* — Build held-out evaluation dataset excluding training occupancy combinations.
- [[processing_multifreq]] *(runnable)* — Build multifreq training dataset (layout-centric storage).
- [[refresh_multifreq_train_meta]] — Lightweight JSON metadata refresh for data_multifreq_train.
- [[verify_layout_store]] *(runnable)* — Verify, migrate, and prune layout-centric multifreq dataset storage.
- [[visualize_sample]] *(runnable)* — Interactive viewer for processed heatmap, impedance, and occupancy samples.

## `pipelines/dataset/`

- [[pipelines.dataset.__init__]] — Package: pipelines.dataset
- [[clean_dataset_symlinks]] — Remove broken symlinks and materialize valid ones as real files (no symlinks in datasets).
- [[clean_train_metadata]] — Remove append progress, batch registries, and manifest backups from data_multifreq_train.
- [[dedupe_mhz_manifest]] — Remove duplicate manifest rows for one MHz (keep newest sample_N per design_id).
- [[extract_subset]] *(runnable)* — Extract a layout subset from full multifreq into a smaller folder.
- [[remove_mhz_from_train]] — Remove manifest rows + heatmap/PI_freq files for given MHz values in data_multifreq_train.
- [[subsample_inverse_k]] *(runnable)* — Inverse-K exponential subsampling of multifreq dataset layouts.

## `pipelines/dataset_sim/`

- [[pipelines.dataset_sim.__init__]] — Dataset simulation pipeline for combinations.csv layouts.
- [[append_queue]] — Persistent append job queue — sim enqueues MHz; one worker processes serially.
- [[combinations]] — Load decap combination rows from CSV.
- [[combinations_multitype]] — Load multi-type decap combination rows from CSV (type codes 0/1/2).
- [[pipelines.dataset_sim.ecadstar]] — ECADStar batch automation helpers for combinations simulation.
- [[move_outputs]] — Move ECADStar PI-* folders into destination (names unchanged).
- [[pipelines.dataset_sim.paths]] — Shared Windows paths and heatmap location helpers for sim → append pipeline.
- [[peb]] — Build ECADStar .peb files for combinations batch simulation.
- [[peb_multitype]] — Build ECADStar .peb files for multi-type combinations batch simulation.
- [[run_append_locked]] — Run one append job under an exclusive lock; refresh dataset_meta on success.
- [[run_append_worker]] — Process append_merged queue jobs one at a time (survives sim pipeline continuing).
- [[run_combinations_sim_pipeline]] *(runnable)* — Combinations.csv → ECADStar simulation pipeline (impedance, then PI-Distribution per MHz).
- [[run_multitype_sim_pipeline]] *(runnable)* — Multi-type combinations CSV -> ECADStar simulation pipeline.
- [[trigger_append]] — Enqueue append jobs and ensure a detached worker is running.
- [[verify_pipeline_paths]] — Verify sim → move → append path alignment for the combinations pipeline.

## `pipelines/heatmaps/`

- [[pipelines.heatmaps.__init__]] — Package: pipelines.heatmaps
- [[_install_append_merged_script]] *(runnable)* — One-shot installer for append_merged_combinations_multifreq.py (pipelines/data is cursorignored).
- [[_install_append_restore_49k_script]] *(runnable)* — Installer for append_restore_49k_legacy_multifreq.py (pipelines/data is cursorignored).
- [[append_legacy_19k_multifreq]] *(runnable)* — Append or replace legacy 19k layout heatmaps from Dataset_19k into data_multifreq_train.
- [[change_frequency]] *(runnable)* — Change PI-Distribution frequency in a single .peb file.
- [[generate_peb_from_csv]] *(runnable)* — Generate ECADStar .peb file(s) from a decap combinations CSV (52-column 0/1 rows).
- [[merge_combination_csvs]] *(runnable)* — Merge old + new decap layout CSVs for unified PEB / multifreq simulation.
- [[regenerate_mhz_pebs]] *(runnable)* — Batch-regenerate anchor-frequency PEB files from combined_all.peb.
- [[sample_multitype_combinations]] *(runnable)* — Sample multi-type decap layout CSVs (type codes, K <= 5).
- [[sample_new_combinations]] *(runnable)* — Sample new decap layout combinations for PI simulation (inverse-K, exclude existing).

## `pipelines/latent/`

- [[pipelines.latent.__init__]] — Package: pipelines.latent
- [[build_report]] *(runnable)* — Build latent opt report — self-contained HTML with embedded impedance/heatmap/occ plots.
- [[compare_report]] *(runnable)* — Latent run compare report — step 2: PI move, impedance plots, and report.
- [[export_peb]] *(runnable)* — Latent run export PEB — step 1: scrap layout export and PEB generation.
- [[find_feasible]] *(runnable)* — Find feasible configs — Monte Carlo sampling and filtering via VAE surrogate.
- [[generate_run_report]] *(runnable)* — Run report generator — Markdown summary for a latent optimization run.
- [[optimization_loader]] — Lazy loader for pipelines.latent.optimize.
- [[optimize]] *(runnable)* — Latent impedance optimization — gradient search in frozen VAE latent space.
- [[plot_results]] — Plot latent opt results — deprecated stub; redirects to export/compare pipeline.
- [[scrap_pipeline]] — Latent scrap pipeline — shared export, PEB, and impedance-compare helpers.

## `pipelines/normalize/`

- [[pipelines.normalize.__init__]] — Package: pipelines.normalize
- [[apply_stats]] *(runnable)* — Normalize a held-out dataset using training normalization stats.
- [[build_train_norm_unbounded]] — Build unbounded robust per-MHz dataset from raw ``data_multifreq_train``.
- [[compute_stats]] *(runnable)* — Raw dataset min/max and percentile statistics.
- [[multifreq]] *(runnable)* — Multifreq dataset normalization pipeline.
- [[verify]] *(runnable)* — Interactive normalization verification viewer.

## `pipelines/visualize/`

- [[pipelines.visualize.__init__]] — Package: pipelines.visualize
- [[pipelines.visualize.heatmap]] *(runnable)* — EM solver MAP file to heatmap visualizer.
- [[pipelines.visualize.impedance]] *(runnable)* — Impedance profile log-log comparison plotter.
