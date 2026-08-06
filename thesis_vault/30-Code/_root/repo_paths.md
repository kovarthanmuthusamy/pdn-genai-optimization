---
title: repo_paths
type: code
path: repo_paths.py
group: (root)
loc: 129
tags: [code, repo_paths.py]
---

# repo_paths

> Single source of truth for repository root and canonical paths.

**Source:** `repo_paths.py` · 129 lines

## Purpose

```text
Single source of truth for repository root and canonical paths.

All entry scripts should import from here instead of hard-coding ``parents[N]`` or
absolute paths like ``/home/ubuntu/genai_pdn``.

Usage::

    from repo_paths import REPO_ROOT, setup_path, repo_path
    setup_path()  # ensure repo root is on sys.path
    data = repo_path("datasets", "data_multifreq_train")
```

## Constants

| Name | Value |
|------|-------|
| `REPO_ROOT` | `Path(__file__).resolve().parent` |
| `PROJECT_ROOT` | `REPO_ROOT` |
| `DATASETS_DIR` | `REPO_ROOT / 'datasets'` |
| `EXPERIMENTS_DIR` | `REPO_ROOT / 'experiments'` |
| `SCRAP_DIR` | `REPO_ROOT / 'scrap'` |
| `PIPELINES_DIR` | `REPO_ROOT / 'pipelines'` |
| `TOOLS_DIR` | `REPO_ROOT / 'tools'` |
| `CONFIGS_DIR` | `REPO_ROOT / 'configs'` |
| `ACTIVE_LEARNING_DIR` | `REPO_ROOT / 'active_learning_pi'` |
| `HEATMAPS_DATA` | `REPO_ROOT / 'data' / 'heatmaps'` |
| `LATENT_RUNS_DATA` | `REPO_ROOT / 'data' / 'latent_runs'` |
| `DATA_MULTI_NORM_UNBOUNDED` | `REPO_ROOT / 'data_multi_norm_unbounded'` |
| `DATA_MULTI_NORM_ROBUST` | `REPO_ROOT / 'data_multi_norm_robust'` |
| `DATA_MULTI_NORM` | `REPO_ROOT / 'data_multi_norm'` |

## Functions

- **`setup_path()`** — Ensure repo root is on ``sys.path``; return ``REPO_ROOT``.
- **`repo_path(*parts: str)`** — Build a path under the repository root.
- **`bootstrap_from(caller_file: str | Path, depth: int)`** — Ensure repo root is on ``sys.path`` before ``import repo_paths``.
- **`strip_stale_prefix(path_str: str)`** — Remove known machine-specific repo prefixes, returning a repo-relative path.
- **`resolve_repo_path(path: str | Path, *, must_exist: bool=False)`** — Resolve a config/script path under ``REPO_ROOT``.
- **`resolve_config_paths(cfg: dict, keys: tuple[str, ...]=('data_dir', 'experiment_dir', 'resume_checkpoint'))`** — Return a shallow copy of *cfg* with selected path keys resolved under ``REPO_ROOT``.

## Imported by

- [[_analyze_hm_pattern]]
- [[_audit_decap_locality]]
- [[_audit_physics]]
- [[_bench_physics_overhead]]
- [[_bench_train_step]]
- [[_bootstrap]]
- [[_breakdown_hm_loss]]
- [[_breakdown_hm_real]]
- [[_inspect_layout]]
- [[_smoke_exp043_struct]]
- [[_test_expert_kl]]
- [[active_learning_pi.al.paths]]
- [[append_legacy_19k_multifreq]]
- [[append_merged_combinations_multifreq]]
- [[append_peb_batch_raw]]
- [[append_queue]]
- [[append_restore_49k_legacy_multifreq]]
- [[apply_stats]]
- [[build_comparison_report]]
- [[build_report]]
- [[build_train_norm_unbounded]]
- [[build_vault]]
- [[change_frequency]]
- [[check_exp044_norm_vs_train]]
- [[check_gmax_low_value_skew]]
- [[check_imp_layout]]
- [[check_mask]]
- [[check_multifreq_counts]]
- [[check_norm]]
- [[clean_dataset_symlinks]]
- [[clean_train_metadata]]
- [[compare]]
- [[compare_multifreq_datasets]]
- [[compare_report]]
- [[compute_stats]]
- [[copy_multifreq_k_subset]]
- [[dataset_meta]]
- [[dedupe_mhz_manifest]]
- [[delete_combinations_append]]
- [[diagnose_exp044_sweep]]
- [[diagnose_sweep_vs_dataset]]
- [[dis_con]]
- [[evaluate_hole_finding]]
- [[exp051_eval_common]]
- [[exp052_eval_common]]
- [[exp053_eval_common]]
- [[experiment_paths]]
- [[experiments.exp037_lat_change.codes.evaluate_vae]]
- [[experiments.exp037_lat_change.codes.inference_vae]]
- [[experiments.exp037_lat_change.codes.visualize_latent]]
- [[experiments.exp038_true_multi.codes.evaluate_vae]]
- [[experiments.exp038_true_multi.codes.inference_vae]]
- [[experiments.exp038_true_multi.codes.visualize_latent]]
- [[experiments.exp039_improved_heatmap.codes.codes.evaluate_vae]]
- [[experiments.exp039_improved_heatmap.codes.codes.inference_vae]]
- [[experiments.exp039_improved_heatmap.codes.codes.visualize_latent]]
- [[experiments.exp039_improved_heatmap.codes.inference_vae]]
- [[experiments.exp039_improved_heatmap.codes.train_vae_simple]]
- [[experiments.exp040.codes.codes.evaluate_vae]]
- [[experiments.exp040.codes.codes.inference_vae]]
- [[experiments.exp040.codes.codes.visualize_latent]]
- [[experiments.exp040.codes.inference_vae]]
- [[experiments.exp040.codes.train_vae_simple]]
- [[experiments.exp041.codes.codes.evaluate_vae]]
- [[experiments.exp041.codes.codes.inference_vae]]
- [[experiments.exp041.codes.codes.visualize_latent]]
- [[experiments.exp041.codes.inference_vae]]
- [[experiments.exp041.codes.train_vae_simple]]
- [[experiments.exp042.codes.inference_vae]]
- [[experiments.exp042.codes.train_vae_simple]]
- [[experiments.exp043.codes.inference_vae]]
- [[experiments.exp044.codes.inference_vae]]
- [[experiments.exp044.codes.train_vae_simple]]
- [[experiments.exp045.codes.inference_vae]]
- [[experiments.exp045.codes.train_vae_simple]]
- [[experiments.exp046.codes.eval_real_data_sweep]]
- [[experiments.exp046.codes.inference_vae]]
- [[experiments.exp046.codes.train_vae_simple]]
- [[experiments.exp047.codes.eval_real_data_sweep]]
- [[experiments.exp047.codes.inference_vae]]
- [[experiments.exp047.codes.train_vae_simple]]
- [[experiments.exp048.codes.eval_real_data_sweep]]
- [[experiments.exp048.codes.inference_vae]]
- [[experiments.exp048.codes.train_vae_simple]]
- [[experiments.exp049.codes.eval_real_data_sweep]]
- [[experiments.exp049.codes.inference_vae]]
- [[experiments.exp049.codes.train_vae_simple]]
- [[experiments.exp050.codes.eval_real_data_sweep]]
- [[experiments.exp050.codes.inference_vae]]
- [[experiments.exp050.codes.train_vae_simple]]
- [[experiments.exp051_new_datas_appended.codes.eval_real_data_sweep]]
- [[experiments.exp051_new_datas_appended.codes.inference_vae]]
- [[experiments.exp051_new_datas_appended.codes.train_vae_simple]]
- [[experiments.exp052_unbounded_pearson.codes.eval_real_data_sweep]]
- [[experiments.exp052_unbounded_pearson.codes.inference_vae]]
- [[experiments.exp052_unbounded_pearson.codes.train_vae_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.eval_real_data_sweep]]
- [[experiments.exp053_peak_log1p_losses.codes.inference_vae]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]
- [[experiments.exp054_K_30.codes.inference_vae]]
- [[experiments.exp054_K_30.codes.train_vae_simple]]
- [[experiments.exp055_hard_occ.codes.inference_vae]]
- [[experiments.exp055_hard_occ.codes.train_vae_simple]]
- [[experiments.exp056_graph_vae.codes.inference_vae]]
- [[experiments.exp056_graph_vae.codes.train_vae_simple]]
- [[experiments.exp057_structured_graph.codes.inference_vae]]
- [[experiments.exp057_structured_graph.codes.train_vae_simple]]
- [[experiments.exp058_asymmetric_kl.codes.inference_vae]]
- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]]
- [[experiments.exp059_capacity_freq.codes.inference_vae]]
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]]
- [[experiments.exp060_multitype_occ.codes.inference_vae]]
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]
- [[export_peb]]
- [[extract_subset]]
- [[find_feasible]]
- [[finetune_exp057]]
- [[finetune_exp058]]
- [[finetune_occ_only_exp057]]
- [[finetune_occ_only_exp058]]
- [[gan_paths]]
- [[generate_peb_from_csv]]
- [[generate_run_report]]
- [[generate_samples_and_peb]]
- [[impedance_decade_diversity]]
- [[latent_stats]]
- [[merge_combination_csvs]]
- [[move_and_compare]]
- [[move_pi_to_real]]
- [[multifreq]]
- [[multifreq_anchors]]
- [[multifreq_layout_store]]
- [[multifreq_move_and_compare]]
- [[optimize]]
- [[pipelines.visualize.heatmap]]
- [[pipelines.visualize.impedance]]
- [[processing_eval]]
- [[processing_multifreq]]
- [[quick_traversal]]
- [[read_new_stats]]
- [[refresh_multifreq_train_meta]]
- [[regenerate_mhz_pebs]]
- [[remove_mhz_from_train]]
- [[run]]
- [[run_all_k]]
- [[run_append_locked]]
- [[run_append_worker]]
- [[run_combinations_sim_pipeline]]
- [[run_multifreq_heatmap_sweep]]
- [[run_multifreq_sweep_pipeline]]
- [[run_multitype_sim_pipeline]]
- [[sample_multitype_combinations]]
- [[sample_new_combinations]]
- [[subsample_inverse_k]]
- [[test_sweep_load_exp044]]
- [[trigger_append]]
- [[validate_acquisition_ab]]
- [[verify]]
- [[verify_layout_store]]
- [[verify_peb_gmax_match]]
- [[verify_pipeline_paths]]
- [[visualize_sample]]
