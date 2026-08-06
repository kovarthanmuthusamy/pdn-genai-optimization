---
title: scratch (code index)
type: index
tags: [index, code, scratch]
---

# scratch/ — code index

21 modules.

## `scratch/`

- [[_analyze_hm_pattern]] — Quick spatial analysis of sweep heatmaps vs real.
- [[_audit_decap_locality]] — Check if GT occupancy correlates with local low-Z in exp043 train space.
- [[_audit_physics]] — Audit PhysicsCritic + radius_influence for exp043 log1p/gmax heatmaps.
- [[_bench_physics_overhead]] — Compare one training step with vs without PhysicsLoss.
- [[_bench_train_step]] — Rough per-batch timing breakdown for exp043 training step.
- [[_breakdown_hm_loss]] — Break down heatmap loss components at init.
- [[_breakdown_hm_real]] — Per-term heatmap loss on one real train batch (untrained model).
- [[_smoke_exp043_struct]] — Smoke test: 8×8 heatmap bottleneck + pattern/spread losses.
- [[_test_expert_kl]]
- [[_transform_compare]]
- [[check_exp044_norm_vs_train]] — Check exp044 normalization consistency and encode-native reconstruction quality.
- [[check_gmax_low_value_skew]] — Temp check: low-value skew in data_multifreq_gmax normalized heatmaps.
- [[check_imp_layout]]
- [[check_log1p_train_space]] — Verify log1p train-space roundtrip + skew improvement.
- [[check_multifreq_counts]]
- [[check_norm]]
- [[diagnose_exp044_sweep]] — Quick encode vs layout diagnostic for exp044 at off-anchor MHz.
- [[diagnose_sweep_vs_dataset]] — Compare sweep inference vs dataset-ground-truth paths (encode / layout).
- [[read_new_stats]]
- [[test_sweep_load_exp044]] — Quick smoke test: exp044 VAEInference loads for sweep.
- [[verify_peb_gmax_match]] — TEMPORARY one-off: verify PEB-batch append + gmax dataset alignment.
