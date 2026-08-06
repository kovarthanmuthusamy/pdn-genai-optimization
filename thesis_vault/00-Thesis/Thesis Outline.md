---
title: Thesis Outline
type: moc
tags: [moc, thesis]
---

# Thesis Outline

Chapter map for the PDN generative-surrogate thesis. Each section lists the vault notes
that hold the material, so a chapter can be drafted by opening its row and following links.

Notes under `10-Concepts/` are mirrors of `docs/*.md` — **edit the source doc, not the mirror**,
then re-run `tools/build_vault.py`. Notes here in `00-Thesis/` and in `15-Ideas/` are yours;
the generator never overwrites them.

---

## 1. Introduction & problem statement

| Material | Notes |
|----------|-------|
| Problem framing, target impedance, combinatorial search | [[framework-overview]] |
| Why the conventional loop is expensive | [[framework-overview]] |

Key facts to state: binary placement $\mathbf b \in \{0,1\}^{52}$, decap budget $K$,
231-point spectrum over 1–600 MHz, feasibility as $Z(f) \le Z_\text{target}(f)$.

## 2. Background & related work

| Material | Notes |
|----------|-------|
| Why a GNN over the PCB slot graph | [[gnn-rationale]] |
| Product-of-Experts multimodal VAE | [[model-architecture]] |

## 3. Dataset & data pipeline

| Material | Notes |
|----------|-------|
| Layout store, anchors, K≤30 filter, sampling | [[dataset]] |
| Simulation → append → normalize | [[data-pipeline]] |
| Robust per-MHz normalization and loss design | [[normalization-and-losses]] |
| Dataset directories | [[data_multifreq_train_norm_unbounded]], [[data_multifreq_al_overlay_exp059]] |
| Code | [[processing_multifreq]], [[dataset_meta]], [[ingest_labels]] |

⚠️ Datasets are untracked. Report row counts from `dataset_meta.json`, not from memory.

## 4. Surrogate architecture (Stage 1)

| Material | Notes |
|----------|-------|
| PoE VAE, structured latent, FiLM conditioning | [[model-architecture]] |
| Occupancy + spectrum graph encoders | [[gnn-rationale]] |
| Current model | [[exp059_capacity_freq]] |
| Code | [[experiments.exp059_capacity_freq.codes.vae_poe_freq]], [[experiments.exp059_capacity_freq.codes.graph_occ]], [[experiments.exp059_capacity_freq.codes.graph_imp]] |

The architectural claim of the chapter — modality dropout is what makes the
**occupancy-only encode path** usable at inference — should be cited against
`layout_train_prob` / `occ_only_encode_prob` in the experiment config table.

## 5. Training procedure

| Material | Notes |
|----------|-------|
| Loop structure, schedules, DDP | [[training-procedure]] |
| Loss composition | [[normalization-and-losses]] |
| Code | [[experiments.exp059_capacity_freq.codes.train_core]], [[experiments.exp059_capacity_freq.codes.distributed_train]] |

## 6. Inverse design (Stage 2)

| Material | Notes |
|----------|-------|
| Latent search, STE top-K, selection rule | [[latent-optimization]] |
| Occupancy → spectrum forward model | [[impedance-surrogate]] |
| Code | [[optimize]], [[find_feasible]] |

⚠️ **Scope caveat to state explicitly:** `pipelines/latent/optimize.py` still defaults to
`EXPERIMENT = "exp038_true_multi"`. Stage 2 is not wired to [[exp059_capacity_freq]].
Any end-to-end claim must either re-run Stage 2 on exp059 or be scoped to exp038.

## 7. Active learning

| Material | Notes |
|----------|-------|
| Option-B MC cycle, 8 stages | [[active-learning]] |
| Residual-GP acquisition, UCB, `score_mode: mu` | [[gp-error-surrogate]] |
| Code | [[pipeline]], [[gp_error_surrogate]], [[inference_pool]], [[per_k_acquire]] |
| Results | [[al_exp059_gp_error_001]], [[acquisition_ab_validation_exp059]], [[hole_finding_exp059]] |

## 8. Evaluation

| Material | Notes |
|----------|-------|
| Primary QC metrics | [[evaluation-metrics]] |
| Held-out, novelty, diagnostics | [[evaluation-suite]] |
| Verdicts and evidence | [[Claim Ledger]] |

## 9. Limitations & future work

| Material | Notes |
|----------|-------|
| Threats to validity, claims hygiene | [[limitations-and-validity]] |

Points that must survive into the written chapter: single-board training,
surrogate-space feasibility requiring ground-truth re-simulation, and the
Stage-2/Stage-1 misalignment noted in §6.

---

## Experiment lineage for the methods chapter

[[experiment-lineage]] carries the full arc. The short version used in the narrative:

[[exp043]] → [[exp054_K_30]] → [[exp055_hard_occ]] → [[exp056_graph_vae]] →
[[exp057_structured_graph]] → [[exp058_asymmetric_kl]] → **[[exp059_capacity_freq]]** → [[exp060_multitype_occ]]

## Writing rules carried over from the repo

From `.cursor/rules/numerical-claims.mdc`: never assert improvement, ranking quality, or
"uncertainty works" without either citing a numerical artifact or labelling the statement
a hypothesis. [[Claim Ledger]] is where each thesis claim meets its evidence.
