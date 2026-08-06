# Decision report — `al_exp059_gp_error_001` iter 3

**Generated:** 2026-07-23 14:07:29 UTC  
**Experiment:** `experiments/exp059_capacity_freq`  
**Acquisition mode:** `gp_error`  
**Overall:** **PASS**  

Numerical evidence supports directional acquisition + FT improvement (PASS: acq_direction, acq_ab_gp_vs_random, hole_finding, ft_physical_improve, eval_budget_trust, train_off_anchor_logged). Remaining optional gaps: acq_ab_gp_vs_mc.

---

## How to use this report

Every workflow stage must leave **numerical** artifacts. This ledger turns them into
claims you can accept or reject:

1. **acq_direction** — is uncertainty / GP scoring pointing at true error?
2. **acq_ab_gp_vs_random** — equal-budget GP vs random (and MC if present)?
3. **hole_finding** — does GP top-K enrich true high-error holes vs random?
4. **ft_physical_improve** — did fine-tune actually improve physical p99?
5. **eval_budget_trust** — was the ECAD budget fully evaluated?
6. **train_off_anchor_logged** — did training log off-anchor numbers?

Only claim a thesis result when blocking claims are **PASS** (or explicitly scoped).

---

## Claim checklist

| Status | Claim | Key evidence |
| --- | --- | --- |
| PASS | `acq_direction` — Acquisition / uncertainty scores rank true ECAD error (Spearman > 0). | spearman_badness_vs_p99_err=0.2460; top_vs_bottom_half_lift=1.0849; n=248; Scoring is directionally correct on labeled off-anchor (overall and/or MHz-stratified). |
| PASS | `acq_ab_gp_vs_random` — Equal-budget GP(mu) beats random on holdout (top-k true residual + Spearman>0). | gp_lift_vs_random=2.1629; verdict=PASS — GP(mu) top-k true residual > random and Spearman>0; PASS — GP(mu) top-k true residual > random and Spearman>0 |
| UNCERTAIN | `acq_ab_gp_vs_mc` — Equal-budget GP beats MC auto_bad on the same holdout subset. | MC not included (INCLUDE_MC=False) — optional when GPU free |
| PASS | `hole_finding` — GP acquisition enriches true high-error holes vs random (labeled holdout). | pool_enrichment=2.9435; hole_capture_rate=0.5500; hole_capture_lift_vs_chance=2.7473; GP acquisition enriches true holes at equal budget (primary k_80: lift_vs_random=3.18x, pool_enrichment=2.94x, hole_capture=0.55, capture_lift=2.75x). Mid-band structured check also PASS. |
| PASS | `ft_physical_improve` — Fine-tune reduced off-anchor physical p99 MAE vs pre-finetune. | p99_mae_pct=4.9168; p99_mae_delta=-0.4284; n_pre=248; n_post=248; Physical accuracy improved on labeled AL layouts. |
| PASS | `eval_budget_trust` — ECAD ingest covered ≥90% of selected budget (evaluation not starved). | num_ingest_ok=640; num_selected=640; Ingest coverage OK. |
| PASS | `train_off_anchor_logged` — Training off-anchor (layout_cross) metrics were logged for this checkpoint epoch. | Use FG MSE / Pearson trends across cycles as secondary evidence. |

---

## Key numbers (this cycle)

- Pre p99 MAE: 8.7125 → Post: 8.2841 (Δ -0.4284, 4.9%)
- Acquisition Spearman(badness, p99_err): 0.2460 (n=248, ranking_ok=True)
- Pool → ECAD: candidates=12800 selected=640 ingest_ok=640/640
- Acquisition A/B artifact: `active_learning_pi/runs/acquisition_ab_validation_exp059/LATEST_acquisition_ab.json`

---

## Artifacts

- `decision_ledger.json` (machine-readable claims)
- `DECISION_REPORT.md` (this file)
- `iter_0003/eval_cycle_summary.json`
- `iter_0003/acquisition_rank_quality.json`
- `iter_0003/CYCLE_EVAL_REPORT.md`
