# Decision report — `al_exp059_gp_error_001` iter 2

**Generated:** 2026-07-21 10:19:17 UTC  
**Experiment:** `experiments/exp059_capacity_freq`  
**Acquisition mode:** `gp_error`  
**Overall:** **FAIL**  

Do not claim improvement / correct uncertainty until FAIL claims are resolved: acq_direction

---

## How to use this report

Every workflow stage must leave **numerical** artifacts. This ledger turns them into
claims you can accept or reject:

1. **acq_direction** — is uncertainty / GP scoring pointing at true error?
2. **acq_ab_gp_vs_random** — equal-budget GP vs random (and MC if present)?
3. **ft_physical_improve** — did fine-tune actually improve physical p99?
4. **eval_budget_trust** — was the ECAD budget fully evaluated?
5. **train_off_anchor_logged** — did training log off-anchor numbers?

Only claim a thesis result when blocking claims are **PASS** (or explicitly scoped).

---

## Claim checklist

| Status | Claim | Key evidence |
| --- | --- | --- |
| FAIL | `acq_direction` — Acquisition / uncertainty scores rank true ECAD error (Spearman > 0). | spearman_badness_vs_p99_err=-0.0716; top_vs_bottom_half_lift=0.8028; n=247; Scoring does NOT beat chance on labeled off-anchor — do not trust AL selection. |
| PASS | `acq_ab_gp_vs_random` — Equal-budget GP(mu) beats random on holdout (top-k true residual + Spearman>0). | gp_lift_vs_random=2.7134; verdict=PASS — GP(mu) top-k true residual > random and Spearman>0; PASS — GP(mu) top-k true residual > random and Spearman>0 |
| UNCERTAIN | `acq_ab_gp_vs_mc` — Equal-budget GP beats MC auto_bad on the same holdout subset. | MC not included (INCLUDE_MC=False) — optional when GPU free |
| PASS | `ft_physical_improve` — Fine-tune reduced off-anchor physical p99 MAE vs pre-finetune. | p99_mae_pct=4.1741; p99_mae_delta=-0.4371; n_pre=247; n_post=247; Physical accuracy improved on labeled AL layouts. |
| PASS | `eval_budget_trust` — ECAD ingest covered ≥90% of selected budget (evaluation not starved). | num_ingest_ok=640; num_selected=640; Ingest coverage OK. |
| PASS | `train_off_anchor_logged` — Training off-anchor (layout_cross) metrics were logged for this checkpoint epoch. | Use FG MSE / Pearson trends across cycles as secondary evidence. |

---

## Key numbers (this cycle)

- Pre p99 MAE: 10.4728 → Post: 10.0356 (Δ -0.4371, 4.2%)
- Acquisition Spearman(badness, p99_err): -0.0716 (n=247, ranking_ok=False)
- Pool → ECAD: candidates=12800 selected=640 ingest_ok=640/640
- Acquisition A/B artifact: `/home/ubuntu/genai_pdn/active_learning_pi/runs/acquisition_ab_validation_exp059/LATEST_acquisition_ab.json`

---

## Artifacts

- `decision_ledger.json` (machine-readable claims)
- `DECISION_REPORT.md` (this file)
- `iter_0002/eval_cycle_summary.json`
- `iter_0002/acquisition_rank_quality.json`
- `iter_0002/CYCLE_EVAL_REPORT.md`
