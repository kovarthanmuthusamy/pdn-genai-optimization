# Decision report — `al_exp059_gp_error_001` iter 1

**Generated:** 2026-07-21 09:10:00 UTC  
**Experiment:** `experiments/exp059_capacity_freq`  
**Acquisition mode:** `gp_error`  
**Overall:** **UNCERTAIN**  

Incomplete evidence — finish ECAD eval + acquisition A/B before thesis claims. UNCERTAIN: acq_direction, acq_ab_gp_vs_mc

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
| UNCERTAIN | `acq_direction` — Acquisition / uncertainty scores rank true ECAD error (Spearman > 0). | n=23; n=23 < min_rank_n=30; direction signal underpowered |
| PASS | `acq_ab_gp_vs_random` — Equal-budget GP(mu) beats random on holdout (top-k true residual + Spearman>0). | gp_lift_vs_random=2.7134; verdict=PASS — GP(mu) top-k true residual > random and Spearman>0; PASS — GP(mu) top-k true residual > random and Spearman>0 |
| UNCERTAIN | `acq_ab_gp_vs_mc` — Equal-budget GP beats MC auto_bad on the same holdout subset. | MC not included (INCLUDE_MC=False) — optional when GPU free |
| PASS | `ft_physical_improve` — Fine-tune reduced off-anchor physical p99 MAE vs pre-finetune. | p99_mae_pct=1.4562; p99_mae_delta=-0.1942; n_pre=23; n_post=23; Physical accuracy improved on labeled AL layouts. |
| PASS | `eval_budget_trust` — ECAD ingest covered ≥90% of selected budget (evaluation not starved). | num_ingest_ok=80; num_selected=80; Ingest coverage OK. |
| PASS | `train_off_anchor_logged` — Training off-anchor (layout_cross) metrics were logged for this checkpoint epoch. | Use FG MSE / Pearson trends across cycles as secondary evidence. |

---

## Key numbers (this cycle)

- Pre p99 MAE: 13.3339 → Post: 13.1397 (Δ -0.1942, 1.5%)
- Acquisition Spearman(badness, p99_err): 0.4012 (n=23, ranking_ok=True)
- Pool → ECAD: candidates=1600 selected=80 ingest_ok=80/80
- Acquisition A/B artifact: `/home/ubuntu/genai_pdn/active_learning_pi/runs/acquisition_ab_validation_exp059/LATEST_acquisition_ab.json`

---

## Artifacts

- `decision_ledger.json` (machine-readable claims)
- `DECISION_REPORT.md` (this file)
- `iter_0001/eval_cycle_summary.json`
- `iter_0001/acquisition_rank_quality.json`
- `iter_0001/CYCLE_EVAL_REPORT.md`
