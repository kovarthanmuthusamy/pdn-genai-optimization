# Decision report — `al_exp058_001` iter 1

**Generated:** 2026-07-20 18:54:43 UTC  
**Experiment:** `experiments/exp058_asymmetric_kl`  
**Acquisition mode:** `mc`  
**Overall:** **FAIL**  

Do not claim improvement / correct uncertainty until FAIL claims are resolved: ft_physical_improve

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
| PASS | `acq_direction` — Acquisition / uncertainty scores rank true ECAD error (Spearman > 0). | spearman_badness_vs_p99_err=0.0661; top_vs_bottom_half_lift=1.2102; n=248; Scoring is directionally correct on labeled off-anchor. |
| UNCERTAIN | `acq_ab_gp_vs_random` — Equal-budget GP(mu) beats random on holdout (top-k true residual + Spearman>0). | No acquisition A/B JSON linked — run validate_acquisition_ab.py |
| FAIL | `ft_physical_improve` — Fine-tune reduced off-anchor physical p99 MAE vs pre-finetune. | p99_mae_pct=-0.6251; p99_mae_delta=0.0474; n_pre=248; n_post=248; Physical p99 MAE did not improve — FT claim not supported on this cycle. |
| PASS | `eval_budget_trust` — ECAD ingest covered ≥90% of selected budget (evaluation not starved). | num_ingest_ok=640; num_selected=640; Ingest coverage OK. |
| PASS | `train_off_anchor_logged` — Training off-anchor (layout_cross) metrics were logged for this checkpoint epoch. | Use FG MSE / Pearson trends across cycles as secondary evidence. |

---

## Key numbers (this cycle)

- Pre p99 MAE: 7.5833 → Post: 7.6307 (Δ 0.0474, -0.6%)
- Acquisition Spearman(badness, p99_err): 0.0661 (n=248, ranking_ok=True)
- Pool → ECAD: candidates=12800 selected=640 ingest_ok=640/640

---

## Artifacts

- `decision_ledger.json` (machine-readable claims)
- `DECISION_REPORT.md` (this file)
- `iter_0001/eval_cycle_summary.json`
- `iter_0001/acquisition_rank_quality.json`
- `iter_0001/CYCLE_EVAL_REPORT.md`
