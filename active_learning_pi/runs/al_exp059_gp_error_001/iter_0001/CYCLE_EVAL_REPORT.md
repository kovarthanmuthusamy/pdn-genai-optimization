# AL Cycle Evaluation Report — iteration 1

**Run:** `al_exp059_gp_error_001`  
**Generated:** 2026-07-21 09:10 UTC  
**Experiment:** `experiments/exp059_capacity_freq`  
**Acquisition mode:** `gp_error`  
**Off-anchor MHz evaluated:** 155, 250, 265

---

## 0. Decision checklist (see also `DECISION_REPORT.md`)

_Numerical claims are judged after collecting cycle evals + equal-budget A/B. Regenerate with `COMMAND=evaluate-decision`._

**Overall:** **UNCERTAIN** — Incomplete evidence — finish ECAD eval + acquisition A/B before thesis claims. UNCERTAIN: acq_direction, acq_ab_gp_vs_mc

| Status | Claim | Note |
| --- | --- | --- |
| UNCERTAIN | acq_direction | n=23 < min_rank_n=30; direction signal underpowered |
| PASS | acq_ab_gp_vs_random | PASS — GP(mu) top-k true residual > random and Spearman>0 |
| UNCERTAIN | acq_ab_gp_vs_mc | MC not included (INCLUDE_MC=False) — optional when GPU free |
| PASS | ft_physical_improve | Physical accuracy improved on labeled AL layouts. |
| PASS | eval_budget_trust | Ingest coverage OK. |
| PASS | train_off_anchor_logged | Use FG MSE / Pearson trends across cycles as secondary evidence. |

_Machine-readable:_ `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp059_gp_error_001/decision_ledger.json`

---

## 1. Cycle overview

| Metric | Value |
| --- | --- |
| Candidates generated | 1600 |
| Selected for ECAD | 80 |
| Ingested heatmaps (ok/total) | 80 / 80 |
| Overlay samples (cumulative) | 80 |
| Overlay added (last build) | 80 |
| Selection badness range | 0.664399 … 0.725729 |
| K range | 3, 4, 5, 6, 7, 8, 9, 10 |
| Ingested K distribution | {3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 10, 10: 10} |
| Per-K pools | 200 candidates → worst 10 each |
| Total ECAD selected | 80 |
| MHz strata / K (pool) | {'120.0': 30, '150.0': 50, '155.0': 100, '170.0': 40, '180.0': 50, '200.0': 90, '230.0': 90, '250.0': 140, '265.0': 130, '270.0': 80, '280.0': 70, '300.0': 70} |
| ECAD MHz distribution | {'120.0': 1, '150.0': 8, '155.0': 8, '180.0': 8, '200.0': 8, '230.0': 8, '250.0': 8, '265.0': 7, '270.0': 8, '280.0': 8, '300.0': 8} |

---

## 2. Fine-tune
| Metric | Value |
| --- | --- |
| Start epoch | 450 |
| End epoch | 455 |
| Extra epochs | 5 |
| Checkpoint | `experiments/exp059_capacity_freq/checkpoints/last_model.pt` |
| Overlay dataset | `datasets/data_multifreq_al_overlay_exp059` |
| Overlay sample weight | 128.0 |
| Off-anchor eval MHz | 155, 250, 265 |
| Off-anchor MHz weights | — |
| Early stop (off-anchor MSE) | True |
| Early-stop patience | 3 |
| Best off-anchor checkpoint | `/home/ubuntu/genai_pdn/experiments/exp059_capacity_freq/checkpoints/best_off_anchor_model.pt` |
| Val loss (start) | 0.6952 |
| Val loss (end) | 0.8260 |
| Val loss Δ | 0.1308 |
| Train total loss (final ep) | 58.7189 |
| Train heatmap loss (final ep) | 1.0429 |
| Val heatmap loss (final ep) | 0.0611 |

---

## 3. Acquisition ranking quality

_Does acquisition-time **badness** rank true ECAD **p99 abs error**? (positive Spearman ⇒ scoring is directionally correct.)_

| Metric | Value |
| --- | --- |
| n (labeled off-anchor) | 23 |
| Spearman(badness, p99_err) | 0.4012 |
| Pearson(badness, p99_err) | 0.4525 |
| Ranking OK (Spearman > 0) | True |
| Mean p99 err (all labeled) | 13.3339 |
| Mean p99 err (top half badness) | 16.4670 |
| Mean p99 err (bottom half badness) | 10.3543 |
| Top / bottom half lift | 1.5904 |
| n selected ∩ labeled | 23 |
| Mean p99 err (selected) | 13.3339 |
| Selected / all lift | 1.0000 |
| Mean p99 err (top-k by badness) | 13.3339 |

---

## 4. ECAD label evaluation (physical p99)

| Phase | n | p99 MAE | p99 median AE |
| --- | --- | --- | --- |
| Pre-finetune | 23 | 13.3339 | 11.6453 |
| Post-finetune | 23 | 13.1397 | 11.5980 |

### Pre → post change

- **p99 MAE Δ:** -0.1942 (1.5% better vs pre-finetune)
- **p99 median AE Δ:** -0.0473

### Per-layout detail (off-anchor MHz only)

| candidate_id | MHz | real p99 | pre pred | pre abs err | post pred | post abs err |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 155.0 | 3.8079 | 5.0611 | 1.2532 | 5.0086 | 1.2007 |
| 6 | 250.0 | 20.5422 | 6.0432 | 14.4989 | 6.2029 | 14.3392 |
| 57 | 265.0 | 8.4618 | 4.8439 | 3.6179 | 4.6582 | 3.8036 |
| 249 | 155.0 | 3.3300 | 4.7906 | 1.4606 | 4.7094 | 1.3794 |
| 273 | 265.0 | 6.7328 | 4.0033 | 2.7295 | 4.1922 | 2.5407 |
| 390 | 250.0 | 20.1414 | 5.7205 | 14.4208 | 5.8789 | 14.2625 |
| 400 | 250.0 | 18.8152 | 3.6340 | 15.1813 | 4.5555 | 14.2597 |
| 534 | 155.0 | 3.5565 | 4.4963 | 0.9398 | 4.4819 | 0.9255 |
| 667 | 155.0 | 16.2259 | 4.5806 | 11.6453 | 4.6838 | 11.5421 |
| 727 | 250.0 | 36.8648 | 3.1453 | 33.7196 | 3.6200 | 33.2448 |
| 734 | 265.0 | 16.0643 | 4.4802 | 11.5842 | 3.8609 | 12.2034 |
| 841 | 265.0 | 9.1341 | 4.4522 | 4.6819 | 4.5232 | 4.6109 |
| 906 | 250.0 | 33.1821 | 2.1141 | 31.0681 | 2.5846 | 30.5975 |
| 943 | 155.0 | 26.8191 | 3.8283 | 22.9908 | 3.7231 | 23.0960 |
| 1041 | 155.0 | 16.2980 | 4.5266 | 11.7714 | 4.6999 | 11.5980 |
| 1191 | 250.0 | 39.1711 | 2.0914 | 37.0797 | 2.6569 | 36.5142 |
| 1197 | 265.0 | 40.7916 | 3.8822 | 36.9095 | 3.8230 | 36.9686 |
| 1298 | 265.0 | 29.1809 | 4.7843 | 24.3966 | 4.6556 | 24.5253 |
| 1302 | 250.0 | 6.3004 | 2.5647 | 3.7357 | 2.5906 | 3.7097 |
| 1388 | 155.0 | 3.6482 | 4.4617 | 0.8135 | 4.1290 | 0.4808 |
| 1421 | 155.0 | 6.3473 | 3.4261 | 2.9212 | 4.3550 | 1.9923 |
| 1501 | 250.0 | 19.9271 | 4.0006 | 15.9265 | 4.3422 | 15.5849 |
| 1563 | 265.0 | 7.0441 | 3.7111 | 3.3330 | 4.2109 | 2.8332 |

---

## 5. Training validation — off-anchor (`layout_cross`)

_Epoch 455 — from `/home/ubuntu/genai_pdn/experiments/exp059_capacity_freq/metrics/off_anchor_eval.csv`_

| MHz | FG MSE | Pearson | peak loc err | n |
| --- | --- | --- | --- | --- |
| 155.0 | 1.1224 | 0.4642 | 0.0317 | 1600 |
| 250.0 | 1.2417 | 0.3890 | 0.0441 | 1600 |
| 265.0 | 1.1501 | 0.4465 | 0.0362 | 1600 |

---

## 6. Artifacts

- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp059_gp_error_001/iter_0001/acquisition_rank_quality.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp059_gp_error_001/iter_0001/eval_off_anchor_pre_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp059_gp_error_001/iter_0001/eval_off_anchor_post_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp059_gp_error_001/iter_0001/eval_cycle_summary.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp059_gp_error_001/iter_0001/scored_candidates_post_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp059_gp_error_001/iter_0001/CYCLE_EVAL_REPORT.md`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp059_gp_error_001/DECISION_REPORT.md`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp059_gp_error_001/decision_ledger.json`
