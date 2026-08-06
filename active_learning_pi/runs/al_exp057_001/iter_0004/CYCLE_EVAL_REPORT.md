# AL Cycle Evaluation Report — iteration 4

**Run:** `al_exp057_001`  
**Generated:** 2026-07-13 16:49 UTC  
**Experiment:** `experiments/exp057_structured_graph`  
**Off-anchor MHz evaluated:** 80, 250, 500

---

## 1. Cycle overview

| Metric | Value |
| --- | --- |
| Candidates generated | 384 |
| Selected for ECAD | 48 |
| Ingested heatmaps (ok/total) | 48 / 48 |
| Overlay samples (cumulative) | 88 |
| Overlay added (last build) | 48 |
| Selection badness range | 21.066977 … 83.283298 |

---

## 2. Fine-tune
| Metric | Value |
| --- | --- |
| Start epoch | 1100 |
| End epoch | 1150 |
| Extra epochs | 50 |
| Checkpoint | `experiments/exp057_structured_graph/checkpoints/last_model.pt` |
| Overlay dataset | `datasets/data_multifreq_al_overlay_exp057` |
| Overlay sample weight | 40.0 |
| Val loss (start) | 1.6459 |
| Val loss (end) | 1.5756 |
| Val loss Δ | -0.0704 |
| Train total loss (final ep) | 4.4120 |
| Train heatmap loss (final ep) | 0.1718 |
| Val heatmap loss (final ep) | 0.0043 |

---

## 3. ECAD label evaluation (physical p99)

| Phase | n | p99 MAE | p99 median AE |
| --- | --- | --- | --- |
| Pre-finetune | 36 | 6.9056 | 5.2907 |
| Post-finetune | 36 | 5.9510 | 3.9412 |

### Pre → post change

- **p99 MAE Δ:** -0.9546 (13.8% better vs pre-finetune)
- **p99 median AE Δ:** -1.3496

### Per-layout detail (off-anchor MHz only)

| candidate_id | MHz | real p99 | pre pred | pre abs err | post pred | post abs err |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 250.0 | 20.5430 | 9.7236 | 10.8194 | 9.8878 | 10.6552 |
| 9 | 250.0 | 22.6673 | 6.5842 | 16.0831 | 7.3376 | 15.3297 |
| 18 | 250.0 | 6.1635 | 11.1416 | 4.9781 | 8.1511 | 1.9875 |
| 23 | 250.0 | 24.6452 | 7.6066 | 17.0386 | 10.1656 | 14.4796 |
| 30 | 250.0 | 7.4295 | 11.0006 | 3.5711 | 6.0376 | 1.3919 |
| 33 | 250.0 | 5.5492 | 7.9578 | 2.4086 | 8.1665 | 2.6173 |
| 35 | 250.0 | 8.9790 | 8.4168 | 0.5622 | 8.4330 | 0.5461 |
| 53 | 250.0 | 16.0094 | 6.8534 | 9.1560 | 11.6166 | 4.3928 |
| 55 | 250.0 | 19.5051 | 9.3400 | 10.1652 | 4.5670 | 14.9382 |
| 81 | 250.0 | 20.8392 | 8.1316 | 12.7077 | 13.1617 | 7.6775 |
| 95 | 80.0 | 2.6180 | 9.5878 | 6.9698 | 9.0386 | 6.4206 |
| 96 | 250.0 | 4.9906 | 9.1567 | 4.1661 | 5.2481 | 0.2574 |
| 106 | 250.0 | 22.7915 | 7.2516 | 15.5399 | 8.1544 | 14.6371 |
| 120 | 250.0 | 5.2683 | 7.2510 | 1.9828 | 5.9545 | 0.6863 |
| 124 | 250.0 | 24.0757 | 7.9780 | 16.0977 | 9.1770 | 14.8987 |
| 129 | 250.0 | 10.2365 | 5.9230 | 4.3135 | 13.0130 | 2.7765 |
| 136 | 250.0 | 14.4327 | 8.8294 | 5.6033 | 12.8988 | 1.5339 |
| 144 | 250.0 | 15.3211 | 5.9890 | 9.3321 | 10.1062 | 5.2149 |
| 149 | 250.0 | 11.6434 | 9.6364 | 2.0070 | 13.3546 | 1.7112 |
| 155 | 250.0 | 22.5726 | 6.9211 | 15.6515 | 8.8154 | 13.7572 |
| 156 | 250.0 | 6.8786 | 12.7329 | 5.8543 | 8.6934 | 1.8147 |
| 162 | 250.0 | 10.7564 | 9.4820 | 1.2744 | 11.0467 | 0.2903 |
| 206 | 250.0 | 11.2952 | 9.2677 | 2.0275 | 12.9473 | 1.6521 |
| 246 | 250.0 | 23.5155 | 7.7909 | 15.7247 | 8.9409 | 14.5746 |
| 261 | 250.0 | 9.1485 | 7.5126 | 1.6359 | 9.7531 | 0.6046 |
| 283 | 250.0 | 7.7838 | 9.4136 | 1.6298 | 12.4915 | 4.7077 |
| 296 | 250.0 | 12.5364 | 10.8033 | 1.7332 | 11.4809 | 1.0556 |
| 299 | 250.0 | 14.8805 | 8.1650 | 6.7155 | 6.9616 | 7.9189 |
| 314 | 250.0 | 24.7392 | 10.6164 | 14.1229 | 7.9805 | 16.7588 |
| 324 | 250.0 | 17.0513 | 14.7582 | 2.2931 | 8.1838 | 8.8675 |
| 330 | 250.0 | 18.6140 | 9.7410 | 8.8730 | 13.4113 | 5.2027 |
| 343 | 250.0 | 5.2668 | 9.2838 | 4.0169 | 7.2922 | 2.0254 |
| 359 | 250.0 | 6.6347 | 6.8743 | 0.2396 | 8.4340 | 1.7993 |
| 362 | 250.0 | 15.1715 | 6.4379 | 8.7336 | 8.6747 | 6.4969 |
| 365 | 250.0 | 7.7644 | 8.8283 | 1.0638 | 8.8313 | 1.0668 |
| 377 | 250.0 | 5.5978 | 9.1067 | 3.5089 | 9.0873 | 3.4895 |

---

## 4. Training validation — off-anchor (`layout_cross`)

_Epoch 1150 — from `/home/ubuntu/genai_pdn/experiments/exp057_structured_graph/metrics/off_anchor_eval.csv`_

| MHz | FG MSE | Pearson | peak loc err | n |
| --- | --- | --- | --- | --- |
| 100.0 | 1.4003 | 0.4698 | 0.0337 | 3840 |
| 270.0 | 0.6635 | 0.7486 | 0.0236 | 3840 |
| 400.0 | 0.7347 | 0.6940 | 0.0232 | 3840 |

---

## 5. Artifacts

- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0004/eval_off_anchor_pre_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0004/eval_off_anchor_post_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0004/eval_cycle_summary.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0004/scored_candidates_post_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0004/CYCLE_EVAL_REPORT.md`
