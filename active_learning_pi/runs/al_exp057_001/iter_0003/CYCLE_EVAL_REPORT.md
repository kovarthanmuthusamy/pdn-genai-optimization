# AL Cycle Evaluation Report — iteration 3

**Run:** `al_exp057_001`  
**Generated:** 2026-07-13 14:59 UTC  
**Experiment:** `experiments/exp057_structured_graph`  
**Off-anchor MHz evaluated:** 80, 250, 500

---

## 1. Cycle overview

| Metric | Value |
| --- | --- |
| Candidates generated | 96 |
| Selected for ECAD | 16 |
| Ingested heatmaps (ok/total) | 16 / 16 |
| Overlay samples (cumulative) | 40 |
| Overlay added (last build) | 16 |
| Selection badness range | 16.746939 … 63.495316 |

---

## 2. Fine-tune
| Metric | Value |
| --- | --- |
| Start epoch | 1050 |
| End epoch | 1100 |
| Extra epochs | 50 |
| Checkpoint | `experiments/exp057_structured_graph/checkpoints/last_model.pt` |
| Overlay dataset | `datasets/data_multifreq_al_overlay_exp057` |
| Overlay sample weight | 40.0 |
| Val loss (start) | 1.6794 |
| Val loss (end) | 1.6459 |
| Val loss Δ | -0.0335 |
| Train total loss (final ep) | 4.6913 |
| Train heatmap loss (final ep) | 0.1800 |
| Val heatmap loss (final ep) | 0.0052 |

---

## 3. ECAD label evaluation (physical p99)

| Phase | n | p99 MAE | p99 median AE |
| --- | --- | --- | --- |
| Pre-finetune | 10 | 7.5938 | 7.6916 |
| Post-finetune | 10 | 7.6104 | 6.9400 |

### Pre → post change

- **p99 MAE Δ:** 0.0166 (0.2% worse vs pre-finetune)
- **p99 median AE Δ:** -0.7516

### Per-layout detail (off-anchor MHz only)

| candidate_id | MHz | real p99 | pre pred | pre abs err | post pred | post abs err |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | 250.0 | 22.7942 | 7.8234 | 14.9709 | 6.9191 | 15.8752 |
| 14 | 250.0 | 5.9005 | 8.3182 | 2.4177 | 5.9660 | 0.0655 |
| 49 | 250.0 | 16.7518 | 8.9180 | 7.8338 | 4.4831 | 12.2687 |
| 50 | 250.0 | 4.4757 | 7.0905 | 2.6149 | 8.9872 | 4.5115 |
| 54 | 250.0 | 19.7726 | 7.3923 | 12.3803 | 8.4296 | 11.3430 |
| 58 | 250.0 | 20.7267 | 8.4707 | 12.2560 | 5.9476 | 14.7791 |
| 66 | 250.0 | 13.7054 | 6.1559 | 7.5494 | 5.5973 | 8.1081 |
| 74 | 80.0 | 2.1489 | 10.1205 | 7.9716 | 7.9208 | 5.7719 |
| 82 | 250.0 | 6.6079 | 7.6313 | 1.0235 | 7.3702 | 0.7623 |
| 89 | 250.0 | 13.4576 | 6.5375 | 6.9200 | 10.8385 | 2.6191 |

---

## 4. Training validation — off-anchor (`layout_cross`)

_Epoch 1100 — from `/home/ubuntu/genai_pdn/experiments/exp057_structured_graph/metrics/off_anchor_eval.csv`_

| MHz | FG MSE | Pearson | peak loc err | n |
| --- | --- | --- | --- | --- |
| 100.0 | 1.2763 | 0.4745 | 0.0338 | 3840 |
| 270.0 | 0.7136 | 0.7347 | 0.0240 | 3840 |
| 400.0 | 0.7685 | 0.6775 | 0.0234 | 3840 |

---

## 5. Artifacts

- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0003/eval_off_anchor_pre_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0003/eval_off_anchor_post_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0003/eval_cycle_summary.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0003/scored_candidates_post_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0003/CYCLE_EVAL_REPORT.md`
