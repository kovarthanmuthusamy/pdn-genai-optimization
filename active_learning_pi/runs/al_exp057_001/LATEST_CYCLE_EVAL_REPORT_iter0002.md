# AL Cycle Evaluation Report — iteration 2

**Run:** `al_exp057_001`  
**Generated:** 2026-07-13 13:03 UTC  
**Experiment:** `experiments/exp057_structured_graph`  
**Off-anchor MHz evaluated:** 80, 250, 500

---

## 1. Cycle overview

| Metric | Value |
| --- | --- |
| Candidates generated | 96 |
| Selected for ECAD | 16 |
| Ingested heatmaps (ok/total) | 16 / 16 |
| Overlay samples (cumulative) | 24 |
| Overlay added (last build) | 0 |
| Selection badness range | 35.261315 … 158.118609 |

---

## 2. Fine-tune
| Metric | Value |
| --- | --- |
| Start epoch | 1000 |
| End epoch | 1050 |
| Extra epochs | 50 |
| Checkpoint | `experiments/exp057_structured_graph/checkpoints/last_model.pt` |
| Overlay dataset | `datasets/data_multifreq_al_overlay_exp057` |
| Overlay sample weight | 40.0 |
| Val loss (start) | 1.8361 |
| Val loss (end) | 1.6794 |
| Val loss Δ | -0.1567 |
| Train total loss (final ep) | 4.5482 |
| Train heatmap loss (final ep) | 0.1815 |
| Val heatmap loss (final ep) | 0.0049 |

---

## 3. ECAD label evaluation (physical p99)

| Phase | n | p99 MAE | p99 median AE |
| --- | --- | --- | --- |
| Pre-finetune | 8 | 7.5399 | 4.7624 |
| Post-finetune | 8 | 8.1935 | 5.1091 |

### Pre → post change

- **p99 MAE Δ:** 0.6536 (8.7% worse vs pre-finetune)
- **p99 median AE Δ:** 0.3468

### Per-layout detail (off-anchor MHz only)

| candidate_id | MHz | real p99 | pre pred | pre abs err | post pred | post abs err |
| --- | --- | --- | --- | --- | --- | --- |
| 18 | 250.0 | 16.9538 | 6.8887 | 10.0652 | 3.7501 | 13.2037 |
| 22 | 250.0 | 31.9765 | 5.9869 | 25.9896 | 4.5175 | 27.4590 |
| 36 | 250.0 | 7.5024 | 6.7865 | 0.7159 | 8.4836 | 0.9812 |
| 39 | 250.0 | 10.1283 | 6.6903 | 3.4380 | 5.2037 | 4.9247 |
| 56 | 250.0 | 13.6662 | 4.8485 | 8.8177 | 5.3909 | 8.2753 |
| 64 | 80.0 | 2.0939 | 8.1805 | 6.0867 | 7.3875 | 5.2936 |
| 74 | 250.0 | 4.4069 | 6.5608 | 2.1539 | 6.1190 | 1.7121 |
| 76 | 250.0 | 10.2640 | 7.2118 | 3.0522 | 6.5653 | 3.6986 |

---

## 4. Training validation — off-anchor (`layout_cross`)

_Epoch 1050 — from `/home/ubuntu/genai_pdn/experiments/exp057_structured_graph/metrics/off_anchor_eval.csv`_

| MHz | FG MSE | Pearson | peak loc err | n |
| --- | --- | --- | --- | --- |
| 100.0 | 1.2573 | 0.4630 | 0.0341 | 3840 |
| 270.0 | 0.7295 | 0.7322 | 0.0244 | 3840 |
| 400.0 | 0.8013 | 0.6590 | 0.0242 | 3840 |

---

## 5. Artifacts

- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0002/eval_off_anchor_pre_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0002/eval_off_anchor_post_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0002/eval_cycle_summary.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0002/scored_candidates_post_finetune.json`
- `/home/ubuntu/genai_pdn/active_learning_pi/runs/al_exp057_001/iter_0002/CYCLE_EVAL_REPORT.md`
