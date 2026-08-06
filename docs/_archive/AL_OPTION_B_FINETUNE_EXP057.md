# Active-learning Option B — exp057 full loop

## One command (recommended)

Edit `pipelines/active_learning/run.py` CONFIG block, then:

```bash
python pipelines/active_learning/run.py
```

Default: `COMMAND = "full"`.

Runs **all 8 steps in sequence**:

| Step | Action |
|------|--------|
| 1 | **Per-K acquire** — for K=3…10: 400 candidates each → worst 10 each (**3200 scored, 80 ECAD**) |
| 2 | (included in step 1) MC infer + uncertainty |
| 3 | (included in step 1) Select worst per K |
| 4 | ECADStar simulate (**80** PI jobs) |
| 5 | Ingest + normalize + **pre-finetune evaluate** |
| 6 | Build overlay dataset (cumulative) |
| 7 | Fine-tune exp057: **resume `last_model.pt` → +50 epochs** |
| Step 8 | Post-finetune eval + **`CYCLE_EVAL_REPORT.md`** |

## Config (`active_learning_pi/config/exp057.json`)

| Key | Value | Notes |
|-----|-------|-------|
| `candidates_per_k` | **400** | Scored per K value (3–10) |
| `worst_per_k` | **10** | ECAD sims per K |
| `k_sweep_per_pool` | **true** | Separate pool per K each cycle |
| `finetune.checkpoint_path` | `last_model.pt` | Always resume latest |
| `finetune.extra_epochs` | **50** | e.g. epoch 1000 → 1050 |
| `evaluation.post_finetune` | **true** | Step 8 after fine-tune in `full` |
| `evaluation.infer_scope` | **ingested** | Re-score only ECAD layouts (faster) |
| `eval_off_anchor_mhz` | **90, 265, 435, 510** | True off-anchor ECAD eval (between training anchors) |

Runtime writes `config_al_finetune.runtime.yaml` with correct `num_epochs` from checkpoint.

### Evaluation outputs (per iteration)

| File | When |
|------|------|
| `eval_off_anchor_pre_finetune.json` | After ingest (step 5) |
| `eval_off_anchor_post_finetune.json` | After fine-tune (step 8) |
| `eval_cycle_summary.json` | Pre vs post + training off-anchor metrics |
| `CYCLE_EVAL_REPORT.md` | Human-readable full cycle report |
| `scored_candidates_post_finetune.json` | Fine-tuned model predictions on ingested layouts |

## Other commands

Edit `COMMAND` in `pipelines/active_learning/run.py`:

| COMMAND | Effect |
|---------|--------|
| `full` | All 8 steps (default) |
| `cycle` | Steps 1–6, no fine-tune |
| `evaluate-post-finetune` | Step 8 only on `ITERATION` |
| `evaluate-full` | Same as `evaluate-post-finetune` |
| `finetune` | Overlay + train only |
| Set `PROPOSE_ONLY = True` | Generate + infer only (no ECAD) |

Fine-tune standalone: `python pipelines/active_learning/finetune_exp057.py`

## Resume behaviour

- Main training `config.yaml`: `resume_checkpoint: last_model.pt` (epoch 1000)
- Each AL fine-tune: reads epoch from `last_model.pt`, trains to **epoch + extra_epochs**
- After fine-tune completes, `last_model.pt` updates → next `full` run continues from new epoch
