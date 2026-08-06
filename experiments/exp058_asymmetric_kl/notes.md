# Experiment: exp058_asymmetric_kl

## Goal
**Fresh train from scratch** (does **not** load exp057 weights) with:

1. Lower KL (`beta=0.03`) + higher `free_bits=0.12`
2. Asymmetric train near AL path (`layout_p=0.9`, `occ_only=0.7` + distill)
3. Same arch as exp057 (`latent_dim=65`) for recipe comparison
4. AL scoring always occ-only; promote via `layout_cross`

## vs exp057
| Knob | exp057 | exp058 |
|------|--------|--------|
| Init | continued / AL finetune | **from scratch** |
| `beta_final` | ~0.05 | **0.03** |
| `free_bits` | 0.08 | **0.12** |
| Train mix | varied | **asymmetric 0.9 / 0.7** |
| LR (base) | finetune-scale later | **3e-5**, 400 epochs |

## Run order
```bash
# 1) Fresh base train (required first)
bash experiments/exp058_asymmetric_kl/run_train_gpu1.sh

# 2) Optional short occ-only align (after last_model.pt exists)
COMMAND=occ-warmup python pipelines/active_learning/run.py

# 3) AL cycles
COMMAND=full python pipelines/active_learning/run.py
```

`checkpoints/` starts empty. Do not copy exp057 `last_model.pt` into this run.
