# Physics removed + resume from epoch 125

## Changes (`config.yaml`)

| Setting | Before | After |
|---------|--------|-------|
| `physics_ri_weight` | 1.0 | **0.0** |
| `physics_critic_sup_weight` | 2.0 | **0.0** |
| `physics_ar_weight` | 0.5 | **0.0** |
| `experiment_dir` | `experiments/exp043` | **`runs/run_20260616T165326Z`** |
| `resume_checkpoint` | `null` | **125** |

`PhysicsLoss` is not constructed when RI and critic weights are 0 (~10–11 s/epoch faster).

**Still active:** `heatmap_phys_p99_weight` (peak Ω in physical units) — not part of `PhysicsLoss`.

## Resume

- Checkpoint: `runs/run_20260616T165326Z/checkpoints/checkpoint_epoch_125.pt`
- Training continues from **epoch 126** in the same run directory (logs/metrics append).
- Optimizer may skip loading if param groups changed (physics params removed); `reset_lr_on_resume: true` resets LR to `5e-5`.

## Restart command

```bash
cd /home/ubuntu/gan
.venv/bin/python experiments/exp043/codes/train_vae_simple.py
```
