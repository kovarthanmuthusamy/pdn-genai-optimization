# Experiment exp044 — encode-first latent (latent-opt aligned)

exp042 architecture + dataset, retrained for **heatmap in latent** (Path A).

## vs exp042

| | exp042 | exp044 |
|---|---|---|
| Dataset | `data_multifreq_norm` (log1p z-score) | **Same** |
| Model | `MultiInputVAEPoeFreq` (freq PoE private 8) | **Same** |
| `layout_train_prob` | **0.92** (layout/sweep) | **0.15** (~85% encode) |
| Cross-freq z | layout z (`cross_freq_layout_z_only`) | **encode z** (`cross_freq_use_encode_z`) |
| Goal | Sweep / layout_cross | **Latent opt** (hm + imp in z) |

## Training paths

**Encode (85%):**
```text
z = PoE(heatmap_encoder(GT), occ, imp, freq_poe)
heatmap_out = decode(z, K, π)
```

**Layout aux (15%):** occasional sweep-style gradient.

**Cross-freq:** always `encode(GT hm, occ, imp)` → decode @ alt MHz.

## Train

```bash
cd /home/ubuntu/gan
.venv/bin/python experiments/exp044/codes/train_vae_simple.py
```

Outputs → `experiments/exp044/runs/run_<UTC>/` (fresh; `resume_checkpoint: null`).

## Joint latent optimization

After a checkpoint exists:

```bash
.venv/bin/python experiments/exp044/codes/latent_optimization_joint.py \
  --checkpoint experiments/exp044/runs/run_<UTC>/checkpoints/checkpoint_epoch_25.pt \
  --sample-index 0 --mhz 200 --steps 200
```

Optimizes `z` with `decode_heatmap(z)` + `decode_impedance(z)` vs targets in **train norm space**.

Physical Ω for plots: `exp(z * log_std + log_mean) - 1` from `normalization_stats.json`.

## Monitor

- Val `heatmap_loss` — should stay ~1–2 (z-score space)
- `off_anchor_eval` `encode_cross` — primary metric for latent-opt decode quality
- `layout_cross` — secondary (only 15% layout training)

## Code map

```text
experiments/exp044/
  config.yaml
  codes/
    train_vae_simple.py      # encode-first config + patches
    run_epoch_encode.py      # cf loss uses encode z
    vae_poe_freq.py          # + decode_heatmap()
    latent_optimization_joint.py
    exp044_eval_common.py
```
