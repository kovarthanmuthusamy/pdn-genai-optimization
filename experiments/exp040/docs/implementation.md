# exp040 — Factorized PI-frequency heatmap VAE

## Goal

Improve **off-anchor MHz** PI-Distribution (spatial heatmaps) when training data exists only at **discrete anchor frequencies**, for the workflow:

1. Latent search on occupancy + impedance → PI **spectrum** vs Hz  
2. Pick peak MHz from the spectrum  
3. Generate **spatial PI map** at those MHz for ECAD / PDN reference  

exp039 conditions the full heatmap decoder on `(z_layout, K, PI_freq)`. exp040 **factorizes frequency**:

```text
H(x,y | z, K, MHz) ≈ Σ_k α_k(MHz) · B_k(x,y | z, K)  [+ residual]
```

- **B_k**: spatial modes from layout (shared across MHz for one design)  
- **α_k**: mixing weights from **PI_freq only** (smooth in log-Hz)  
- **Residual** (optional): small correction using the legacy exp038 decode path  

## Repository layout

| Path | Role |
|------|------|
| `codes/vae_factorized_freq.py` | `FactorizedFreqVAE` model |
| `codes/synthetic_freq_blend.py` | Between-anchor training targets (blend GT maps) |
| `codes/train_vae_simple.py` | Training (patches exp038 loop) |
| `codes/inference_vae.py` | `VAEInference` for sweeps / scripts |
| `config.yaml` | Hyperparameters |
| `docs/implementation.md` | This file |

Shared utilities (dataloader, losses, eval) remain under `experiments/exp038_true_multi/codes/`.

### Heatmap z clipping (log1p + z-score)

Training and inference clip normalized heatmaps to `Heatmap.clip_min` / `clip_max` from
`normalization_stats.json` (~tail trim; full stored range is wider). Targets are clipped at
dataset load; reconstructions are clipped in `heatmap_loss` and before `exp` at inference so
`exp(z)` cannot blow up. No dataset rebuild required.

## Model (`FactorizedFreqVAE`)

### Encoder (unchanged)

- Heatmap expert: `K` + **PI_freq**  
- Occupancy / impedance experts: **K** only  
- PoE fusion → `z`, same as exp038  

### Heatmap decoder (new)

1. **Basis trunk** `basis_fc` + deconvs: input `[z, K_emb]` — **no PI_freq**  
2. Output **M** channels → `B_1…B_M` at 64×64  
3. **`freq_alpha_head(FreqConditioner(PI_freq))` → α** shape `(B, M)`  
4. **Mix:** `H = einsum('bm,bmhw->bhw', α, B)`  
5. If `use_freq_residual`: add `residual_gain * legacy_decode(z, K, PI_freq)`  

### Inference modes

Same API as exp038: `model.inference(..., mode="layout"|"anchor_blend"|...)`.  
`decode_heatmap_blended` still log-blends maps between bracketing anchors.

### Regularizer

`mode_orthogonality_weight` × MSE(**Gram** of normalized modes, **I**) encourages diverse spatial bases.

## Training extras

### Synthetic mid-MHz blends (`synthetic_blend_prob`)

When the dataloader provides `heatmap_norm_alt` / `PI_freq_alt` (same layout, other anchor):

- Sample `t ~ U(0,1)`  
- Target map ← `(1-t)·HM_native + t·HM_alt` (normalized log space)  
- `PI_freq` ← linear blend of norm scalars  

No new ECAD; teaches off-anchor MHz on the **layout** path.

### Layout cross-freq (`cross_freq_layout_z_only`)

Cross-frequency loss uses `encode_layout_latent` (same as multifreq sweep).

## Configuration (key fields)

| Key | Default | Meaning |
|-----|---------|---------|
| `num_freq_modes` | 6 | Number of spatial modes M |
| `freq_alpha_softmax` | true (phase 3) | If true, α = softmax (else linear α; can be negative) |
| `use_freq_residual` | false (phase 3) | If false, train/decode factorized path only; legacy decoder frozen |
| `residual_gain_init` | 0.15 | Initial scale of residual branch |
| `mode_orthogonality_weight` | 0.05 | Mode diversity loss |
| `synthetic_blend_prob` | 0.25 | Fraction of batches with blended targets |
| `layout_train_prob` | 0.92 | Train with layout z (inference-aligned) |

See `config.yaml` for full schedule / loss weights (aligned with exp039 phase-2).

### Curriculum epochs and resume

Phase start epochs (`heatmap_focus_start_epoch`, `impedance_peak_focus_epoch`, etc.) are computed from `*_frac × num_epochs` **only on a fresh run**.

On **resume** (default `recalculate_curriculum_on_resume: false`), thresholds are **restored from the checkpoint’s saved config**, so increasing `num_epochs` from 200→350 does **not** move heatmap-focus from epoch 100→175 mid-run.

To force the old behaviour (re-scale all `*_frac` against the new `num_epochs`), set `recalculate_curriculum_on_resume: true`.

You can also pin phases with explicit integers in yaml (overrides frac), e.g. `"heatmap_focus_start_epoch": 100`.

## Commands

```bash
cd ~/gan

# Train (fresh exp040 — do not expect full weight load from exp039)
python experiments/exp040/codes/train_vae_simple.py

# Optional: fine-tune from exp039 (partial load — new basis / alpha heads random)
# Set in config.yaml: "resume_checkpoint": "experiments/exp039_improved_heatmap/checkpoints/last_model.pt"

# Multifreq sweep (point EXPERIMENT_DIR at exp040 in run_multifreq_heatmap_sweep.py)
python scrap/generation/run_multifreq_heatmap_sweep.py

# Off-anchor eval (exp038 script + exp040 checkpoint path)
python experiments/exp038_true_multi/codes/eval_cross_freq.py --ckpt experiments/exp040/checkpoints/last_model.pt
```

**Note:** `run_multifreq_heatmap_sweep.py` imports `{EXPERIMENT_DIR}.codes.inference_vae`. Set `EXPERIMENT_DIR = "experiments/exp040"` at the top of that script (or add exp040 `inference_vae` to PYTHONPATH).

## Checkpoint compatibility

| Component | Load from exp039 |
|-----------|------------------|
| Encoders, occ/imp decoders, K/freq conditioners | Yes (shapes match) |
| `basis_*` decoder, `freq_alpha_head` | **New** — random init |
| `heatmap_dec_deconv2` (1 ch → M ch) | **Not** compatible |

Recommended: train exp040 from scratch or short fine-tune after exp039 with low LR.

## Evaluation

Use the same metrics as exp039:

- `layout_cross` / `encode_cross` at 100, 175, 350 MHz  
- `evaluate_vae.py` / `visualize_latent.py` (copy from exp039 and set `FactorizedFreqVAE` import)  
- ECAD compare on sweep folders  

Success criterion: lower **`layout_cross`** between anchors vs exp039 at same epoch budget, especially 100–175 MHz.

## Limitations

- Does not replace the need for **extra anchor simulations** at MHz where peaks cluster.  
- Synthetic blends are **approximate** physics; use for training only.  
- Very high MHz (e.g. 350) may still need `anchor_blend` or more anchors.  

## References

- exp038 / exp039: multifreq `PI_freq` VAE baseline  
- Conversation design: frequency factorization (option 2) for limited multifreq data  
