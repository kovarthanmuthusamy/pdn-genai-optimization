# exp034 — Impedance Curve Quality: Problem Analysis & Fixes

Inherits all architecture and loss changes from exp033. This experiment addresses three new problems
observed after inspecting the exp033 training metrics and generated samples.

---

## Problems Identified in exp033

### Problem 1 — Physics slope loss caused a learning shock at epoch 50

`_physics_stage_weights` applied a **hard switch**: slope loss was zero for epochs 0–49, then
jumped to full weight (`physics_slope_weight=0.5`) at epoch 50 in a single step.

The physics_losses plot from exp033 showed `physics_slope_loss` spiking to **~12** at epoch 50.
With weight 0.5 this injected **6 units of additional loss** in one step — larger than the entire
impedance reconstruction loss at that point (~0.55). The model was forced to abruptly smooth every
curve it had learned. Peak structure built over 50 epochs was destroyed in a few gradient steps.
After epoch 50 the slope loss returned to ~0, meaning the curves were then smooth — but the decoder
had been trained to avoid sharp peaks, and that bias persisted for the remaining 350 epochs.

**Decision:** Set `physics_slope_weight = 0.0` (off entirely). The slope constraint was empirically
doing more harm than good — real PDN curves can have sharp resonance peaks and the slope limit of
100 Ohm/dec was too restrictive. The `anti_resonance_loss` (weight=0.5) is retained as the sole
shape regularizer. If slope regularization is needed in future experiments it should be introduced
with a long linear anneal (`physics_slope_anneal_epochs`) rather than a hard switch.

---

### Problem 2 — Modality dropout too high, starving the impedance latent

`modality_dropout = 0.4` meant 40% of training batches dropped the impedance encoder entirely.
In those batches the 24 shared latent dims had to represent impedance from heatmap/occupancy alone.
The decoder learned to produce a plausible "average" impedance curve regardless of the impedance
signal — correct overall shape (inductive rise → SRF dip → rising tail) but peak amplitude and
position regressed toward the dataset mean.

This is consistent with the generated samples: the SRF dip location was correct but the
high-frequency region showed noisy small oscillations instead of distinct resonance peaks.

**Fix:** Reduced `modality_dropout` from 0.4 → **0.2** and `modality_dropout_start` from 0.2 → **0.1**.
The impedance encoder is now active in 80% of batches, giving the latent space a reliable
impedance-specific signal.

---

### Problem 3 — Impedance loss weight too low; training not converged at 400 epochs

`impedance_weight = 1.0` vs `heatmap_weight = 2.0` and `occupancy_weight = 3.0`. The impedance
decoder received 2–3× weaker gradient signal than the other two modalities.

The loss_components plot from exp033 showed val impedance loss was **still trending downward at
epoch 400** (0.57 → 0.11, not plateaued), while heatmap and occupancy had already converged.
The training was terminated before impedance reconstruction had fully converged.

**Fixes:**
- Raised `impedance_weight` from 1.0 → **3.0** (equal priority with occupancy)
- Extended `num_epochs` from 400 → **600** to ensure impedance has time to converge

---

## Summary of Config Changes (exp033 → exp034)

| Parameter | exp033 | exp034 | Reason |
|---|---|---|---|
| `impedance_weight` | 1.0 | **3.0** | Equal gradient priority with other modalities |
| `modality_dropout` | 0.4 | **0.2** | Less impedance encoder suppression |
| `modality_dropout_start` | 0.2 | **0.1** | Gentler ramp-in at training start |
| `physics_slope_weight` | 0.5 | **0.0** | Slope loss caused peak destruction at ep50 |
| `num_epochs` | 400 | **600** | Val impedance loss not converged at 400 |

Physics slope anneal logic was also added to `_physics_stage_weights` (linear ramp ep50→150)
as a safety measure — it has no effect since `physics_slope_weight=0.0` but is available for
future experiments to re-enable slope regularization without a hard shock.

---

## Files Changed

| File | Change |
|---|---|
| `codes/train_vae_simple.py` | `impedance_weight` 1.0→3.0; `modality_dropout` 0.4→0.2; `modality_dropout_start` 0.2→0.1; `physics_slope_weight` 0.5→0.0; `physics_slope_anneal_epochs=100` added; `num_epochs` 400→600; `experiment_dir` updated to exp034 |
| `codes/inference_vae.py` | `exp` path and import updated to exp034 |
