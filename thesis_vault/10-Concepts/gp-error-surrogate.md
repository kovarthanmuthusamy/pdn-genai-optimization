---
title: gp-error-surrogate
type: concept
source: docs/gp-error-surrogate.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/gp-error-surrogate.md` — edit the source file, then re-run `tools/build_vault.py`.

# GP error-surrogate active learning

Thesis-oriented summary of **latent-space error-GP** acquisition: predict *where the VAE is wrong*, rather than trusting the VAE’s own uncertainty.

**Full framework (metrics, equations, validation tables):** `../active_learning_pi/GP_ERROR_SURROGATE_FRAMEWORK.md` (`../active_learning_pi/GP_ERROR_SURROGATE_FRAMEWORK.md`)  
**Code:** `active_learning_pi/al/gp_error_surrogate_test.py`  
**Empirically exercised on:** exp059 (controls on exp058)

For the production MC-uncertainty AL loop (Option B), see [[active-learning|active-learning.md]].

---

## 1. Motivation (falsified baseline)

MC dropout / latent-resampling uncertainty on a smooth VAE was found to be **weakly correlated** with true ECAD error (reported Spearman ~0.07 in the framework doc). Self-reported variance stays small even in blind spots.

**Design principle:** fit a second model (Gaussian Process / SVGP) on **residuals** between VAE predictions and ECAD on labeled layouts:

\[
a(x)=\mu_e(x)+\kappa\,\sigma_e(x)
\]

- \(\mu_e\): expected error (systematic failure)
- \(\sigma_e\): uncertainty of that estimate (sparsity / holes)

---

## 2. Offline → online workflow

1. **Offline:** on labeled layouts, encode features \(x\), compute scalar error \(y\) (e.g. structural \(1-\rho\), MAE/MSE), fit SVGP \(x\mapsto(\mu_e,\sigma_e)\). Validate holdout Spearman(\(\mu_e\), true \(y\)) and \(\sigma_e\) calibration.
2. **Online AL:** sample a large candidate pool (never enumerate \(2^{52}\)); form cheap \(x\); rank by UCB; cluster/dedupe; ECAD top-\(M\); ingest; fine-tune VAE; **refit GP** (finetune shifts latents).
3. **Downstream:** optional trust penalty in latent opt; confidence overlays on peaks.

---

## 3. Feature path (critical)

At AL time there is **no** ECAD heatmap, so full-encode \(z\) is unavailable.

| Path | Feature \(x\) | Cost | Role |
|------|---------------|------|------|
| `full` | encode(real heatmap) | Needs ECAD | Upper bound (leaky for AL) |
| `occ` | layout encode(occ) | Cheap | Deployed path — error often **not** smooth in \(z\) |
| `pred` | encode(VAE’s own predicted heatmap) | Cheap | **Self-prediction bootstrap** — recovers much of full-path signal |

Thesis claim should specify which path was used for acquisition.

---

## 4. Error targets \(y\)

Prefer smooth, GP-friendly scalars in normalized space:

| Target | Use |
|--------|-----|
| `struct` (\(1-\mathrm{Pearson}\)) | Shape-first ranking |
| `mae` / `mse` | Full-field error (good “whole modality” signal) |
| Soft / sharp peak distance | Peak KPI (smoother than hard argmax) |

Hard peak-pixel distance alone was weakly learnable — avoid as sole GP target.

---

## 5. Relation to Option-B MC AL

| Approach | Status in repo | Strength |
|----------|----------------|----------|
| MC uncertainty (Option B) | Default (`acquisition_mode` unset / `"mc"`) | Simple; implemented end-to-end with ECAD |
| Error-GP UCB (`pred`+SVGP) | **Opt-in** via `acquisition_mode: "gp_error"` | Addresses “confidently wrong” regions if residuals are informative |
| Random | Opt-in `acquisition_mode: "random"` (`exp057_random.json`) | Equal-budget control for thesis |

**Production opt-in config:** `../active_learning_pi/config/exp059_gp_error.json` (`../active_learning_pi/config/exp059_gp_error.json`)  
**Random control:** `../active_learning_pi/config/exp059_random.json` (`../active_learning_pi/config/exp059_random.json`)  
**Entry:** `pipelines/active_learning/run.py` → `CONFIG_PATH` defaults to exp059 GP.

Mitigations implemented:

| Gap | Mitigation |
|-----|------------|
| No proof GP beats random | [[validate_acquisition_ab|`validate_acquisition_ab.py`]] — labeled holdout A/B (**defaults to exp059**) |
| Self-pred misses true holes | Optional novelty backstop (`score_mode: ucb_novelty`); **default off** until A/B says otherwise |
| σ poorly calibrated | Default production `score_mode: "mu"` (expected residual), not UCB |
| Heatmap MSE ≠ physical p99 | Default GP target **`p99_ae`** (physical FG p99 abs error) aligned with `acq_direction` |
| Broken pred_p99 denorm | AL inference uses `engine.denorm_heatmap_physical(mhz=…)` (per-MHz), not legacy global log-z |
| GP must refit each cycle | Artifact cleared at start of each `per_k_acquire` |
| Capacity train has occ=0 | After capacity `last_model.pt`, AL FT uses `config_al_finetune.yaml` (layout 0.9 / occ 0.7) |

**Prior smoke A/B (exp057):** `gp_mu_only` beat random. **Re-run on exp059** before thesis claims.

### Decision reporting (always-on)

Every AL cycle must leave numerical artifacts and end with a **decision report**:

| Stage | Numbers | Artifact |
|-------|---------|----------|
| Acquire | badness / GP μ scores | `scored_candidates.json` |
| Pre-FT | p99 MAE + Spearman(score, err) | `eval_off_anchor_pre_finetune.json`, `acquisition_rank_quality.json` |
| Equal-budget A/B | GP vs random (vs MC) top-k residual | `LATEST_acquisition_ab.json` |
| Post-FT | p99 Δ + training off-anchor | `eval_cycle_summary.json`, `CYCLE_EVAL_REPORT.md` |
| Decide | PASS/FAIL/UNCERTAIN claims | `DECISION_REPORT.md`, `decision_ledger.json` |

Regenerate: `COMMAND=evaluate-decision` in `pipelines/active_learning/run.py`.

```bash
# Equal-budget falsification (no new ECAD) — edit CONFIG in the script first
python active_learning_pi/al/validate_acquisition_ab.py

# Point pipelines/active_learning/run.py CONFIG_PATH at exp057_gp_error.json
# Prefer PROPOSE_ONLY / cycle-without-ECAD first to inspect scores.
```

Implementation: [[gp_error_surrogate|`../active_learning_pi/al/gp_error_surrogate.py`]] + hook in [[inference_pool|`../active_learning_pi/al/inference_pool.py`]].

**Recommended thesis stance:** treat GP-AL as a **methodological extension** with validation metrics; compare equal-budget MC vs GP vs random acquisition before claiming superiority. Do not conflate GP cycles with completed Option-B cycle results unless both were run under the same protocol.

---

## 6. Pros / cons

| Pros | Cons |
|------|------|
| Targets actual residual structure | Needs enough labeled residuals to fit |
| \(\mu_e+\sigma_e\) separates bias vs sparsity | Sensitive to feature path (`occ` vs `pred`) |
| Compatible with layout-holdout hygiene (exp059) | Extra compute; must refit after each VAE FT |
| Clear falsification (Spearman vs true error) | Not a substitute for ECAD verification of selected batches |


## Implemented by

- [[gp_error_surrogate]] — `active_learning_pi/al/gp_error_surrogate.py`
- [[validate_acquisition_ab]] — `active_learning_pi/al/validate_acquisition_ab.py`
- [[per_k_acquire]] — `active_learning_pi/al/per_k_acquire.py`
