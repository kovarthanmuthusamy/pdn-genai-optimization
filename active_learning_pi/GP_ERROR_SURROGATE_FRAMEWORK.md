# Latent-Space Error-GP Active Learning for PDN VAEs

**Framework document** — end-to-end process, design rationale, metrics, and validation  
**Scope:** heatmap (and extensible multi-modality) uncertainty acquisition without enumerating \(2^{52}\) layouts  
**Primary code:** `active_learning_pi/al/gp_error_surrogate_test.py`  
** empirically validated on:** `experiments/exp059_capacity_freq` (and controls on `exp058`)

---

## 0. Problem statement

### 0.1 What we need

We train a multi-input VAE on ~24k layouts × 24 PI-frequency anchors. The combinatorial layout space is \(2^{52}\). After training, the model is wrong in some regions (especially mid-band heatmaps, high-K peak flips, and future unseen decap types). We want **active learning (AL)** to:

1. Score a large *sampled* candidate pool cheaply (no ECAD).
2. Select a small batch of the *worst* candidates for ECAD simulation.
3. Fine-tune on those labels and repeat.

### 0.2 Why the old approach fails

The previous AL signal was **single-model Monte-Carlo uncertainty** (dropout / latent resampling → `peak_loc_spread`, `heatmap_mc_sample_mse` in `active_learning_pi/al/robust_stats.py`). Empirical acquisition-rank quality was ~Spearman **0.07** vs true ECAD error — essentially uninformative.

**Reason:** a continuous VAE is smooth by construction. Its *own* predictive variance is small and roughly constant across input space, including in the blind spots we care about. Self-reported confidence cannot find “confidently wrong” holes.

### 0.3 Design principle

> Do **not** ask the VAE “how uncertain are you?”  
> Instead, train a **second model** (a Gaussian Process) that predicts **where the VAE is wrong**, using residuals we already have on labeled layouts.

The GP’s predictive mean \(\mu_e\) = expected VAE error (systematic bias).  
The GP’s predictive std \(\sigma_e\) = uncertainty about that error estimate (data sparsity / holes).

Acquisition = Upper Confidence Bound:

\[
a(x) = \mu_e(x) + \kappa\,\sigma_e(x)
\]

---

## 1. End-to-end workflow (top → bottom)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE A — Offline: build the error surrogate (no new ECAD)             │
│                                                                         │
│  Labeled layouts (24k × anchors)                                        │
│       │                                                                 │
│       ├─► VAE forward (occ or pred path) → heatmap / impedance pred     │
│       ├─► Compare to ECAD ground truth → scalar error y                 │
│       └─► Form GP input feature x  (latent z, or pseudo-full-z)         │
│                                                                         │
│  Fit SVGP:  x → (μ_e, σ_e)                                              │
│  Validate: holdout Spearman(μ_e, true y); σ calibration & OOD elev.     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE B — Online AL cycle                                              │
│                                                                         │
│  Sample candidate pool (e.g. 200k–1M random manufacturable layouts)     │
│       │  (never enumerate 2^52)                                         │
│       ▼                                                                 │
│  Cheap feature x via Self-Prediction Bootstrap (Section 4)              │
│       │                                                                 │
│       ▼                                                                 │
│  SVGP → μ_e, σ_e  →  a = μ_e + κ σ_e                                    │
│       │                                                                 │
│       ▼                                                                 │
│  Cluster / dedupe → top-M → ECADStar → ingest labels → finetune VAE     │
│       │                                                                 │
│       ▼                                                                 │
│  Re-encode labeled set → refit SVGP (VAE finetune shifts z)             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE C — Downstream uses of the same GP                               │
│                                                                         │
│  • Latent optimization trust penalty: L_mask(z) + λ σ_e(z)              │
│  • Peak / design visualization: attach σ_e as confidence                │
│  • Decap-type extension: new type → high σ_e → AL auto-prioritizes     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. What the GP measures (and what it does not)

### 2.1 Inputs \(x\) and target \(y\)

| Symbol | Meaning in this project | Why |
|--------|-------------------------|-----|
| \(x\) | Feature of a **layout** (latent / pseudo-full-z / later: physics features) | Must be available at AL time without ECAD |
| \(y\) | Scalar **true VAE error** vs ECAD on that layout (and frequency) | Supervised residual — what AL should minimize |

The GP is **not** a generative model of heatmaps. It is a **calibrated error regressor**:

\[
p(y \mid x) = \mathcal{N}\bigl(\mu_e(x),\; \sigma_e^2(x)\bigr)
\]

### 2.2 Two complementary signals

| Signal | Meaning | Captures |
|--------|---------|----------|
| \(\mu_e\) | “How wrong is the VAE *expected* to be here?” | **Systematic** failure (mid-band blur, high-K peak flips) — even in *dense* data |
| \(\sigma_e\) | “How little data / how far from known residuals?” | **Sparsity / dataset holes** — if the hole is *visible* in \(x\)-space |

**Critical:** \(\sigma_e\) alone does **not** equal “model is wrong.” Dense-but-biased regions have high \(\mu_e\) and low \(\sigma_e\). Sparse-but-easy regions can have high \(\sigma_e\) and low \(\mu_e\). UCB uses both.

### 2.3 Occ-only vs full encode — the information bottleneck

At AL scoring time we have **occupancy + K + PI-freq**, not the ECAD heatmap. So we cannot use the full-encode latent (which needs the real heatmap).

Falsification results on exp059 (held-out):

| Path | What \(x\) is | Cost | Struct Spearman (random) | Notes |
|------|---------------|------|--------------------------|-------|
| `full` | encode(real heatmap) | Needs ECAD | ~0.95 | Upper bound / leaky |
| `occ` | encode_layout_latent(occ) | Cheap | ~0.17 | Deployed path; error **not** smooth in \(z\) |
| `pred` | encode(VAE’s own predicted heatmap) | Cheap | ~0.86 | Self-prediction bootstrap |

**Conclusion:** raw occ-\(z\) cannot support a useful error-GP. Self-prediction bootstrap recovers most of the full-path signal without ECAD (Section 4).

---

## 3. Error targets \(y\) — peak, structure, MAE, and MSE

### 3.1 Currently implemented targets

All computed in **normalized** heatmap space (denorm-invariant for ranking / correlation):

| Target key | Definition | Role |
|------------|------------|------|
| `peak_px` | Euclidean distance between hard argmax peaks | Diagnostic only — discontinuous; GP-unfriendly (~0.1 Spearman) |
| `peak_soft_px` | Soft / mass-centroid peak distance | Eval-aligned; smoother than hard argmax |
| `peak_sharp_px` | Temperature soft-argmax peak distance | Training-aligned peak surrogate |
| `struct` | \(1 - \mathrm{Pearson}(\mathrm{pred}, \mathrm{gt})\) | Scale-invariant **shape** error — primary ranking metric in validation |
| `mae` | Mean abs error over pixels | Full-field magnitude+shape blend |

### 3.2 Can we use MSE for “full modality uncertainty”?

**Yes — and for full-modality capture it is often the right primary \(y\).**

Reasoning:

1. **Peak error** answers: “Is the hotspot in the right place?” — high value for design visualization, but sparse / discontinuous / misses global field failure.
2. **Structural (\(1-\rho\))** answers: “Is the spatial pattern right?” — excellent for heatmaps when scale is separately calibrated.
3. **MAE / MSE** answer: “Is the *entire field* wrong?” — integrates every pixel (and can be extended to every modality).

**MSE vs MAE for the GP:**

| Property | MSE | MAE |
|----------|-----|-----|
| Smoothness for GP regression | Excellent | Excellent |
| Sensitivity to outliers / rare hotspots | High (squares large residuals) | Milder |
| Alignment with VAE training loss | Often closer (many VAE terms are MSE-family) | Closer to robust eval |
| “Full modality” interpretation | Energy of residual field | Average absolute residual |

**Recommendation for this framework:**

- **Primary AL target for full-modality uncertainty:** `mse` (heatmap) and/or multi-modal sum (below).
- **Secondary / product-specific:** `struct` (shape-first) and a *smooth* peak target if peaks are the design KPI.
- **Do not** use hard `peak_px` as the sole GP target — falsification showed it is essentially unlearnable.

### 3.3 Multi-modality full uncertainty (heatmap + impedance + occupancy)

If the goal is “the model is uncertain / wrong *as a whole*,” define a **joint residual**:

\[
y_{\mathrm{joint}} =
\alpha_{\mathrm{hm}}\,\mathrm{MSE}(\hat H, H)
+ \alpha_{\mathrm{imp}}\,\mathrm{MSE}(\hat Z, Z)
+ \alpha_{\mathrm{occ}}\,\mathrm{BCE\ or\ MSE}(\hat O, O)
\]

with weights \(\alpha\) chosen so terms are on comparable scales (e.g. normalize each residual by its training-set std, or use log1p of each MSE).

**Why this is valid for GP-AL:**

- Still a single scalar \(y\) → one SVGP (or a multi-output GP).
- Captures holes where *any* modality fails, not only heatmap peaks.
- At scoring time, use the **same self-prediction bootstrap** for each modality you can decode cheaply (heatmap + impedance from occ path; occupancy decode if used).

**Caveat:** joint \(y\) will prioritize modalities with larger dynamic range unless you normalize. Always standardize per-term before summing.

### 3.4 Why normalized-space residuals (not Ω)

Denormalization snaps / interpolates per-MHz stats. Comparing in **train-normalized** space avoids confounding GP targets with denorm calibration bugs (already fixed separately in `src_vae/others/norm_stats.py` via log-frequency interpolation). Ranking and correlation metrics stay meaningful; physical Ω can be used for *reporting* after selection.

---

## 4. Self-prediction bootstrap (“encode your own prediction”)

### 4.1 The chicken-and-egg

Full-encode \(z\) supports a strong error-GP (Spearman ~0.95 on `struct`) **because** \(x\) contains the real heatmap. At AL time we do not have that heatmap — computing it is exactly the expensive ECAD step AL is trying to avoid.

### 4.2 The idea

Use the VAE as a **cheap oracle of its own belief**:

```
occupancy, K, PI_freq
        │
        ▼
  encode_layout_latent  ──►  z_occ
        │
        ▼
  decode  ──►  Ĥ_pred  (and optionally Ẑ_pred)
        │
        ▼
  encode(Ĥ_pred, occ, imp, K, PI)  ──►  z_pred   ← GP input x
        │
        ▼
  SVGP(z_pred) → μ_e, σ_e
```

**Training the GP:** fit on \(x = z_{\mathrm{pred}}\) (not real-heatmap \(z\)), paired with \(y =\) true error of the **same** occ-path prediction vs ECAD. Train and deploy share the same feature distribution → no train/deploy mismatch.

### 4.3 Why this works (reasoned)

1. **Systematic errors are functions of the layout + the model’s own prediction.** Mid-band blur, wrong peak multimodality, magnitude bias — these already appear in \(\hat H\). Re-encoding \(\hat H\) places the layout in a latent neighborhood that reflects *how the model is thinking*, which is most of what full-\(z\) provided for residual regression.
2. **It is not free energy / information creation.** It cannot recover residuals that are pure simulation noise or truly novel physics the model has never expressed. It recovers the **systematic** component.
3. **Empirically (exp059, MHz-band OOD holdout, SVGP):**

| Path | `struct` μ Spearman | Relative to full |
|------|---------------------|------------------|
| full | ~0.85 | 100% |
| **pred** | **~0.72** | **~85%** |
| occ | ~0.15 | ~18% |

So pred ≈ full for **ranking**, at occ cost.

### 4.4 Why occ alone fails (even when occ is trained)

On exp058 (occ path trained + distilled) and exp059 (occ disabled in training), occ-\(z\) still gave ~0.1 Spearman. The occ encoder is an information bottleneck: layouts with different true residual can share similar occ-\(z\), so \(y\) is not a smooth function of \(x\). Distillation to a shared teacher can also collapse diversity. Self-prediction partially *undoes* that bottleneck by injecting the decoded field back into the heatmap encoder.

### 4.5 What self-prediction does **not** guarantee

Novel / epistemic holes (new decap type, regime unlike any labeled residual) can look “fine” under the model’s own prediction → \(z_{\mathrm{pred}}\) looks familiar → \(\mu_e\) and \(\sigma_e\) may both fail to flag them. Mitigation: UCB with \(\sigma_e\) **plus** one independent backstop (latent-density / small ensemble / physics residual). See Section 8.

---

## 5. Latent-path definitions (implementation)

Implemented in `gp_error_surrogate_test.py` as `--path`:

| Flag | Encode \(x\) | Prediction used for \(y\) | Use |
|------|--------------|---------------------------|-----|
| `full` | `encode(H_real, …)` | reconstruct from full \(z\) | Upper-bound / research only |
| `occ` | `encode_layout_latent(occ, …)` | decode from occ \(z\) | Ablation of bottleneck |
| `pred` | `encode(Ĥ_occ, …)` | same Ĥ_occ vs ECAD | **Deployed AL feature** |

`--compare-paths full,pred,occ` evaluates all three on **identical** layouts and the **same** holdout split (apples-to-apples).

---

## 6. GP model: from exact GP to SVGP

### 6.1 Why a Gaussian Process

- Closed-form predictive uncertainty \(\sigma_e\) (epistemic + residual).
- Strong inductive bias: similar \(x\) → similar \(y\) (smoothness in latent space).
- No need for a large second neural net if the residual surface is smooth.

Falsification: on full-\(z\), even a simple sklearn GP + kNN baseline showed `struct` Spearman ~0.95 / ~0.79 → smoothness assumption holds when \(x\) is good.

### 6.2 Why sklearn exact GP was insufficient

- Exact GP is \(O(N^3)\) → fit capped (~1500 points).
- Isotropic RBF in 128-D → poorly calibrated \(\sigma_e\) (σ-calibration Spearman ~0.07–0.08).
- Still useful as a baseline.

### 6.3 SVGP (Sparse Variational GP) — current default for quality

**Backend:** `gpytorch` ApproximateGP + `VariationalELBO`  
**CLI:** `--gp svgp --n-inducing 256 --svgp-epochs 100`

**What is trained:**

1. **Inducing points** \(Z_m\) (\(m \approx 256\)) — learnable locations in latent space summarizing the dataset.
2. **Variational posterior** \(q(u)\) over inducing function values (Cholesky variational distribution).
3. **Kernel hyperparameters** — ScaleKernel × **ARD-RBF** (per-dimension lengthscales) + Gaussian likelihood noise.
4. **Mean** — constant mean (targets are normalized to zero mean / unit variance during fit).

**Training objective:** maximize the variational ELBO (equivalently minimize negative ELBO) with Adam, minibatching over all fit points (no \(N^3\) wall).

**Prediction:** posterior predictive \(\mathcal{N}(\mu, \sigma^2)\) through the likelihood; denormalize \(\mu,\sigma\) back to error units.

**Critical hyperparameter lesson:** default RBF lengthscale (~0.7) in 128-D standardized space makes the kernel ≈ 0 everywhere → constant predictions → NaN Spearman. **Initialize lengthscale ≈ \(\sqrt{d}\)** so distances are on-scale; ARD then learns per-dim scales.

### 6.4 SVGP vs sklearn on the same OOD MHz holdout (pred path)

| Metric | sklearn GP | SVGP |
|--------|------------|------|
| `struct` ranking μ | ~0.66 | **~0.72** |
| Full-path `struct` μ | ~0.80 | **~0.85** |
| Per-point σ calibration (full) | ~0.08 | **~0.14** |
| σ OOD elevation (pred) | ~1.27× | ~1.03× |

**Interpretation:** SVGP is the better **error ranker** (\(\mu_e\)). Its \(\sigma_e\) can be *less* elevated on holes because ARD extrapolates more confidently. Therefore acquisition should emphasize \(\mu_e\) with a moderate \(\kappa\), not \(\sigma_e\) alone.

---

## 7. Evaluation protocol (why each check exists)

### 7.1 Metrics

| Metric | Formula / procedure | Why we use it |
|--------|---------------------|---------------|
| **Spearman(\(\mu_e\), \(y_{\mathrm{true}}\))** | Rank correlation on held-out set | Primary: does acquisition *order* match true error? (AL only needs ranking) |
| **Pearson(\(\mu_e\), \(y_{\mathrm{true}}\))** | Linear correlation | Secondary: magnitude calibration of \(\mu_e\) |
| **kNN baseline Spearman** | Mean \(y\) of \(k\) nearest neighbors in \(x\) | Assumption-light smoothness check; if kNN fails, GP cannot succeed |
| **σ calibration Spearman(\(\sigma_e\), \(\|\mathrm{resid}\|\))** | Does uncertainty track GP’s own residuals? | Trust \(\sigma_e\) for UCB |
| **σ OOD elevation** | \(\mathrm{median}(\sigma_{\mathrm{hole}}) / \mathrm{median}(\sigma_{\mathrm{in\text{-}dist}})\) | Does \(\sigma_e\) rise on a *constructed* dataset hole? |

### 7.2 Holdout regimes (`--holdout`)

| Mode | Procedure | What it tests |
|------|-----------|---------------|
| `random` | Random fit/test split | Interpolation — optimistic upper bound |
| `k` | Hold out whole K bands from fit; test only on those K | Extrapolation to unseen decap counts |
| `mhz` | Hold out whole mid MHz bands | Extrapolation to unseen frequency regimes (closest proxy to “holes”) |
| `type` | Hold out a decap type | True novelty for multi-type extension (needs multi-type data) |

**Why structured OOD is mandatory:** random holdout overstates performance. AL’s purpose is to find holes; only band holdouts simulate holes.

**Leakage control:** `StandardScaler` is fit on the **fit split only**.

### 7.3 Empirical verdict thresholds (used in the test script)

| Spearman(\(\mu_e\), \(y\)) | Verdict |
|----------------------------|---------|
| \(> 0.4\) | STRONG — GP viable |
| \(0.15\)–\(0.4\) | WEAK — latent poorly predicts error |
| \(< 0.15\) | FAIL — choose different \(x\) or \(y\) |

### 7.4 How to run

```bash
# Deployable path + SVGP + frequency hole
python active_learning_pi/al/gp_error_surrogate_test.py \
  --exp experiments/exp059_capacity_freq \
  --path pred --holdout mhz --gp svgp --svgp-epochs 100 --n-samples 4000

# Head-to-head full vs pred vs occ (same layouts)
python active_learning_pi/al/gp_error_surrogate_test.py \
  --compare-paths full,pred,occ --holdout mhz --gp svgp --n-samples 4000
```

---

## 8. Acquisition policy for production AL

### 8.1 Recommended score

\[
a(x) = \underbrace{\mu_e(x)}_{\text{known-bad / systematic}}
      + \kappa\,\underbrace{\sigma_e(x)}_{\text{sparsity}}
      + \beta\,\underbrace{b(x)}_{\text{independent backstop}}
\]

- Start with \(\kappa \in [0.5, 2]\), \(\beta\) small until backstop is calibrated.
- Prefer `pred` path + SVGP + \(y \in \{\mathrm{MSE}, \mathrm{struct}, y_{\mathrm{joint}}\}\).

### 8.2 Independent backstop \(b(x)\) (required for novel types)

Self-prediction is model-looking-at-itself. Add **one** of:

1. **Latent-density / typicality** of occ-\(z\) or physics features under the training aggregate posterior (Mahalanobis / GMM).
2. **Small ensemble disagreement** on peaks / \(Z(f)\) (expensive but strongest epistemic signal).
3. **Physics residual** (coarse resonance estimate vs predicted impedance peaks).

Role of \(b\): catch “all paths agree but are all wrong” and brand-new decap types.

### 8.3 Batch construction

1. Score large pool with \(a(x)\).
2. Cluster / Hamming-dedupe so ECAD budget is not wasted on near-duplicates (`_hamming` already exists in `acquisition.py`).
3. Stratify by K and MHz quotas if needed (`select_worst_for_simulation`).
4. Simulate → ingest → finetune → **refit SVGP** (finetune shifts \(z\); Stage A must be refreshed).

### 8.4 Validate every cycle

Reuse `evaluate_acquisition_rank_quality` (`evaluate_off_anchor.py`), but target the **same \(y\)** used for the GP (e.g. heatmap MSE or peak_soft), not only p99 magnitude. Positive Spearman ⇒ acquisition is directionally correct.

---

## 9. Interaction with the rest of the PDN workflow

### 9.1 Training (VAE)

- Capacity / conditioning (exp059: larger latent, multiscale FiLM, spectral loss) improves **smoothness of error in \(z\)** → makes the GP work better. Capacity and AL-uncertainty are complementary, not alternatives.
- For deployed AL, the occ / layout path used at scoring must be trained (exp058-style). Self-prediction still helps when occ is weak.

### 9.2 Denormalization

Off-anchor magnitude crush from nearest-anchor snapping was fixed by **log-frequency interpolation** of per-MHz stats in `norm_stats.py`. Keep GP targets in normalized space; use corrected denorm for physical QC and reports.

### 9.3 Latent optimization (Stage-2)

Optional trust term: minimize impedance-mask loss \(+ \lambda \sigma_e(z)\), or use \(\sigma_e\) as a post-hoc reject filter on candidate optima so Stage-2 does not “optimize into a hole.”

### 9.4 Decap-type extension

Encode types by physical parameters (C / ESR / ESL), not opaque one-hots. Seed a small labeled set per new type → SVGP \(\sigma_e\) rises where that type appears → AL allocates budget automatically. Confirm with `--holdout type` when multi-type data exists.

---

## 10. Failure modes and mitigations (summary)

| Failure | Symptom | Mitigation |
|---------|---------|------------|
| Bad \(x\) (occ bottleneck) | All targets FAIL | Use `pred` path; add physics features |
| Discontinuous \(y\) (hard peak) | Peak Spearman ~0.1 | Use MSE / struct / soft peak |
| Feedback bias | High Spearman only on selected set | Fixed random holdout; mix random AL picks |
| Poor \(\sigma_e\) | elev ≈ 1, cal ≈ 0 | Rely on \(\mu_e\); improve kernel / inducing; add backstop |
| Stale GP after finetune | Rank quality drops | Re-encode + refit SVGP each cycle |
| Novel type hole | Self-prediction misses | Independent backstop \(b(x)\) |
| High-D kernel collapse | Constant preds / NaN | Init lengthscale \(\sqrt{d}\); ARD |

---

## 11. Recommended default configuration

| Knob | Default | Rationale |
|------|---------|-----------|
| Feature path | `pred` | ~85% of full ranking, no ECAD |
| GP | `svgp`, 256 inducing, ~100 epochs | Scales; better \(\mu_e\) |
| Primary \(y\) | **heatmap MSE** (full modality) + optional `struct` | Captures whole-field uncertainty; smooth |
| Peak KPI | soft/sharp peak as *secondary* | Design-relevant but not sole target |
| Holdout for QA | `mhz` and `k` | Honest hole simulation |
| Acquisition | \(\mu_e + \kappa\sigma_e + \beta b\) | Systematic + sparsity + novelty |
| Pool | Sampled manufacturable layouts | Avoid \(2^{52}\) / Hamming saturation |

---

## 12. Documented process checklist (framework ops)

1. **Dump residuals** on val/train: for each sample compute \(x_{\mathrm{pred}}\) and \(y\) (MSE / joint / struct).
2. **Fit SVGP** on fit split; standardize \(x\) on fit only.
3. **Falsify** with `--holdout mhz` and `--holdout k`; require `struct`/`mse` Spearman \(> 0.4\) on `pred`.
4. **Compare** `--compare-paths full,pred,occ` — pred must be near full, far above occ.
5. **Wire acquisition** in `acquisition.py` / `inference_pool.py`: replace MC badness with SVGP UCB on `pred` features.
6. **Run AL cycle**; compute acquisition-rank quality vs true ECAD residual.
7. **Refit SVGP** after each VAE finetune.
8. **Add backstop** before multi-type rollout; validate with `--holdout type`.

---

## 13. Key empirical results (reference)

exp059, identical layouts where applicable:

| Setting | `struct` μ | Comment |
|---------|------------|---------|
| full, random, sklearn | ~0.95 | Mechanism proven |
| pred, random, sklearn | ~0.86 | Cheap ≈ full (interpolation) |
| pred, mhz OOD, sklearn | ~0.66–0.68 | Hole ranking usable |
| pred, mhz OOD, **SVGP** | **~0.72** | Best deployable ranker so far |
| occ, any | ~0.1–0.27 | Do not use alone |
| σ elev pred (sklearn / SVGP) | ~1.27× / ~1.03× | Prefer μ-driven UCB |

---

## 14. Answers to common design questions

**Q: How does the GP train?**  
On pairs \((x_i, y_i)\) from already-labeled layouts. \(x_i =\) self-prediction latent; \(y_i =\) chosen residual (MSE recommended for full modality). SVGP maximizes ELBO with inducing points + ARD kernel.

**Q: What does it measure?**  
A distribution over **VAE error**, not over heatmaps. \(\mu_e\) = expected error; \(\sigma_e\) = uncertainty of that estimate.

**Q: Can I use MSE instead of peak error?**  
**Yes.** For full-modality / whole-field uncertainty, MSE (or joint multi-modal MSE) is preferred as the primary \(y\). Keep smooth peak metrics as secondary KPIs. Avoid hard argmax as the sole target.

**Q: Does this find dataset holes?**  
It finds **systematic** high-error regions reliably via \(\mu_e\). Dataset sparsity holes are partially reflected in \(\sigma_e\) *if* they remain visible in \(x\)-space. Truly novel holes need an independent backstop.

**Q: Is this cheaper than full-encode uncertainty?**  
Yes — one/two VAE forwards + one SVGP predict; no ECAD for scoring.

---

## 15. Related files

| Path | Role |
|------|------|
| `active_learning_pi/al/gp_error_surrogate_test.py` | Falsification / SVGP / path compare / OOD holdout |
| `active_learning_pi/al/acquisition.py` | Batch selection (to be wired to GP scores) |
| `active_learning_pi/al/inference_pool.py` | Current MC pool scoring (legacy) |
| `active_learning_pi/al/robust_stats.py` | Peak / MC uncertainty helpers |
| `active_learning_pi/al/evaluate_off_anchor.py` | Acquisition rank quality vs ECAD |
| `src_vae/others/norm_stats.py` | Per-MHz norm / off-anchor interpolation |
| `experiments/exp059_capacity_freq/` | High-capacity VAE used for validation |

---

*This document is the design-of-record for latent error-GP active learning in this repo. Update Section 13 when new holdout / multi-type results land.*
