# PDN Decap Optimization via VAE Surrogate + Latent Optimization

A data-driven method for **Power Delivery Network (PDN) decoupling-capacitor (decap) placement**
on PCBs. Instead of the manual trial-and-error loop an engineer runs today, this project trains a
generative **surrogate** of a single design and then performs **gradient-based latent optimization**
to propose decap configurations whose predicted PI-spectrum meets a target impedance — turning a
2⁵² combinatorial search into a guided inverse-design problem.

> **Scope:** research prototype on a *single* PCB design. The system is **decision support** — it
> proposes good starting configurations for the engineer's iterative process, not a guaranteed
> optimal solution. Every proposed config should be re-verified against the real PI simulator.

---

## The problem

For a given PCB there are **52 candidate slots** where decaps can be placed. Engineers iterate:

1. Assign decaps to slots (position + type/value).
2. Simulate the **PI-spectrum** (impedance magnitude vs. frequency, e.g. 1 MHz – 600 MHz).
3. Compare against a **target impedance** curve.
4. If peaks exceed target, run **PI-distribution** (spatial hotspot maps) at the peak frequencies.
5. Turn on decaps near the hotspots and repeat.

There is no closed-form solution and the search space (2⁵² ≈ 4.5×10¹⁵) is intractable to brute-force.

## The approach

**Stage 1 — Train a multi-input VAE surrogate** (`experiments/exp043/`)

A product-of-experts VAE learns a shared latent space over three modalities of one design:

| Modality          | Shape       | Meaning                                              |
| ----------------- | ----------- | ---------------------------------------------------- |
| Decap occupancy   | `[52]`      | binary on/off per slot                               |
| PI-spectrum (mag) | `[231]`     | impedance magnitude across the frequency band        |
| PI-distribution   | `[64×64×1]` | spatial hotspot heatmap at ~20 frequency anchors     |

Conditioned on **K** (number of active decaps) and **frequency** (for the distribution head).
The frequency signal is injected as a dedicated PoE expert on heatmap-private latent dimensions
(see [`vae_poe_freq.py`](experiments/exp043/codes/vae_poe_freq.py)).

**Stage 2 — Latent optimization** ([`Latent_opm/latent_optimization_impedance.py`](Latent_opm/latent_optimization_impedance.py))

With the decoder frozen, run gradient descent on the latent vector `z` for each `K`:

```
z → occupancy_decoder → occ → top-K (straight-through) → surrogate → PI-spectrum → loss vs. target
```

The **straight-through estimator** on the hard top-K keeps the surrogate's input in-distribution
(exactly K binary slots) while letting the spectrum-matching gradient flow back to `z`. For each K
the optimizer returns the latent that **satisfies the target with the lowest peak impedance** across
all seeds, or **no solution** if no seed meets the target. Peak frequencies of a solution can then be
rendered as PI-distribution heatmaps for interpretability.

---

## Repository layout

```
experiments/            VAE surrogate experiments (exp029 … exp043; exp043 is current)
  exp043/codes/         model (vae_poe_freq.py), training, inference, evaluation
  exp038_true_multi/    multifreq baseline + impedance surrogate used by latent-opt
Latent_opm/             Stage-2 latent optimization, reporting, feasibility search
Data_Creation/          dataset generation: occupancy, impedance, heatmap, PI-distribution
scripts/                normalization + latent-statistics utilities
configs/                target_impedance.npy, frequency grid, anchors, masks
evaluation/             novelty / quality evaluation of generated samples
source/, src_vae/       shared model + loss code
viewer/, visualization/ result viewers and plotting
```

## Data format

The dataset itself is **not** committed (see *Excluded from the repo* below). Expected layout:

```
datasets/data_multifreq/
  manifest.csv          one row per (layout, frequency) sample
  layouts/              per-layout occupancy vectors
  Imp/                  PI-spectrum magnitudes  [231]
  PI_freq/              frequency-conditioning vectors
  heatmap/              PI-distribution maps     [64×64]
  Occ_map/              occupancy maps
```

Normalization statistics live in `datasets/data_multifreq_norm/normalization_stats.json`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pyyaml matplotlib pandas pillow   # + project deps
```

Tested with PyTorch 2.7 (CUDA 11.8). A GPU is recommended for training.

## Usage

**Train the surrogate (Stage 1):**

```bash
python experiments/exp043/codes/train_vae_simple.py
```

Hyperparameters are in [`experiments/exp043/config.yaml`](experiments/exp043/config.yaml)
(latent_dim=42, K-balanced sampling, frequency-conditioned heatmap head, etc.).

**Run latent optimization (Stage 2):**

```bash
# uses experiments/exp038_true_multi checkpoint + impedance surrogate by default
python Latent_opm/latent_optimization_impedance.py
```

Outputs go to `Latent_opm/runs/<experiment>/<run-idx>/K##/` as `best_latent.npy`,
`best_occupancy_topk.npy`, `best_metrics.json`, or `no_solution.json`. Key knobs (env-overridable
at the top of the script): `NUM_STEPS`, `LR`, `NUM_CANDIDATE_SEEDS`, `BOUNDARY_MARGIN` (feasibility
margin), `SELECT_METRIC` (`max_ohm` = lowest peak).

## Excluded from the repo

To keep the repository light, the following are intentionally **not** tracked (see `.gitignore`):

- `datasets/` — training data (tens of GB)
- model checkpoints (`*.pt`, `checkpoints/`) — ~11 GB; regenerate by training
- `*.peb` simulation/heatmap exports, oversized HTML reports
- `src_gan/`, `temp_visuals/`, `ppt/`, environments and IDE folders

## Status & caveats

- Trained and validated on **one** design; cross-design generalization is future work.
- "Satisfies target" in Stage 2 means the **surrogate predicts** the spectrum clears target —
  proposed configs should be re-simulated with the ground-truth PI solver before use.
- The straight-through top-K (Stage 2) is what makes the search genuinely gradient-guided rather
  than random seed sampling.
