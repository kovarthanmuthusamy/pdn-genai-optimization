---
title: gnn-rationale
type: concept
source: docs/gnn-rationale.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/gnn-rationale.md` — edit the source file, then re-run `tools/build_vault.py`.

# Why GNNs for PDN occupancy and spectra (exp056 → exp057)

Thesis-facing note: **why** graph neural networks (GNNs) were introduced, **what** they encode, and **what numbers** from the architecture lineage support keeping them. Surviving design choices are still used in later models (exp058/059 keep the graph occupancy / structured-graph lineage).

**Scope**

| Experiment | Folder | GNN role |
|------------|--------|----------|
| **exp054** | `exp054_K_30` | MLP occupancy (longer train, to ep **675**) |
| **exp055** | `exp055_hard_occ` | MLP occupancy (hard-occ; stopped @ ep **400**) |
| **exp056** | `exp056_graph_vae` | **Occupancy** GNN only (architecture-only swap vs exp055) |
| **exp057** | `exp057_structured_graph` | Occupancy GNN **+** impedance **spectrum** GNN + occ↔imp trunk |

Implementation: `graph_occ.py`, `graph_imp.py` under `experiments/exp057_structured_graph/codes/` (occ GNN first landed in exp056). Broader architecture: [[model-architecture|model-architecture.md]].

---

## 1. Design reason (domain structure → inductive bias)

PDN layouts and spectra are **not** bags of independent coordinates. The board and the frequency axis have known neighborhood structure that an MLP must re-learn from scratch.

### Occupancy (52 decap slots) — exp056

- Slots sit on a **physical 7×8 PCB grid**. Neighboring pads share vias / planes / return paths; placing a decap at slot \(i\) strongly affects neighbors.
- Represent occupancy as a **graph**: 52 nodes, **4-neighbor** edges on the board grid + self-loops, symmetrically normalized adjacency.
- Message passing (`OccGraphEncoder` / `OccGraphDecoder`) pools local slot context before the PoE latent — instead of a flat MLP \(52\to H\).

**Hypothesis (pre-experiment):** spatial inductive bias should improve layout-conditioned heatmaps / occupancy consistency without changing the loss stack.

### Impedance spectrum (231 bins) — exp057

- \(Z(f)\) is a **1-D ordered signal**: adjacent frequency bins are strongly correlated (resonances span neighborhoods on the log-frequency axis).
- Represent the spectrum as a **chain graph** of 231 nodes; `ImpSpectrumGraphEncoder` / `Decoder` operate on adjacent bins.
- Impedance decode is further conditioned on **GNN(occ)** so layout geometry couples into \(Z(f)\).

**Hypothesis:** a chain GNN + occ-conditioned decode should improve spectral shape / peak fidelity vs a flat MLP over 231 bins.

### Why not keep MLP?

| Alternative | Limitation for this problem |
|-------------|-------------------------------|
| Flat MLP on 52-d occupancy | Ignores known slot adjacency; must learn locality from data alone |
| CNN on a rasterized board | Possible, but slots are a sparse labeled graph, not a dense image; GNN matches the discrete CAD object directly |
| Flat MLP on 231-d spectrum | Ignores frequency-local correlation; harder to share filters across resonances |

GNN choice = **explicit physics/CAD structure in the architecture**, with losses held fixed in the critical ablations below.

---

## 2. Experimental design (what counts as “proof”)

### Clean ablation: occupancy GNN (exp055 → exp056)

- **Same** data (`data_multifreq_train_norm_unbounded`), **same** loss terms/weights, **same** heatmap / impedance / PoE stack.
- **Only** change: occupancy encoder/decoder MLP → PCB-slot GNN.
- Therefore differences are attributable to the **occupancy GNN**, not a new loss recipe.

### Package change: spectrum GNN (exp056 → exp057)

exp057 changes **three** things at once:

1. Structured latent \(z = [z_\mathrm{shared}(42)\,|\,z_\mathrm{peak}(8)\,|\,z_\mathrm{spatial}(15)]\)
2. Spectrum **chain GNN** for impedance
3. Occ↔imp joint trunk (imp decode conditioned on GNN(occ))

So exp057 vs exp056 is **not** a single-variable “spectrum GNN vs MLP” proof. Treat spectrum-GNN benefits as **supported by the package’s metrics + design rationale**, and label any causal claim as **hypothesis / multi-factor**.

---

## 3. Numerical evidence (from training metrics)

Source: `experiments/<exp>/metrics/loss.csv` and `off_anchor_eval.csv` (`kind=layout_cross`).  
All figures below are **logged training metrics**, not a separate ECAD A/B.

### 3.0 Occupancy loss trajectory (MLP plateau vs GNN)

Longer MLP baseline = **exp054_K_30** (to ep **675**). Shorter hard-occ MLP = exp055 (to ep 400).

`val_occupancy_loss` from `loss.csv`:

| Epoch | exp054 MLP | exp055 MLP | exp056 GNN | exp057 GNN+ |
|------:|-----------:|-----------:|-----------:|------------:|
| 100 | 0.470 | 0.382 | 0.490 | 0.492 |
| 200 | 0.357 | 0.287 | 0.368 | 0.389 |
| 400 | 0.194 | **0.167** (stop) | 0.190 | 0.219 |
| 500 | 0.152 | — | 0.143 | 0.172 |
| 600 | 0.122 | — | 0.116 | 0.137 |
| **675** | **0.107** (stop) | — | **0.098** | 0.117 |
| 700 | — | — | 0.093 | 0.110 |
| 900 | — | — | **0.066** | 0.075 |
| ~1435 | — | — | — | **0.038** |

Threshold crossings (`val_occupancy_loss`):

| Threshold | exp054 MLP | exp055 MLP | exp056 GNN | exp057 |
|-----------|------------|------------|------------|--------|
| \< 0.15 | ep **525** | **never** (min 0.167) | ep 500 | ep 575 |
| \< 0.10 | **never** (min **0.107** @675) | never | ep **675** | ep 775 |
| \< 0.05 | never | never | never (min 0.066) | ep **1100** |

**Read (matched numbers):**

- Your “MLP stuck near **0.1**” matches **exp054@675: val_occ = 0.107** — it never crossed 0.10 in 675 epochs.
- At the **same** long horizon (~675), GNN (exp056) is only slightly lower: **0.098 vs 0.107** — not a dramatic gap yet.
- GNN **keeps falling** with more epochs (**0.066** @900; exp057 **0.038** @~1435). MLP was never trained past 675, so “GNN → 0.01” is **not** measured as an equal-budget win; logged GNN floors are **~0.06 / ~0.04**, not 0.01.

**Fair claim:** MLP occupancy loss asymptotes near **~0.11** by ep 675; GNN crosses below 0.10 and continues into the **0.06–0.04** range when trained longer. Equal-epoch GNN edge at 675 is small (**Δ ≈ 0.009** val).

---

### 3.1 Occupancy GNN — matched epoch (exp055 MLP vs exp056 GNN @ ep 400)

| Metric (val) | exp055 MLP @400 | exp056 GNN @400 | Δ (056−055) |
|--------------|-----------------|-----------------|-------------|
| `val_heatmap_loss` | 0.0287 | **0.0169** | **−0.0118** (better) |
| `val_occupancy_loss` | **0.1671** | 0.1896 | +0.0225 (worse) |
| `val_impedance_loss` | **0.1539** | 0.1599 | +0.0060 (≈ flat / slightly worse) |
| `val_total_loss` | **2.780** | 3.060 | +0.280 (worse) |

Off-anchor heatmap shape @ ep 400 (`layout_cross`):

| MHz | exp055 Pearson | exp056 Pearson |
|-----|----------------|----------------|
| 100 | **0.393** | 0.365 |
| 270 | **0.613** | 0.513 |
| 400 | **0.586** | 0.454 |

**Read carefully:** at the **same** epoch, the occupancy GNN does **not** dominate every metric. It **does** cut validation heatmap loss sharply; occupancy BCE and off-anchor Pearson at ep 400 favor the MLP. That is consistent with “inductive bias helps the spatial map path” rather than “GNN wins every head immediately.”

### 3.2 Occupancy GNN — after full exp056 train (@ ep 900)

| MHz | exp055 @400 Pearson | exp056 @900 Pearson | FG MSE @900 |
|-----|---------------------|---------------------|-------------|
| 100 | 0.393 | **0.411** | 1.768 |
| 270 | 0.613 | **0.670** | **0.805** |
| 400 | 0.586 | **0.613** | **0.860** |

**Verdict (occupancy GNN):** after completing the GNN train, mid/high-anchor **layout_cross Pearson improves** vs the MLP stop at ep 400 (270: 0.613 → 0.670; 400: 0.586 → 0.613). This is **supportive but not a same-epoch proof** that GNN is strictly better than continuing the MLP to 900 (that MLP-to-900 run was not logged as a parallel arm). Label: **directional evidence + design rationale**, not a fully controlled equal-epoch final bake-off.

### 3.3 Structured graph package (exp057 @ ep 400) vs prior stops

Same off-anchor MHz, `layout_cross` Pearson:

| MHz | exp055 @400 | exp056 @400 | exp056 @900 | **exp057 @400** |
|-----|-------------|-------------|-------------|-----------------|
| 100 | 0.393 | 0.365 | 0.411 | **0.412** |
| 270 | 0.613 | 0.513 | 0.670 | **0.694** |
| 400 | 0.586 | 0.454 | 0.613 | **0.628** |

At **matched ep 400**, the **exp057 package** (structured latent + spectrum GNN + occ↔imp) already exceeds both the MLP baseline and the early occupancy-GNN checkpoint on 270/400 Pearson.

Matched-epoch `val_impedance_loss` (056 vs 057) still often favors occupancy-only GNN (e.g. @400: 056 **0.160** vs 057 0.220) — so impedance head loss alone is **not** the selling point of the package; **off-anchor heatmap Pearson** is.

**Verdict (exp057 / spectrum GNN):** keep GNN + structured coupling because the **combined** model improves the physical QC we care about (off-anchor Pearson at 270/400) at equal epoch vs 055/early-056. Do **not** claim “spectrum GNN alone caused ΔX” without a dedicated MLP-spectrum control (**not measured**).

---

## 4. What we keep in later experiments

| Later model | GNN inheritance |
|-------------|-----------------|
| exp058 | Same structured-graph / occ path lineage; AL-oriented train mix |
| **exp059** (current AL track) | Capacity / FiLM / spectral-loss upgrades **on top of** the graph-structured surrogate family |

Active learning and residual-GP work (`docs/gp-error-surrogate.md`) **assume** this architecture (including `encode_occupancy_latent` / layout GNN path for the cheap `pred` bootstrap).

---

## 5. Thesis wording (recommended)

**Safe claims (supported)**

1. Occupancy was switched from MLP → PCB-slot GNN in an **architecture-only** step (exp055→056) with **losses held fixed**.
2. **Occupancy loss:** longer MLP (**exp054@675**) bottoms at **val_occ = 0.107** (never \< 0.10). GNN (**exp056@675**) is **0.098**; with more epochs GNN reaches **0.066** (@900) and exp057 **~0.038**. Equal-epoch Δ at 675 is small (~0.009).
3. At ep 400, GNN occupancy also reduced **`val_heatmap_loss`** (0.0287 → 0.0169).
4. After full exp056 training, off-anchor Pearson at 270/400 MHz exceeds the exp055@400 MLP stop.
5. The exp057 **structured-graph package** (incl. spectrum chain GNN) reaches higher off-anchor Pearson at 270/400 by ep 400 than exp055@400 and exp056@400.

**Do not claim without new experiments**

1. “GNN occupancy strictly beats MLP at every epoch / every metric.” (ep-400 off-anchor Pearson contradicts that.)
2. “Spectrum GNN alone improved impedance by X%.” (no single-factor ablation.)
3. “GNN is required for residual-GP AL.” (AL uses the pred path of the trained model; GNN is part of that model, but GP A/B does not isolate GNN.)

---

## 6. Figure / citation checklist

When writing the methods chapter:

1. Draw the **52-slot 4-neighbor graph** and the **231-bin chain graph**.
2. Cite the **architecture-only** exp055→056 table (heatmap val loss + off-anchor Pearson with epoch labels).
3. Cite the **exp057@400** off-anchor Pearson table with an explicit “multi-factor package” footnote.
4. Point code to `graph_occ.py` / `graph_imp.py` and metrics CSVs under `experiments/exp055_*`, `exp056_graph_vae`, `exp057_structured_graph`.

---

## 7. Related docs

| Doc | Role |
|-----|------|
| [[model-architecture|model-architecture.md]] | Full exp054→057 stack |
| [[experiment-lineage|experiment-lineage.md]] | Where 056/057 sit vs 058/059 |
| [[evaluation-metrics|evaluation-metrics.md]] | How Pearson / FG MSE are defined |
| Archive: `_archive/EXP056_GRAPH_VAE.md`, `_archive/EXP057_STRUCTURED_GRAPH_VAE.md` | Implementation session notes |


## Implemented by

- [[experiments.exp059_capacity_freq.codes.graph_occ]] — `experiments/exp059_capacity_freq/codes/graph_occ.py`
- [[experiments.exp059_capacity_freq.codes.graph_imp]] — `experiments/exp059_capacity_freq/codes/graph_imp.py`
