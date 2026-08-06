# Hole-finding evaluation report

**Generated:** 2026-07-23T16:07:21  
**Experiment:** `experiments/exp059_capacity_freq`  
**Path / target:** `pred` / `p99_ae`  
**Overall:** **PASS**  

GP acquisition enriches true holes at equal budget (primary k_80: lift_vs_random=3.18x, pool_enrichment=2.94x, hole_capture=0.55, capture_lift=2.75x). Mid-band structured check also PASS.

---

## What this proves

On a **fully labeled holdout** (no new ECAD), acquisition scores every layout.
We then ask: does GP's top-K contain **true high-error holes** better than random?

| Metric | Meaning |
| --- | --- |
| Lift vs random | mean true error in GP top-K / random top-K |
| Pool enrichment | mean true error in GP top-K / holdout mean |
| Hole capture | fraction of GP top-K that are true top-Q% worst |
| Capture lift | capture / chance rate |
| Oracle efficiency | GP top-K mean / perfect top-K mean (≤1) |

---

## Pass checklist

| Check | Result |
| --- | --- |
| `lift_vs_random` | PASS |
| `pool_enrichment` | PASS |
| `hole_capture` | PASS |
| `spearman_positive` | PASS |
| `midband_structured` | PASS |

---

## Results by budget

### Budget K=40

- Spearman(score, y): **0.4572**
- GP top-K mean y: **7.6178** (random 3.4263, oracle 35.1345, pool 3.2736)
- Lift vs random: **2.223×**
- Pool enrichment: **2.327×**
- Hole capture (Q=0.2): **0.475** (chance 0.200, lift 2.373×)
- Recall of holes: **0.093**
- Oracle efficiency: **0.217**

### Budget K=80

- Spearman(score, y): **0.4572**
- GP top-K mean y: **9.6358** (random 3.0288, oracle 24.2584, pool 3.2736)
- Lift vs random: **3.181×**
- Pool enrichment: **2.944×**
- Hole capture (Q=0.2): **0.550** (chance 0.200, lift 2.747×)
- Recall of holes: **0.215**
- Oracle efficiency: **0.397**

### Budget K=160

- Spearman(score, y): **0.4572**
- GP top-K mean y: **7.2419** (random 3.3856, oracle 15.6436, pool 3.2736)
- Lift vs random: **2.139×**
- Pool enrichment: **2.212×**
- Hole capture (Q=0.2): **0.450** (chance 0.200, lift 2.248×)
- Recall of holes: **0.351**
- Oracle efficiency: **0.463**

---

## Structured mid-band hole check

Fit GP **outside** MHz∈[150.0, 280.0], score **inside** (n_fit=1907, n_test=653).

- Lift vs random (K=80): **2.242×**
- Pool enrichment: **2.346×**
- Hole capture lift: **2.056×**
- Spearman: **0.3705**

---

## How to cite

Claim only if overall is **PASS**: residual-GP acquisition on the `pred` path with target `p99_ae` enriches true VAE holes vs random at equal budget on a labeled holdout. Scope mid-band separately if structured check fails.

Artifact: `/home/ubuntu/genai_pdn/active_learning_pi/runs/hole_finding_exp059/LATEST_hole_finding.json`
