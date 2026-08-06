---
title: Claim Ledger
type: moc
tags: [moc, thesis, evidence]
---

# Claim Ledger

Every claim the thesis might make, paired with the artifact that supports it.
Verdicts below are transcribed from `active_learning_pi/runs/al_exp059_gp_error_001/DECISION_REPORT.md`
(iter 4, generated 2026-07-23). Re-run `COMMAND=evaluate-decision` in
`pipelines/active_learning/run.py` to refresh, then re-run `tools/build_vault.py`.

Source of record: [[al_exp059_gp_error_001]] · overall verdict **PASS**.

---

## Active-learning claims (exp059, gp_error acquisition)

| Claim | Verdict | Key numbers | Safe to write? |
|-------|---------|-------------|----------------|
| `acq_direction` — acquisition scores rank true ECAD error | PASS | Spearman(badness, p99_err) = **0.0087**, n=248; top/bottom-half lift = 0.6477 | ⚠️ **See caution below** |
| `acq_ab_gp_vs_random` — equal-budget GP(μ) beats random | PASS | gp_lift_vs_random = **2.16×** | Yes, with budget stated |
| `acq_ab_gp_vs_mc` — GP beats MC auto_bad | UNCERTAIN | not run (`INCLUDE_MC=False`) | No — must not claim |
| `hole_finding` — GP enriches true high-error holes | PASS | pool_enrichment = **2.94×**, hole_capture = 0.55, capture_lift = 2.75×, lift_vs_random = 3.18× (k=80) | Yes |
| `ft_physical_improve` — fine-tune reduced off-anchor p99 MAE | PASS | 9.3628 → 8.8432, Δ = **−0.5196 (5.55%)**, n=248 both sides | Yes |
| `eval_budget_trust` — ECAD budget fully evaluated | PASS | ingest_ok = **640/640** selected | Yes (supporting) |
| `train_off_anchor_logged` — off-anchor metrics logged | PASS | logged for checkpoint epoch | Yes (supporting) |

Cycle scale: candidates = 12800 → selected = 640 → ingested = 640.

### ⚠️ Caution on `acq_direction`

The ledger marks this PASS, but the two headline numbers do not obviously support a
strong claim: Spearman is **0.0087** — statistically indistinguishable from zero at
n=248 — and the reported top-vs-bottom-half lift is **0.6477**, i.e. below 1.

Before this appears in the thesis, resolve:

1. What pass criterion does `decision_report.py` apply to `acq_direction`? (Is PASS driven
   by the MHz-stratified check rather than the overall Spearman?)
2. Is `top_vs_bottom_half_lift` oriented so that lower is better, or is <1 genuinely adverse?

Until that is settled, write the acquisition story on `acq_ab_gp_vs_random` (2.16× lift) and
`hole_finding` (2.94× enrichment), which are unambiguous, and treat overall-Spearman
directionality as **not yet demonstrated**. Code to check: [[decision_report]], [[validate_acquisition_ab]].

---

## Architecture claims

| Claim | Status | Evidence needed / available |
|-------|--------|------------------------------|
| GNN encoders beat MLP on occupancy | Documented | [[gnn-rationale]] cites the exp055→056→057 arc — verify the numbers are still current for exp059 before citing |
| Structured latent improves spatial fidelity | Documented | [[exp057_structured_graph]] notes; superseded by exp059 capacity study |
| Capacity + multi-scale FiLM fixes mid-band blur | **Hypothesis** | [[exp059_capacity_freq]] `notes.md` states the intent and the metric to watch (off-anchor FG-MSE / Pearson at 155/250/265 MHz). Confirm from `experiments/exp059_capacity_freq/metrics/off_anchor_eval.csv` before claiming |
| Removing U-Net skips matches stage-2 usage | Documented decision | exp059 notes, "Update" section |
| Adversarial/GAN loss deliberately excluded | Design decision | exp059 notes — argued, not measured |

## Inverse-design claims

| Claim | Status | Note |
|-------|--------|------|
| STE top-K restores gradient flow through discrete read-out | Mechanism, documented | [[latent-optimization]], `_ste_topk` in [[optimize]] |
| Optimizer returns lowest-peak feasible placement per K | Implemented behaviour | [[latent-optimization]] selection rule |
| End-to-end proposals validated against ECAD | **Not established for exp059** | Stage 2 still runs on exp038 checkpoints — see [[Thesis Outline]] §6 |

## Claims that must NOT be made

- GP beats MC acquisition — `acq_ab_gp_vs_mc` is **UNCERTAIN**, the comparison was never run.
- Any exp059 end-to-end inverse-design result — Stage 2 is not wired to exp059.
- Cross-design generalization — the surrogate is trained on a single board.
- Surrogate feasibility as sign-off — [[limitations-and-validity]] requires ground-truth re-simulation.

---

## Refresh procedure

```bash
# regenerate the ledger from existing cycle numbers
# (set COMMAND = "evaluate-decision" in the CONFIG block first)
python pipelines/active_learning/run.py

# equal-budget acquisition A/B, no new ECAD
python active_learning_pi/al/validate_acquisition_ab.py

# rebuild this vault's generated notes
python tools/build_vault.py
```
