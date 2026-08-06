---
title: cursor-handoff
type: concept
source: docs/cursor-handoff.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/cursor-handoff.md` — edit the source file, then re-run `tools/build_vault.py`.

# Cursor handoff — GP error-surrogate AL + exp059

**Last updated:** 2026-07-20  
**Use at start of a new chat:** `@docs/cursor-handoff.md` (and optionally `@active_learning_pi/GP_ERROR_SURROGATE_FRAMEWORK.md`)

**Decision (user):** further experiments and AL use **`exp059_capacity_freq` only**. exp057/exp058 are legacy; occ-only quality is similarly weak on 058 vs 059, so stick with the capacity/FiLM model.

**Agent protocol:** `.cursor/rules/numerical-claims.mdc` — always ground suggestions/claims in numerical evaluation (or label as unmeasured hypothesis). End-of-cycle rollup: `DECISION_REPORT.md`.

---

## 1. Workspace

| Item | Value |
|------|-------|
| **Project root** | `/home/ubuntu/genai_pdn` |
| **OS** | WSL2 Ubuntu |
| **Python** | Prefer `venv-cgan` / `.venv`; `gpytorch` for SVGP |
| **Primary model** | `experiments/exp059_capacity_freq` |
| **AL configs** | `active_learning_pi/config/exp059_gp_error.json` (default), `exp059_random.json`, `exp059.json` (MC) |
| **`.cursorignore`** | Blocks `experiments/`, `datasets/`, `*.pt` — use shell for those |

### Docs to load first

| Doc | Purpose |
|-----|---------|
| active_learning_pi/GP_ERROR_SURROGATE_FRAMEWORK.md (`../active_learning_pi/GP_ERROR_SURROGATE_FRAMEWORK.md`) | Design-of-record for residual-GP AL |
| [[gp-error-surrogate|docs/gp-error-surrogate.md]] | Thesis summary + mitigations |
| This handoff | Current state / next steps |

---

## 2. Two-phase exp059 plan (important)

Current **capacity** training (running or completed) uses full-encode only:

- `layout_train_prob=0`, `occ_only_encode_prob=0` — intentional for mid-band sharpness study.

**AL / deployment scoring still needs an occ→heatmap path** (pred bootstrap). After capacity `last_model.pt` exists:

1. Keep capacity checkpoint as base.
2. Run AL cycles with `config_al_finetune.yaml` (already created): `layout_train_prob=0.9`, `occ_only_encode_prob=0.7`, overlay + distill.
3. Acquisition: **`score_mode: mu`** residual-GP (`exp059_gp_error.json`), not MC.
4. Control: `exp059_random.json` at equal ECAD budget.

Do **not** expect strong AL from a pure capacity checkpoint with occ never trained — the GP still ranks residuals of whatever path exists, but FT should turn on layout/occ mix.

---

## 3. What this thread accomplished

### A. exp059 capacity / mid-band heatmaps
- Higher capacity + multi-scale FiLM + spectral loss; layout holdout; skips removed.

### B. Off-anchor denorm interp — fixed in `norm_stats.py`

### C. Residual-GP AL (production opt-in)
- `pred` bootstrap + GP(`mu`) acquisition wired; novelty optional (default off).
- Equal-budget validator: `al/validate_acquisition_ab.py` (defaults → **exp059**).
- Smoke A/B on **exp057** showed `gp_mu` beats random; re-run on **exp059** before claiming.

### D. Open

- [ ] Wait for exp059 capacity train → `last_model.pt`
- [ ] `validate_acquisition_ab.py` on exp059 (PASS before FT claims)
- [ ] AL cycle: GP vs random on exp059 → held-out off-anchor
- [ ] Optional: `INCLUDE_MC=True` A/B when GPU free
- [ ] Optional novelty enable only if A/B PASSes with `ucb_novelty`

---

## 4. File map

```
experiments/exp059_capacity_freq/
  config.yaml                 # capacity phase (layout/occ probs = 0)
  config_al_finetune.yaml     # AL FT overrides (layout 0.9 / occ 0.7)
  config_occ_only_finetune.yaml

active_learning_pi/config/
  exp059_gp_error.json        # PRIMARY AL + GP(mu)
  exp059_random.json          # equal-budget control
  exp059.json                 # MC on exp059

active_learning_pi/al/
  gp_error_surrogate.py
  validate_acquisition_ab.py  # EXPERIMENT default = exp059
  decision_report.py          # PASS/FAIL/UNCERTAIN ledger from collected numericals
  inference_pool.py           # acquisition_mode: gp_error | random | mc

pipelines/active_learning/run.py   # CONFIG_PATH → exp059_gp_error.json
# COMMAND=evaluate-decision regenerates DECISION_REPORT.md
```

---

## 5. Commands

```bash
cd /home/ubuntu/genai_pdn

# After capacity checkpoint exists — acquisition A/B (no new ECAD)
python active_learning_pi/al/validate_acquisition_ab.py

# AL (edit PROPOSE_ONLY=True first to dry-run scoring)
python pipelines/active_learning/run.py
# CONFIG_PATH already exp059_gp_error.json
# full cycle always: pre-FT eval → FT → post-FT eval → DECISION_REPORT.md

# Regenerate decision ledger from existing cycle numbers
# COMMAND=evaluate-decision in run.py
```

**Always-on numerical path:** ingest → pre-FT eval (rank quality + p99) → post-FT eval → `DECISION_REPORT.md` with PASS/FAIL/UNCERTAIN claims (scoring direction, GP vs random A/B, FT improvement, budget trust).

---

## 6. Hard-won decisions

1. Residual-GP acquisition > MC self-uncertainty.
2. Default score = **μ_e**; novelty-UCB only after A/B PASS.
3. **exp059 only** going forward for model + AL.
4. Capacity phase ≠ AL-ready occ path — use `config_al_finetune.yaml` after capacity.
5. Off-anchor denorm must interpolate.

---

## 7. Next tasks

1. Finish/confirm exp059 capacity train → `last_model.pt`.
2. Run `validate_acquisition_ab.py` on exp059; archive PASS/FAIL JSON.
3. One ECAD A/B: `exp059_gp_error` vs `exp059_random` → same FT → held-out metrics.


## Implemented by

- [[run]] — `pipelines/active_learning/run.py`
- [[validate_acquisition_ab]] — `active_learning_pi/al/validate_acquisition_ab.py`
- [[decision_report]] — `active_learning_pi/al/decision_report.py`
