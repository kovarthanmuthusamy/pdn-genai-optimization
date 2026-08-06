# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A master's-thesis research codebase: a **generative-AI surrogate + inverse-design framework for PCB Power Delivery Network (PDN) decap placement**. A multi-input Product-of-Experts VAE learns one board's mapping between decap occupancy `[52]`, PI-spectrum `[231]` (1–600 MHz), and spatial PI-distribution heatmaps `[64×64]` conditioned on decap budget `K` and inspection frequency. With the decoder frozen, gradient-based latent optimization searches for placements meeting a target impedance; an active-learning loop fine-tunes the surrogate on newly ECAD-simulated high-uncertainty layouts.

There is **no package, no build system, no test suite, no linter config**. Everything is run as scripts/modules from the repo root.

## Current model track (important — the root README is stale)

- **`experiments/exp059_capacity_freq` is the only model used going forward** for both training and AL (see `docs/cursor-handoff.md`, `.cursor/rules/numerical-claims.mdc`). `exp057_structured_graph` and `exp058_asymmetric_kl` are legacy; `exp060_multitype_occ` is an exploratory fork (multi-type one-hot occupancy `52×T`).
- Root `README.md` still describes exp057 as current. `docs/README.md` and `docs/cursor-handoff.md` are the accurate status.
- exp059 has **two phases**, and the distinction matters:
  - `config.yaml` — *capacity phase*, full-encode only (`layout_train_prob`/`occ_only_encode_prob` were 0 during that study; current values 0.65/0.4).
  - `config_al_finetune.yaml` — *AL fine-tune* overrides (`layout_train_prob=0.9`, `occ_only_encode_prob=0.7`, overlay + distillation). AL scoring needs the occ→heatmap path, so a pure capacity checkpoint is not AL-ready.

## Commands

Activate the environment first (`/home/ubuntu/venv-cgan` is the one used in the handoff docs; `.venv` also exists):

```bash
cd /home/ubuntu/genai_pdn && source /home/ubuntu/venv-cgan/bin/activate
```

### Train the surrogate (exp059)

```bash
export CUDA_VISIBLE_DEVICES=0
export VAE_EXPERIMENT_DIR=$(pwd)/experiments/exp059_capacity_freq
unset VAE_CONFIG_PATH
python -m experiments.exp059_capacity_freq.codes.train_vae_simple

# 2-GPU DDP
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 -m experiments.exp059_capacity_freq.codes.train_vae_simple
```

Training must be launched as a **module** (`-m`), not a file path. Checkpoints land in `experiments/<exp>/checkpoints/last_model.pt`; metrics/CSVs in `experiments/<exp>/metrics/`.

### Active learning (primary loop)

```bash
python pipelines/active_learning/run.py          # CONFIG_PATH defaults to config/exp059_gp_error.json
python active_learning_pi/al/validate_acquisition_ab.py   # equal-budget acquisition A/B, no new ECAD
```

Never run `active_learning_pi/al/pipeline.py` directly — it delegates to `run.py`. Set `COMMAND` in `run.py`'s CONFIG block: `full` (8 steps: generate → infer → select-bad → build-peb/simulate → ingest → build-overlay → finetune → post-FT eval), or a single step (`generate`, `infer`, `evaluate-decision`, `finetune`, `occ-warmup`, …). `PROPOSE_ONLY=True` dry-runs scoring with no ECAD and no fine-tune.

AL configs: `exp059_gp_error.json` (primary, residual-GP `mu` acquisition) · `exp059_random.json` (equal-budget control) · `exp059.json` (MC acquisition).

### Other entry points

```bash
python pipelines/data/processing_multifreq.py     # build dataset from raw ECAD export
python pipelines/normalize/multifreq.py           # robust per-MHz normalization
python pipelines/latent/optimize.py               # Stage-2 inverse design
python pipelines/latent/export_peb.py             # export solutions to ECADSTAR PEB
python pipelines/latent/compare_report.py         # surrogate vs simulated comparison
python evaluation/vae/run_vae_eval.py             # reconstruction + latent diagnostics
```

`pipelines/latent/optimize.py` still defaults to `EXPERIMENT = "exp038_true_multi"` for the occupancy decoder + impedance surrogate — Stage 2 is *not* wired to exp059. Do not assume it reflects the current model.

## Conventions that will bite you

**CONFIG-only scripts.** Nothing under `pipelines/`, `evaluation/`, or `scrap/` takes CLI arguments. Each has a `# CONFIGURATION — edit these before running` block of module-level constants; edit it, then `python <path>`. Module docstrings carry *What / Usage / Config keys* ("Agent notes"). Code under `libs/` is import-only. When adding a runnable script, follow this pattern rather than adding argparse.

**`config.yaml` files are JSON, not YAML.** `load_yaml_config()` (in each experiment's `<exp>_common.py`) strips `#` comment lines and calls `json.loads`. YAML syntax (bare keys, `-` lists, anchors) will fail to parse. Same for `active_learning_pi/config/*.jsonlike.yaml`.

**Paths go through `repo_paths.py`.** Import `REPO_ROOT`, `repo_path()`, `setup_path()`, `resolve_repo_path()` instead of hard-coding absolutes or `parents[N]`. `resolve_repo_path` deliberately remaps stale prefixes (`/home/ubuntu/gan/`, …) from older checkouts. Scripts run as files bootstrap with `bootstrap_from(__file__, depth=N)` before importing it. Config path values are repo-relative strings resolved at load time.

**Experiments are self-contained forks.** Each `experiments/expNNN_*/codes/` is a full copy of the training stack (`train_core.py`, `vae_poe_freq.py`, `graph_occ.py`, `graph_imp.py`, dataloaders, losses). Editing exp059 does not affect exp057/058/060 and vice versa — fixes must be ported deliberately. Checkpoints are not cross-loadable between experiments (architecture keys differ).

**ECADSTAR runs on Windows from WSL.** `active_learning_pi/al/ecadstar.py` converts to Windows paths; ERF/EMC/PEB locations are absolute `C:\Users\muthusamy\...` strings in the AL config. Simulation steps cannot run without that host.

**Large artifacts are untracked**: `datasets/`, `data_multi_norm_robust/`, `checkpoints/`, `*.pt`, `*.peb`. Expected training data is `datasets/data_multifreq_train_norm_unbounded/` (with `dataset_meta.json` + `normalization_stats.json`); AL overlay is `datasets/data_multifreq_al_overlay_exp059/`.

## Architecture map

**Stage 1 — surrogate** (`experiments/exp059_capacity_freq/codes/`): per-modality Gaussian experts (occupancy GNN over the 52-slot PCB graph, spectrum-chain GNN over the 231-bin impedance, heatmap CNN, plus a frequency expert writing private latent dims) fused by Product-of-Experts. Structured latent (exp059: `latent_dim=128`, `heatmap_private_dim=40`, `cond_dim=32`). Heatmap decoder is FiLM-conditioned on frequency at every scale (`use_multiscale_film`). **Modality dropout** during training is what makes the occupancy-only encode path usable at inference — the only path available when proposing a design. Occupancy is decoded to hard top-K binary before heatmap decode (CAD-aligned).

**Stage 2 — latent optimization** (`pipelines/latent/optimize.py`): everything frozen, Adam over `z` per budget `K` with parallel seeds. `_ste_topk` — hard top-K forward, identity gradient — is the load-bearing detail; without it the graph is severed and the search degenerates to random seed sampling. Loss = exceed/gap vs target + peak/track alignment + anti-resonance physics prior + prior/boundary terms keeping `z` inside the aggregate posterior + diversity. Per K it emits the feasible candidate with lowest peak impedance, or an explicit `no_solution.json`.

**Active learning** (`active_learning_pi/al/`): `candidates.py` → `inference_pool.py` (acquisition_mode `gp_error | random | mc`) → `per_k_acquire.py`/`acquisition.py` → `peb_batch.py`/`ecadstar.py` → `ingest_labels.py` → `robust_normalize.py` → `build_overlay.py` → `finetune_run.py`. `gp_error_surrogate.py` is a residual GP over surrogate error (score = `mu`); `decision_report.py` rolls a cycle's numbers into `runs/<run>/DECISION_REPORT.md` + `decision_ledger.json` with PASS/FAIL/UNCERTAIN verdicts.

## Working rules inherited from `.cursor/rules/`

**Ground every claim in numbers** (`numerical-claims.mdc`, always-on). Do not assert that a change improved something, that ranking/uncertainty "works", or that one method beats another without either (1) citing existing artifacts — `DECISION_REPORT.md`, `decision_ledger.json`, `acquisition_rank_quality.json`, `CYCLE_EVAL_REPORT.md`, Spearman/MAE/lift, equal-budget A/B — or (2) proposing a concrete falsifiable check, or (3) explicitly labelling the statement a hypothesis. When suggesting work, attach the metric to watch, the baseline, and the pass criterion. Lead with the number and verdict, then interpretation.

**Keep context lean** (`token-efficiency.mdc`): prefer grep/targeted reads with line ranges over full-file reads; avoid reading `datasets/`, `checkpoints/`, `*.pt` and other `.cursorignore` entries.

## Thesis knowledge graph (`thesis_vault/`)

`tools/build_vault.py` generates an Obsidian vault at `thesis_vault/` — one note per Python
module (AST-derived docstring, classes, functions, CONFIG constants, internal imports as
wikilinks), plus mirrors of `docs/*.md`, one note per experiment (config + `notes.md` +
lineage), per AL run (with decision-ledger verdicts), and per referenced dataset.

```bash
python tools/build_vault.py     # rewrites generated folders; deterministic
```

Obsidian cannot open a vault over `\\wsl.localhost` (watcher throws `EISDIR` — WSL's 9P
filesystem emits no Windows change notifications), so the script mirrors the vault to
`WINDOWS_MIRROR` (`/mnt/c/Users/muthusamy/thesis_vault`) and Obsidian opens that. Each run
first pulls hand-written notes (`00-Thesis/`, `15-Ideas/`, `README.md`) back from the
mirror so Obsidian edits reach git, then pushes generated folders out. Set
`WINDOWS_MIRROR = None` to disable.

It **deletes and rewrites** `10-Concepts`, `12-Archive`, `20-Experiments`, `30-Code`,
`40-Datasets`, `50-Results`, and `Home.md`. It never touches `00-Thesis/` (chapter map,
claim ledger, reading paths), `15-Ideas/`, or `thesis_vault/README.md` — hand-written.
Concept notes are mirrors, so edit `docs/*.md` and regenerate rather than editing the note.

Note naming: unique filename when the stem is unique repo-wide, else the full dotted path
(the experiment forks mean 24 files named `train_vae_simple.py`). Coverage is tracked +
untracked-not-ignored files, so uncommitted work appears, tagged `#uncommitted`.

## Documentation

`docs/README.md` is the curated index (thesis-grade): `framework-overview.md`, `model-architecture.md`, `experiment-lineage.md`, `latent-optimization.md`, `active-learning.md`, `gp-error-surrogate.md`, `dataset.md`, `data-pipeline.md`, `evaluation-metrics.md`, `limitations-and-validity.md`. Implementation chatter and one-off fixes live in `docs/_archive/` and are not for citation. `docs/cursor-handoff.md` holds current state and open tasks; `active_learning_pi/GP_ERROR_SURROGATE_FRAMEWORK.md` is the design-of-record for residual-GP AL. Per-experiment rationale lives in `experiments/<exp>/notes.md`.
