---
title: optimize
type: code
path: pipelines/latent/optimize.py
group: pipelines/latent
loc: 677
tags: [code, pipelines, runnable]
---

# optimize

> Latent impedance optimization — gradient search in frozen VAE latent space.

**Source:** `pipelines/latent/optimize.py` · 677 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/latent/optimize.py`

## Purpose

```text
Latent impedance optimization — gradient search in frozen VAE latent space.

Purpose:
    For each decap budget K, optimize latent vectors so the decoded placement meets a target
    impedance mask; writes ``best_latent.npy``, occupancy, and metrics per K.

Run:
    python pipelines/latent/optimize.py

Agent notes:
    - What: Stage-2 inverse design — gradient descent in VAE latent space with straight-through top-K.
    - Usage: Set checkpoint paths, ``K_LIST``, and loss weights in CONFIG → run. Outputs under ``data/latent_runs/``.
    - Config keys:
        - ``EXPERIMENT`` / ``CHECKPOINT_PATH`` — VAE checkpoint for occupancy decode
        - ``SURROGATE_CHECKPOINT_PATH`` / ``USE_SURROGATE`` — impedance surrogate vs VAE imp head
        - ``K_LIST`` — decap budgets to solve
        - ``NUM_STEPS``, ``LR``, ``NUM_CANDIDATE_SEEDS`` — optimizer settings
        - ``SELECT_METRIC`` — rank feasible candidates (default ``max_ohm`` = lowest peak)
    - Key symbols: ``optimize_k``, ``run_optimization``
```

## Constants

| Name | Value |
|------|-------|
| `EXPERIMENT` | `'exp038_true_multi'` |
| `CHECKPOINT_PATH` | `f'experiments/{EXPERIMENT}/checkpoints/last_model.pt'` |
| `SURROGATE_CHECKPOINT_PATH` | `f'experiments/{EXPERIMENT}/checkpoints/surrogate_best.pt'` |
| `USE_SURROGATE` | `_env_flag('LATENT_OPT_USE_SURROGATE', '1')` |
| `K_LIST` | `list(range(1, 26))` |
| `NUM_CANDIDATE_SEEDS` | `32` |
| `SEED_MODE` | `os.getenv('LATENT_OPT_SEED_MODE', 'random').strip().lower()` |
| `OPTIMIZE_BATCH_PER_K` | `True` |
| `NUM_STEPS` | `1200` |
| `GRAD_CLIP` | `5.0` |
| `INIT_SHARED_TEMP` | `0.8` |
| `OBJECTIVE_MODE` | `'gap_max'` |
| `LOSS_SPACE` | `'log'` |
| `Z_L2_WEIGHT` | `0.05` |
| `Z_PRIOR_MODE` | `'agg_posterior'` |
| `SHAPE_REG_WEIGHT` | `2.0` |
| `SHAPE_MIN_STD_DLOG` | `0.12` |
| `SHAPE_TRACK_WEIGHT` | `2.0` |
| `POSTERIOR_BOUNDARY_WEIGHT` | `10.0` |
| `POSTERIOR_SIGMA_LIMIT` | `2.5` |
| `PHYSICS_AR_WEIGHT` | `0.5` |
| `PEAK_LOSS_WEIGHT` | `float(os.getenv('LATENT_OPT_PEAK_LOSS_WEIGHT', '1.0'))` |
| `PEAK_INDEX_WEIGHT` | `float(os.getenv('LATENT_OPT_PEAK_INDEX_WEIGHT', '3.0'))` |
| `PEAK_MAG_WEIGHT` | `float(os.getenv('LATENT_OPT_PEAK_MAG_WEIGHT', '2.0'))` |
| `DUAL_TOPK_WEIGHT` | `float(os.getenv('LATENT_OPT_DUAL_TOPK_WEIGHT', '2.5'))` |
| `IMPEDANCE_NUM_PEAKS` | `int(os.getenv('LATENT_OPT_IMPEDANCE_NUM_PEAKS', '8'))` |
| `IMPEDANCE_TOPK_K` | `int(os.getenv('LATENT_OPT_IMPEDANCE_TOPK_K', '20'))` |
| `DIVERSITY_WEIGHT` | `0.35` |
| `OCC_CONFIDENCE_WEIGHT` | `0.5` |
| `BOUNDARY_MARGIN` | `0.1` |
| `GAP_REWARD_WEIGHT` | `1.0` |
| `EXCEED_WEIGHT` | `25.0` |
| `EXCEED_POWER` | `2.0` |
| `SELECT_METRIC` | `'max_ohm'` |
| `NORMALIZATION_STATS_PATH` | `'datasets/data_multifreq_norm/normalization_stats.json'` |
| `TARGET_IMPEDANCE_PATH` | `'configs/target_impedance.npy'` |
| `OUTPUT_ROOT` | `'data/latent_runs'` |
| `DEVICE` | `'cuda' if torch.cuda.is_available() else 'cpu'` |
| `DTYPE` | `torch.float32` |
| `HIST_KEYS` | `('total', 'exceed', 'gap_mean', 'best_score')` |
| `SOLUTION_PREFIX` | `'best_'` |
| `SOLUTION_LATENT_NAME` | `f'{SOLUTION_PREFIX}latent.npy'` |
| `SOLUTION_METRICS_NAME` | `f'{SOLUTION_PREFIX}metrics.json'` |

## Classes

- **`NormStats`**

## Functions

- **`_env_flag(name: str, default: str='1')`**
- **`_parse_master_seed()`**
- **`experiment_runs_dir(root: Path, experiment: str=EXPERIMENT)`**
- **`resolve_run_dir(root: Path, explicit: str | Path | None=None, experiment: str=EXPERIMENT)`** — Resolve a latent-opt run folder (explicit path or latest under runs/<experiment>/).
- **`norm_stats_from_run_config(run_cfg: dict[str, Any], root: Path)`**
- **`allocate_run_dir(root: Path, experiment: str)`** — Next run folder: data/latent_runs/<experiment>/<0|1|2|...>.
- **`resolve_candidate_seeds()`** — Build per-run candidate seeds and metadata for run_config.json.
- **`_project_root()`**
- **`_load_norm_stats(path: Path)`**
- **`_imp_ch0(imp_norm: torch.Tensor)`**
- **`_curves_from_norm(imp_norm: torch.Tensor, stats: NormStats)`**
- **`_k_prior(K: int, latent_dim: int, device: torch.device, latent_stats: dict | None, per_K: dict | None)`**
- **`_sample_z(K: int, n: int, latent_dim: int, device: torch.device, latent_stats: dict | None, per_K: dict | None)`**
- **`_decode_occ(model: torch.nn.Module, z: torch.Tensor, K: int, device: torch.device)`**
- **`_ste_binary(x: torch.Tensor)`**
- **`_ste_topk(occ_prob: torch.Tensor, K: int)`** — Hard top-K mask in the forward pass, identity gradient to ``occ_prob``.
- **`_forward_imp(model: torch.nn.Module, surrogate: torch.nn.Module | None, z: torch.Tensor, K: int, device: torch.device)`**
- **`_peak_spectrum_loss(imp_curve: torch.Tensor, target_row: torch.Tensor)`** — Peak index + magnitude + dual top-k vs target (log or ohm space).
- **`_anti_resonance_loss(imp: torch.Tensor)`**
- **`_base_loss(imp_curve: torch.Tensor, target: torch.Tensor)`**
- **`_z_prior_loss(z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor)`**
- **`_shape_reg(imp_log: torch.Tensor)`**
- **`_posterior_boundary(z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor)`**
- **`_diversity_loss(occ: torch.Tensor)`**
- **`_is_feasible(imp_curve: torch.Tensor, target: torch.Tensor)`**
- **`_select_score(imp_ohm: torch.Tensor)`**
- **`_compute_loss(z: torch.Tensor, imp_norm: torch.Tensor, occ_prob: torch.Tensor, target_row: torch.Tensor, prior_mu: torch.Tensor, prior_std: torch.Tensor, stats: NormStats, *, physics_ar_weight: float)`** — Returns total (scalar), imp_curve (B,231), imp_log, imp_ohm.
- **`_topk_occ(occ_prob: torch.Tensor, K: int)`**
- **`_save_solution(k_dir: Path, best: dict[str, Any], model: torch.nn.Module, K: int, device: torch.device)`**
- **`_optimize_k(K: int, model: torch.nn.Module, surrogate: torch.nn.Module | None, target_row: torch.Tensor, stats: NormStats, latent_dim: int, latent_stats: dict | None, per_K: dict | None, device: torch.device, physics_ar_weight: float, seeds: list[int])`** — Returns best dict, history, mode, n_candidates.
- **`_write_report(run_dir: Path)`**
- **`main()`**

## Imports

- [[csv_to_occupancy]]
- [[experiments.exp038_true_multi.codes.impedance_spectrum_loss]]
- [[experiments.exp038_true_multi.codes.surrogate_impedance]]
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[repo_paths]]

## External dependencies

`experiments`, `importlib`, `libs`, `numpy`, `repo_paths`, `torch`
