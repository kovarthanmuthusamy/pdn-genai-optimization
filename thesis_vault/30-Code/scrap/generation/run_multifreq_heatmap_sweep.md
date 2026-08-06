---
title: run_multifreq_heatmap_sweep
type: code
path: scrap/generation/run_multifreq_heatmap_sweep.py
group: scrap/generation
loc: 1226
tags: [code, scrap, runnable]
---

# run_multifreq_heatmap_sweep

> Multifreq Heatmap Sweep at Fixed K (or explicit K list).

**Source:** `scrap/generation/run_multifreq_heatmap_sweep.py` · 1226 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/generation/run_multifreq_heatmap_sweep.py`

## Purpose

```text
Multifreq Heatmap Sweep at Fixed K (or explicit K list).

Purpose: Decode VAE samples at multiple PI frequencies for one or more K values for
    PI-Distribution heatmaps; write per-freq folders, combined .peb, sweep manifest,
    and summary plots.
Run: python scrap/generation/run_multifreq_heatmap_sweep.py
Inputs / outputs: CHECKPOINT_PATH, DATA_DIR, SWEEP, K_VALUE (int or list[int])
    → {OUTPUT_ROOT}/freq_*MHz/K{k}/; PEB_OUT_FILE (.peb), sweep_freq_manifest.json;
    optional PEB_COPY_DEST.
Dependencies: torch, matplotlib, numpy; dynamic VAEInference from EXPERIMENT_DIR.
Agent notes:
    - Type: CLI generator (does not replace run_all_k K-sweep workflow)
    - Key symbols: main, run_generate, run_val, exported_freq_mhz_list,
    - Config keys: ``MODE``, ``EXPERIMENT_DIR``, ``CHECKPOINT_PATH``, ``DATA_DIR``, ``OUTPUT_ROOT``, ``DENSE_N_POINTS``, ``K_VALUE``, ``NUM_SAMPLES``, ``SHARED_TEMP``, ``SEED``, ``HEATMAP_ONLY_PEB``, ``PEB_OUT_FILE``, ``POWERBUS``, ``FORCE_CPU``, ``VAL_MAX_BATCHES``, ``BACKGROUND_MARGIN``, ``INFERENCE_MODE``, ``PI_REF_MHZ``, ``CALIBRATE_FG_MAX``, ``INFERENCE_FACTORIZED_ONLY``
      write_sweep_freq_manifest, HEATMAP_ONLY_PEB, K_VALUE, OUTPUT_ROOT
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `MODE` | `'generate'` |
| `EXPERIMENT_DIR` | `'experiments/exp050'` |
| `CHECKPOINT_PATH` | `f'{EXPERIMENT_DIR}/checkpoints/last_model.pt'` |
| `DATA_DIR` | `'data_multi_norm_unbounded'` |
| `OUTPUT_ROOT` | `f'{EXPERIMENT_DIR}/multifreq_heatmap_sweep_30'` |
| `DENSE_N_POINTS` | `24` |
| `NUM_SAMPLES` | `2` |
| `SHARED_TEMP` | `1.5` |
| `SEED` | `42` |
| `RUN_MODE` | `'heatmap'` |
| `HEATMAP_ONLY_PEB` | `True` |
| `PEB_OUT_FILE` | `f'{OUTPUT_ROOT}/pi_distribution_K30_freq_sweep.peb'` |
| `POWERBUS` | `'Power_GND'` |
| `COMPONENTS` | `'IC1_Port1'` |
| `FORCE_CPU` | `False` |
| `VAL_MAX_BATCHES` | `0` |
| `BACKGROUND_MARGIN` | `0.5` |
| `QC_SWEEP` | `True` |
| `INFERENCE_MODE` | `'layout_qc'` |
| `LAYOUT_SOURCE` | `'val'` |
| `ALLOW_RANDOM_LAYOUT` | `False` |
| `RUN_SWEEP_QC_EVAL` | `True` |
| `PI_REF_MHZ` | `200.0` |
| `CALIBRATE_FG_MAX` | `False` |
| `INFERENCE_FACTORIZED_ONLY` | `False` |
| `SWEEP_FREQ_MANIFEST` | `'sweep_frequencies_mhz.json'` |

## Classes

- **`_KGenerateContext`**

## Functions

- **`normalize_k_values(k: int | list[int] | tuple[int, ...])`** — Return sorted unique K values in [0, 52] from a scalar or list.
- **`k_output_tag(k: int | list[int] | tuple[int, ...])`** — Folder tag suffix, e.g. 30 or 10_20_30.
- **`_peb_kind()`** — PEB analysis kind for the active run mode.
- **`is_impedance_run_mode()`**
- **`uses_flat_k_layout(out_root: str | Path | None=None)`** — True when outputs live under ``K{k}/`` (impedance) rather than ``freq_*MHz/K{k}/``.
- **`peb_basename_for_k(k: int | list[int] | tuple[int, ...])`** — Combined multifreq sweep PEB filename for one or more K values.
- **`_k_values_from_generate_summary(root: Path)`** — K values actually generated (authoritative when layout_qc drops K).
- **`_k_values_from_output_dirs(root: Path)`** — Infer K from ``freq_*MHz/K*/data_sample_*`` folders.
- **`exported_k_values(out_root: str | Path | None=None)`** — Effective K list for move/compare — matches what generate + PEB actually contain.
- **`_sync_derived_paths()`** — Keep PEB_OUT_FILE aligned with OUTPUT_ROOT and K_VALUE.
- **`sweep_freq_manifest_path(out_root: str | Path | None=None)`**
- **`write_sweep_freq_manifest(mhz_list: list[float], out_root: str | Path | None=None, *, k_values: list[int] | None=None, extra: dict[str, Any] | None=None)`** — Persist MHz + effective K list from the last generate run (move/compare read this).
- **`load_sweep_freq_mhz_list(out_root: str | Path | None=None)`** — MHz from manifest or generate_summary.csv; None if neither exists.
- **`exported_freq_mhz_list(sweep: str | list | tuple | None=None, out_root: str | Path | None=None)`** — Integer MHz list for move/compare (matches folder tags under OUTPUT_ROOT).
- **`_norm_to_mhz(norm: float)`**
- **`_load_experiment_config(path: Path)`**
- **`_load_vae_inference(experiment_dir: str | None=None)`** — Import VAEInference from the experiment package.
- **`_resolve_vae_inference()`** — Resolve VAEInference for current EXPERIMENT_DIR (re-read after config overrides).
- **`_build_mhz_list(sweep: str | list[int] | tuple[float, ...])`**
- **`_freq_tag(mhz: float)`**
- **`_hm_physical(hm_z: torch.Tensor, engine: Any, mhz: float | None=None)`** — Denorm heatmap tensor to Ω (robust per-MHz, global-max, or legacy z-score).
- **`_impedance_log_from_norm(engine: Any, imp_norm: torch.Tensor)`** — Blended log-Ω impedance ``(B, N)`` from the model's normalized output.
- **`_fg_threshold(engine: Any, cfg: dict[str, Any] | None=None)`** — Foreground threshold in normalized heatmap space.
- **`_fg_mse(recon: torch.Tensor, target: torch.Tensor, fg_thr: float)`** — Foreground MSE per sample (B,) in normalized heatmap space.
- **`_hm_stats(hm_phys: np.ndarray, mask: np.ndarray)`**
- **`_override_inference_data_dir(inf_mod: Any)`** — Use pipeline DATA_DIR for norm stats (resolved under REPO_ROOT).
- **`_load_engine(device: torch.device)`**
- **`_copy_peb_if_configured(peb_file: Path)`**
- **`_build_qc_config()`**
- **`_legacy_marginal_layout_draw(engine: Any, device: torch.device, num_samples: int, k: int)`** — Random occ/imp from marginal z — only when ALLOW_RANDOM_LAYOUT=True.
- **`_load_k_generate_context(engine: Any, device: torch.device, qc_cfg: SweepQCConfig, k: int, *, exp_cfg: dict[str, Any] | None, sweep_val_ld: Any, quiet: bool=False)`** — Load fixed layout / latent bundle for one K (cached per run_generate call).
- **`_decode_at_mhz_k(engine: Any, device: torch.device, qc_cfg: SweepQCConfig, ctx: _KGenerateContext, *, mhz: float, k: int)`** — Run model inference for one (MHz, K) pair using a prepared layout context.
- **`run_generate(engine: Any, mhz_list: list[float], out_root: Path, device: torch.device)`** — Decode at each (MHz, K); save heatmaps, occupancy, and combined PEB.
- **`run_val(engine: Any, mhz_list: list[float], out_root: Path, device: torch.device, max_batches: int)`** — Val heatmap recon at native PI_freq + decode same batch at every sweep MHz.
- **`_path_for_report(path: Path)`** — Repo-relative posix path for README (handles relative vs absolute paths).
- **`write_report(out_root: Path, mhz_list: list[float], modes: list[str])`**
- **`run_from_config()`** — Run sweep using module-level CONFIG constants (also used by pipeline orchestrator).
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp038_true_multi.codes.freq_inference_utils]]
- [[generate_peb]]
- [[heatmap_z_clip]]
- [[peb_copy]]
- [[pi_freq_utils]]
- [[repo_paths]]
- [[sweep_latent_opt_rules]]
- [[sweep_qc_eval]]

## Imported by

- [[build_comparison_report]]
- [[compare]]
- [[move_pi_to_real]]
- [[multifreq_move_and_compare]]
- [[run_multifreq_sweep_pipeline]]
- [[test_sweep_load_exp044]]

## External dependencies

`experiments`, `importlib`, `matplotlib`, `numpy`, `repo_paths`, `src_vae`, `torch`
