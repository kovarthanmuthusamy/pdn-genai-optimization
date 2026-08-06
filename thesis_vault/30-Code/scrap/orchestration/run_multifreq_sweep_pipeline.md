---
title: run_multifreq_sweep_pipeline
type: code
path: scrap/orchestration/run_multifreq_sweep_pipeline.py
group: scrap/orchestration
loc: 776
tags: [code, scrap, runnable]
---

# run_multifreq_sweep_pipeline

> Multifreq heatmap sweep pipeline (generate → simulate → compare).

**Source:** `scrap/orchestration/run_multifreq_sweep_pipeline.py` · 776 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/orchestration/run_multifreq_sweep_pipeline.py`

## Purpose

```text
Multifreq heatmap sweep pipeline (generate → simulate → compare).

Purpose:
    End-to-end workflow: VAE sample at one or more K values across a MHz sweep, build PEB,
    optional ECADStar PI simulation, move outputs, and comparison report.

Run:
    python scrap/orchestration/run_multifreq_sweep_pipeline.py

Agent notes:
    - What: Orchestrates multifreq heatmap generation and optional Windows ECADStar automation.
    - Usage: Edit CONFIG (model paths, ``K_VALUE``, ``SWEEP``, ECADStar paths) → run on machine with repo + ECADStar access.
    - Config keys:
        - ``EXPERIMENT_DIR`` / ``CHECKPOINT_PATH`` / ``DATA_DIR`` — exp052 + ``data_multifreq_train_norm_unbounded``
        - ``K_VALUE`` — single int (e.g. ``30``) or list (e.g. ``[10, 20, 30]``)
        - ``OUTPUT_ROOT`` — ``…/multifreq_heatmap_sweep_<K>``; if that folder exists on a
          new generate run, Windows-style `` (1)``, `` (2)``, … is appended instead of overwriting
        - ``INFERENCE_MODE`` / ``QC_SWEEP`` / ``LAYOUT_SOURCE`` — QC sweep aligned with latent optimize
        - ``LATENT_RUN_DIR`` — for ``latent_z`` / ``layout_hybrid`` post-opt QC
        - ``SKIP_GENERATE`` … ``SKIP_REPORT`` — skip individual pipeline steps
        - ``RUN_MODE`` — ``heatmap`` | ``impedance`` (mutually exclusive). Drives PEB kind
          (PI-Distribution vs PI-Spectrum), simulate wait pattern, move rename
          (Heatmap_real_ vs Imp_Real), compare, and the *separate* HTML report.
        - ``COMPONENTS`` — CreatePISpectrum IC port(s) used in impedance-mode PEB
        - ``ECADSTAR_ERF_PATH``, ``ECADSTAR_EMC_OUTPUT_DIR`` — Windows simulation paths
    - Key symbols: ``verify_and_stage_peb_for_ecadstar``, ``step_generate``
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `RUN_MODE` | `'heatmap'` |
| `EXPERIMENT_DIR` | `'experiments/exp059_capacity_freq'` |
| `CHECKPOINT_PATH` | `f'{EXPERIMENT_DIR}/checkpoints/last_model.pt'` |
| `DATA_DIR` | `'datasets/data_multifreq_train_norm_unbounded'` |
| `DENSE_N_POINTS` | `24` |
| `OUTPUT_ROOT` | `f'{EXPERIMENT_DIR}/multifreq_heatmap_sweep_temp_1.5'` |
| `NUM_SAMPLES` | `1` |
| `SHARED_TEMP` | `1.5` |
| `SEED` | `42` |
| `HEATMAP_ONLY_PEB` | `True` |
| `POWERBUS` | `'Power_GND'` |
| `COMPONENTS` | `'IC1_Port1'` |
| `FORCE_CPU` | `False` |
| `BACKGROUND_MARGIN` | `0.5` |
| `QC_SWEEP` | `True` |
| `INFERENCE_MODE` | `'layout_qc'` |
| `LAYOUT_SOURCE` | `'val'` |
| `ALLOW_RANDOM_LAYOUT` | `False` |
| `RUN_SWEEP_QC_EVAL` | `True` |
| `PI_REF_MHZ` | `200.0` |
| `CALIBRATE_FG_MAX` | `False` |
| `INFERENCE_FACTORIZED_ONLY` | `False` |
| `ECADSTAR_ERF_PATH` | `'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc\\H-shape.erf'` |
| `ECADSTAR_EMC_OUTPUT_DIR` | `'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc'` |
| `ENGINEER_EXE` | `'C:\\Program Files\\eCADSTAR\\eCADSTAR 2023.0\\Analysis\\bin\\engineer.exe'` |
| `ECADSTAR_WAIT_FOR_PI` | `True` |
| `ECADSTAR_WAIT_TIMEOUT_SEC` | `7200` |
| `ECADSTAR_WAIT_POLL_SEC` | `40` |
| `ECADSTAR_CLOSE_AFTER_BATCH` | `False` |
| `ECADSTAR_CLEAR_LOCK_FILE` | `True` |
| `SOURCE_EMC_DIR` | `ECADSTAR_EMC_OUTPUT_DIR` |
| `MOVE` | `True` |
| `CLEAN_DEST_BEFORE_PASTE` | `True` |
| `SKIP_GENERATE` | `False` |
| `SKIP_SIMULATE` | `False` |
| `SKIP_MOVE` | `False` |
| `SKIP_COMPARE` | `False` |
| `SKIP_REPORT` | `False` |
| `SALVAGE_MISPLACED` | `False` |
| `METRICS_ONLY` | `False` |
| `SIM_METRICS_PRIMARY_COLUMNS` | `('mhz', 'k', 'sample', 'pearson_r', 'max_diff_ohm', 'pattern_mae', 'max_ratio', 'peak_loc…` |
| `RUN_HEATMAP_COMPARE` | `str(RUN_MODE).strip().lower() == 'heatmap'` |
| `RUN_IMPEDANCE_COMPARE` | `str(RUN_MODE).strip().lower() == 'impedance'` |
| `HEATMAP_LEVELS` | `35` |
| `HEATMAP_VMAX_MODE` | `'max'` |
| `HEATMAP_COLORBAR_TICKS` | `11` |
| `HEATMAP_SHARED_COLOR_SCALE` | `False` |
| `HEATMAP_REAL_VMAX_MATCH_GENERATED` | `False` |
| `_OUTPUT_ROOT_LOCKED` | `False` |

## Functions

- **`_k_values()`**
- **`_k_tag()`**
- **`_windows_unique_dir_name(parent: Path, name: str)`** — Pick a free folder name like Windows: ``name``, ``name (1)``, ``name (2)``, …
- **`_resolve_existing_output_dir(parent: Path, base: str)`** — When skipping generate, reuse the newest matching sweep folder if present.
- **`_sync_output_root(*, allocate_new: bool | None=None)`** — Keep OUTPUT_ROOT aligned with K_VALUE.
- **`_peb_out_file()`**
- **`_expected_peb_name()`**
- **`_peb_md5(path: Path)`**
- **`verify_and_stage_peb_for_ecadstar()`** — Ensure ECADStar loads the same PEB that generate wrote (name + content).
- **`_parse_pi_number(name: str)`**
- **`_peb_group_count(peb_path: Path)`** — Count ``<Group>`` entries in a PEB (authoritative expected PI count).
- **`_expected_pi_count(freq_list: list[int] | None)`** — Expected PI folders = PEB group count, else effective K × freq × samples.
- **`_pi_folder(emc_dir: Path, pi_num: int)`**
- **`_pi_ready_patterns()`** — Files that signal a finished PI output for the active run mode.
- **`_pi_map_mtime(pi_dir: Path)`**
- **`_pi_ready_after(emc_dir: Path, pi_num: int, since_ts: float)`**
- **`wait_for_pi_outputs(freq_list: list[int], *, batch_start_ts: float)`** — Poll EMC until PI-1..PI-N have fresh output files (batch simulation finished).
- **`_ecadstar_lock_path()`**
- **`clear_ecadstar_lock()`** — Remove stale eCADSTAR design lock (.rlk) left after a forced close.
- **`close_ecadstar_piemi()`** — Close eCADSTAR PI/EMI after batch completes without leaving a stale .rlk lock.
- **`_apply_sweep_config(mod)`** — Push pipeline CONFIG into run_multifreq_heatmap_sweep module globals.
- **`_apply_move_config(mod)`**
- **`_apply_compare_config(mc)`** — Push heatmap compare visualization settings into compare.py.
- **`_apply_all_config()`**
- **`_windows_path_str(path: Path | str)`**
- **`_resolve_windows_path(path_str: str)`**
- **`run_ecadstar_batch(peb_path: Path, groot: Path)`** — Load Batch in PI/EMI via the native headless CLI (engineer.exe --batch).
- **`step_generate()`**
- **`step_simulate()`**
- **`step_sim_metrics_only()`** — Write sim_compare_metrics.* only (gen vs ECADStar Real/ — no plots).
- **`step_move_compare_report()`**
- **`_print_config_summary()`**
- **`main()`**

## Imports

- [[heatmap_sim_metrics]]
- [[multifreq_move_and_compare]]
- [[pipelines.dataset_sim.ecadstar]]
- [[repo_paths]]
- [[run_multifreq_heatmap_sweep]]

## External dependencies

`pipelines`, `repo_paths`
