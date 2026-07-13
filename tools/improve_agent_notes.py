#!/usr/bin/env python3
"""Apply improved Agent notes (What / Usage / Config keys) to entry scripts."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# file (relative to repo) -> new module docstring (triple-quoted content only, no quotes)
DOCSTRINGS: dict[str, str] = {
    "pipelines/active_learning/run.py": '''Entry point for the active-learning PI pipeline.

Purpose:
    Drive one AL iteration step or a full cycle: VAE propose → ECADStar simulate → ingest → normalize → evaluate.

Run:
    python pipelines/active_learning/run.py

Agent notes:
    - What: Orchestrates the PI multifreq active-learning loop using ``active_learning_pi.al.pipeline``.
    - Usage: Edit CONFIG below → run the command above. JSON paths live in ``active_learning_pi/config/default.json`` when ``CONFIG_PATH`` is None.
    - Config keys:
        - ``COMMAND`` — step to run: ``cycle`` (full loop), ``generate``, ``infer``, ``simulate``, ``ingest``, ``normalize``, ``evaluate``, ``finetune-hint``
        - ``CONFIG_PATH`` — path to AL JSON config; ``None`` uses default
        - ``ITERATION`` — force iteration index; ``None`` uses latest from disk
        - ``SKIP_SIMULATE`` / ``SKIP_INGEST`` — skip ECADStar or ingest during ``cycle``
        - ``PROPOSE_ONLY`` — if True: only ``generate`` + ``infer`` (no simulation)
''',
    "pipelines/data/data_check_stats.py": '''Comprehensive dataset statistics and quality scoring.

Purpose:
    Compute per-modality stats, skewness, diversity metrics, and an overall quality score;
    optionally writes a markdown snapshot.

Run:
    python pipelines/data/data_check_stats.py

Agent notes:
    - What: QA report for a VAE dataset folder (heatmap / Imp / Occ_map layout).
    - Usage: Set ``DATA_ROOT`` to raw or normalized multifreq path → run script.
    - Config keys:
        - ``DATA_ROOT`` — dataset root with ``heatmap/``, ``Imp/``, ``Occ_map/``
        - ``MODE`` — ``"quick"`` for summary stats; ``"full"`` for detailed analysis
        - ``SAVE_SNAPSHOT`` — write ``data_check_stats.md`` beside the dataset
    - Key symbols: ``calculate_statistics``, ``quick_stats_normalized``
''',
    "pipelines/data/processing_single.py": '''Single-frequency training dataset builder (legacy).

Purpose:
    Scan raw ECADStar ``N_true/`` exports and write per-sample ``sample_*.npy`` heatmap, impedance,
    and 52-d occupancy under a flat dataset layout.

Run:
    python pipelines/data/processing_single.py

Agent notes:
    - What: Legacy single-PI-frequency dataset builder. Prefer ``processing_multifreq.py`` for current work.
    - Usage: Set ``TRAIN_DATA_ROOT`` (output) and raw ``DATA_ROOT`` (or env ``DATA_ROOT``) → run.
    - Config keys:
        - ``TRAIN_DATA_ROOT`` — output root (``heatmap/``, ``Imp/``, ``Occ_map/``)
        - ``DATA_ROOT`` — raw ECAD export folder; ``None`` auto-detects
        - ``MAX_SAMPLES`` — cap samples processed; ``None`` = all
        - ``NUM_WORKERS`` — parallel workers for sample parsing
        - ``CSV_ONLY`` — occupancy-from-CSV mode only (legacy)
        - ``CSV_ONLY_PATH`` — decap CSV folder when ``CSV_ONLY=True``
    - Key symbols: ``create_training_dataset``, ``iter_dataset_samples``
''',
    "pipelines/data/processing_multifreq.py": '''Build multifreq training dataset (layout-centric storage).

Purpose:
    Ingest raw ECADStar multifreq exports into ``layouts/``, ``heatmap/``, ``PI_freq/``, and
    ``manifest.csv`` under a layout store compatible with exp038+ training.

Run:
    python pipelines/data/processing_multifreq.py

Agent notes:
    - What: Primary dataset builder — one row per (layout, MHz anchor) with shared occupancy per layout.
    - Usage: Set ``OUTPUT_ROOT`` and raw ``DATA_ROOT`` → run. Use ``APPEND=True`` to add new MHz only.
    - Config keys:
        - ``OUTPUT_ROOT`` — destination under ``datasets/`` (e.g. ``data_multifreq``)
        - ``DATA_ROOT`` — raw ECADStar export; ``None`` auto-detects
        - ``APPEND`` — add rows without wiping existing output
        - ``APPEND_MHZ_ONLY`` — when appending, restrict to these MHz anchors
    - Key symbols: ``process_multifreq_dataset``, ``TRAIN_DATA_ROOT``
''',
    "pipelines/dataset/extract_subset.py": '''Extract a layout subset from full multifreq into a smaller folder.

Purpose:
    Match layouts present in a reference dataset, then copy or symlink all multifreq rows
    (every MHz anchor) for those layouts from source to destination.

Run:
    python pipelines/dataset/extract_subset.py

Agent notes:
    - What: Build a train/eval split folder by layout ID (not by manifest row filter).
    - Usage: Defaults to dry-run. Set ``EXECUTE=True`` to write ``DST``. Use ``OVERWRITE`` or ``RESUME``, not both.
    - Config keys:
        - ``SRC`` — full multifreq source (e.g. ``datasets/data_multifreq``)
        - ``REF`` — reference with ``layouts/`` to match (e.g. ``data_multifreq_norm``)
        - ``DST`` — output folder to create
        - ``EXECUTE`` — ``False`` = preview only; ``True`` = copy/symlink
        - ``HARD_COPY`` — copy files instead of symlinks
        - ``OVERWRITE`` — wipe ``DST`` before extract
        - ``RESUME`` — fill missing files only
    - Key symbol: ``extract_subset``
''',
    "pipelines/normalize/multifreq.py": '''Multifreq dataset normalization pipeline.

Purpose:
    Read raw multifreq dataset; apply log/global-max normalization; write ``data_multifreq_norm``
    with ``normalization_stats.json`` and updated ``dataset_meta.json``.

Run:
    python pipelines/normalize/multifreq.py

Agent notes:
    - What: Produces VAE-ready normalized tensors from ``pipelines/data/processing_multifreq.py`` output.
    - Usage: Set ``DATA_DIR`` (input) and ``OUTPUT_DIR`` → run. ``APPEND=True`` adds new samples only.
    - Config keys:
        - ``DATA_DIR`` — raw multifreq root
        - ``OUTPUT_DIR`` — normalized output; ``None`` defaults beside input
        - ``APPEND`` — incremental normalize vs full rebuild
    - Key symbols: ``normalize_multifreq``, ``prepare_output_dir``
''',
    "pipelines/latent/optimize.py": '''Latent impedance optimization — gradient search in frozen VAE latent space.

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
''',
    "pipelines/visualize/heatmap.py": '''EM solver MAP file to heatmap visualizer.

Purpose:
    Parse a Z_*.map point cloud, interpolate onto a 64×64 grid with H-shaped board mask,
    and render impedance distribution heatmaps.

Run:
    python pipelines/visualize/heatmap.py

Agent notes:
    - What: Converts ECAD MAP text to numpy heatmap grids and PNG plots.
    - Usage: Set ``MAP_FILE`` in CONFIG → run demo, or import ``plot_heatmap_array`` from other scripts.
    - Config keys:
        - ``MAP_FILE`` — path to ``Z_*.map`` frequency point cloud
    - Key symbols: ``plot_impedance_heatmap_clean``, ``plot_heatmap_array``
''',
    "pipelines/heatmaps/regenerate_mhz_pebs.py": '''Batch-regenerate anchor-frequency PEB files from combined_all.peb.

Purpose:
    Write ``combined_all_{MHz}MHz.peb`` for each MHz in ``ANCHORS_MHZ`` by rewriting PI frequency tags.

Run:
    python pipelines/heatmaps/regenerate_mhz_pebs.py

Agent notes:
    - What: Splits one master PEB into per-anchor-frequency PEB files for ECADStar batch runs.
    - Usage: Point ``INPUT_PEB`` at ``combined_all.peb`` → set ``ANCHORS_MHZ`` → run.
    - Config keys:
        - ``INPUT_PEB`` — source combined PEB under ``data/heatmaps/``
        - ``ANCHORS_MHZ`` — list of MHz values to emit
    - Key symbol: ``write_peb_at_mhz`` (from ``libs.peb.frequency``)
''',
    "pipelines/heatmaps/change_frequency.py": '''Change PI-Distribution frequency in a single .peb file.

Purpose:
    Replace ``EditPIDistribution Frequency="..."`` in a source PEB and write one output file.

Run:
    python pipelines/heatmaps/change_frequency.py

Agent notes:
    - What: Single-frequency PEB rewrite (one MHz). For all anchors use ``regenerate_mhz_pebs.py``.
    - Usage: Set ``INPUT_PEB``, ``SET_FREQ_MHZ``, ``OUTPUT_PEB`` → run.
    - Config keys:
        - ``INPUT_PEB`` — source PEB path
        - ``SET_FREQ_MHZ`` — target inspection frequency (MHz)
        - ``OUTPUT_PEB`` — destination path
''',
    "scrap/orchestration/run_multifreq_sweep_pipeline.py": '''Multifreq heatmap sweep pipeline (generate → simulate → compare).

Purpose:
    End-to-end workflow: VAE sample at fixed K across a MHz sweep, build PEB, optional ECADStar PI
    simulation, move outputs, and comparison report.

Run:
    python scrap/orchestration/run_multifreq_sweep_pipeline.py

Agent notes:
    - What: Orchestrates multifreq heatmap generation and optional Windows ECADStar automation.
    - Usage: Edit CONFIG (model paths, ``K_VALUE``, ``SWEEP``, ECADStar paths) → run on machine with repo + ECADStar access.
    - Config keys:
        - ``EXPERIMENT_DIR`` / ``CHECKPOINT_PATH`` — VAE for generation
        - ``K_VALUE``, ``SWEEP``, ``NUM_SAMPLES`` — layout budget and MHz list
        - ``SKIP_GENERATE`` … ``SKIP_REPORT`` — skip individual pipeline steps
        - ``ECADSTAR_ERF_PATH``, ``ECADSTAR_EMC_OUTPUT_DIR`` — Windows simulation paths
    - Key symbols: ``verify_and_stage_peb_for_ecadstar``, ``step_generate``
''',
    "scrap/generation/generate_samples_and_peb.py": '''Generate VAE samples and PEB for one K.

Purpose:
    Load VAE checkpoint, generate N samples with exactly K active decaps, save ``data_sample_*``
    folders, and write a matching ECADStar ``.peb``.

Run:
    python scrap/generation/generate_samples_and_peb.py

Agent notes:
    - What: Single-K VAE sample export + PEB builder (legacy exp030 path).
    - Usage: Set ``K_VALUE``, ``NUM_SAMPLES``, checkpoint paths, ``OUTPUT_DIR`` → run.
    - Config keys:
        - ``CHECKPOINT_PATH``, ``LATENT_STATS_PATH``, ``MODEL_LATENT_DIM`` — VAE load
        - ``K_VALUE``, ``NUM_SAMPLES``, ``SHARED_TEMP`` — generation
        - ``OUTPUT_DIR``, ``PEB_PATH`` — where ``.npy`` and ``.peb`` are written
    - Key symbols: ``generate_save``, ``main``
''',
    "scrap/generation/generate_samples_and_peb_all_k.py": '''Generate samples and combined PEB for all K (legacy wrapper).

Purpose:
    Loop ``K_MIN``..``K_MAX`` calling single-K generation; stack occupancies into one combined PEB.

Run:
    python scrap/generation/generate_samples_and_peb_all_k.py

Agent notes:
    - What: Multi-K wrapper around ``generate_samples_and_peb`` (legacy). Prefer ``run_all_k.py`` for new sweeps.
    - Usage: Set ``K_MIN``, ``K_MAX``, ``OUT_ROOT``; inherits VAE settings from ``gsp`` module CONFIG.
    - Config keys:
        - ``K_MIN``, ``K_MAX`` — decap budget range
        - ``OUT_ROOT``, ``PEB_OUT_FILE`` — per-K folders and combined PEB output
    - Key symbols: ``main``, ``gsp`` (imported generate module)
''',
    "scrap/comparison/compare_generated_vs_real.py": '''Compare generated vs real for one K.

Purpose:
    Plot heatmap and impedance overlays for one K folder (generated ``.npy`` vs ECADStar ``Real/`` exports).

Run:
    python scrap/comparison/compare_generated_vs_real.py

Agent notes:
    - What: Visual QA — generated VAE samples vs ground-truth PI simulation for a single decap budget.
    - Usage: Set ``K_VALUE`` and ``BASE_GENERATED_DIR`` → run. Also see ``comparison/compare.py``.
    - Config keys:
        - ``K_VALUE`` — decap budget folder ``K{n}`` under base dir
        - ``BASE_GENERATED_DIR`` — root containing ``K1/``, ``K2/``, …
        - ``NUM_SAMPLES`` — how many ``data_sample_*`` to plot
        - ``FREQUENCY_PATH``, ``TARGET_IMPEDANCE_PATH``, ``MASK_PATH`` — shared config arrays
        - ``HEATMAP_OUT_NAME``, ``IMPEDANCE_OUT_NAME`` — output PNG filenames
''',
    "scrap/comparison/compare_generated_vs_real_all_k.py": '''Generated vs real comparison for all K.

Purpose:
    Loop ``K_MIN``..``K_MAX`` and run heatmap+impedance comparison per K folder.

Run:
    python scrap/comparison/compare_generated_vs_real_all_k.py

Agent notes:
    - What: Batch wrapper over ``compare_generated_vs_real`` for K=1..52 (or a subrange).
    - Usage: Set ``K_MIN``, ``K_MAX``, ``GENERATED_BASE_DIR`` → run.
    - Config keys:
        - ``K_MIN``, ``K_MAX`` — inclusive K range
        - ``GENERATED_BASE_DIR`` — root with ``K{n}/`` subfolders
        - ``FAIL_FAST`` — stop on first K that errors
''',
    "scrap/comparison/compare_generated_vs_real_occupancy.py": '''Occupancy checkbox plot for one K.

Purpose:
    Visualize generated 52-slot occupancy vectors (C1..C52) for one K using top-K or threshold policy.

Run:
    python scrap/comparison/compare_generated_vs_real_occupancy.py

Agent notes:
    - What: Renders which decap slots are active per generated sample (bar/checkbox view).
    - Usage: Set ``K_VALUE``, ``BASE_GENERATED_DIR``, ``ACTIVE_POLICY`` → run.
    - Config keys:
        - ``K_VALUE``, ``BASE_GENERATED_DIR`` — which K folder to read
        - ``NUM_SAMPLES`` — rows to plot
        - ``ACTIVE_POLICY`` — ``"topk"`` or ``"threshold"``; ``THRESHOLD`` when threshold mode
        - ``OCCUPANCY_OUT_NAME`` — output PNG name
''',
    "scrap/comparison/compare_generated_vs_real_occupancy_all_k.py": '''Occupancy comparison for all K.

Purpose:
    Loop ``K_MIN``..``K_MAX`` and render occupancy checkbox plots per K folder.

Run:
    python scrap/comparison/compare_generated_vs_real_occupancy_all_k.py

Agent notes:
    - What: Batch wrapper over ``compare_generated_vs_real_occupancy``.
    - Usage: Set ``K_MIN``, ``K_MAX``, ``GENERATED_BASE_DIR`` → run.
    - Config keys:
        - ``K_MIN``, ``K_MAX`` — inclusive K range
        - ``GENERATED_BASE_DIR`` — root with ``K{n}/`` subfolders
        - ``FAIL_FAST`` — stop on first error
''',
    "visualization/visualize_occupancy.py": '''Visualize occupancy vectors from .npy files or folders.

Purpose:
    Render bar charts of 52-slot occupancy vectors; compare multiple folders or single files.

Run:
    python visualization/visualize_occupancy.py

Agent notes:
    - What: Standalone occupancy bar-chart viewer (not tied to VAE comparison pipeline).
    - Usage: Set ``FOLDERS`` (``.npy`` files or directories) and ``OUTPUT_PATH`` → run.
    - Config keys:
        - ``FOLDERS`` — list of paths to ``.npy`` files or folders of ``*.npy``
        - ``OUTPUT_PATH`` — PNG save path; script always saves (no interactive show)
    - Key symbols: ``load_vector``, ``plot_folders``
''',
    "libs/data_creation/occupancy.py": '''Map 52-d decap vectors to 7×8 physical occupancy grids.

Purpose:
    Convert binary decap vectors to board-layout grids (4 invalid cells) with C1..C52 labels.

Run:
    Import only — ``from libs.data_creation.occupancy import create_occupancy_grid``.

Agent notes:
    - What: Shared library mapping 52-d binary vectors to physical 7×8 capacitor grid coordinates.
    - Usage: Import ``create_occupancy_grid`` or call ``csv_to_occupancy_samples`` for batch CSV→npy.
    - Key symbols: ``create_occupancy_grid``, ``LABELS_ORDERED``, ``INVALID_OCC_CELLS``
    - Grid layout: origin lower; bottom row C4,C5,…,C1,C2,C3 per board geometry.
''',
    "experiments/exp038_true_multi/codes/eval_cross_freq.py": '''Evaluate native vs cross-frequency heatmap reconstruction on val set.

Purpose:
    Measure foreground MSE when decoding at native MHz vs cross-anchor MHz (layout-z diagnostic).

Run:
    python experiments/exp038_true_multi/codes/eval_cross_freq.py

Agent notes:
    - What: Cross-frequency generalization eval for exp038 VAE heatmap decoder.
    - Usage: Set ``CHECKPOINT``, ``OFF_ANCHOR_MHZ``, ``MAX_BATCHES`` → run. Writes CSV metrics.
    - Config keys:
        - ``CHECKPOINT`` — ``.pt`` path (default ``checkpoints/last_model.pt``)
        - ``MAX_BATCHES`` — val batches to score; ``0`` = full val set
        - ``OUTPUT_CSV`` — where to write per-MHz MSE rows
        - ``OFF_ANCHOR_MHZ`` — MHz list for off-anchor decode test
''',
    "experiments/exp038_true_multi/codes/eval_val_recon.py": '''Validation reconstruction metrics for exp038.

Purpose:
    Report occupancy BCE, slot accuracy, K-match, and impedance peak MSE on the val set.

Run:
    python experiments/exp038_true_multi/codes/eval_val_recon.py

Agent notes:
    - What: Full val-set reconstruction diagnostic (occ + imp heads) for a saved checkpoint.
    - Usage: Set ``CHECKPOINT_NAME`` in CONFIG → run. Prints per-metric summary to stdout.
    - Config keys:
        - ``CHECKPOINT_NAME`` — filename under ``checkpoints/`` (e.g. ``last_model.pt``)
        - ``MAX_BATCHES`` — limit val batches; ``0`` = all
''',
    "experiments/exp043/codes/vae_poe_freq.py": '''Multi-input VAE with PI_freq Product-of-Experts expert (exp043).

Purpose:
    Extend ``MultiInputVAE`` with a frequency-only PoE expert fused with layout experts for multifreq training.

Run:
    Import only — instantiated by ``train_vae_simple``, ``inference_vae``, ``exp043_eval_common``.

Agent notes:
    - What: Core exp043 model class ``MultiInputVAEPoeFreq`` (shared latent + optional private heatmap dims).
    - Usage: Import and construct with ``latent_dim``, ``cond_dim``, ``use_freq_poe_expert``; not run directly.
    - Key symbols: ``MultiInputVAEPoeFreq``, ``encode_layout_latent``, ``encode_cross_modal``
    - Flag: ``use_freq_poe_expert`` (default True) enables MHz-conditioned PoE fusion.
''',
    "experiments/exp043/codes/synthetic_freq_blend.py": '''Synthetic between-anchor heatmap blending for multifreq training.

Purpose:
    Training augmentation: interpolate heatmap + PI_freq between anchor pairs for off-anchor supervision.

Run:
    Import only — called from ``train_vae_simple._prepare_batch`` when ``synthetic_blend_prob > 0``.

Agent notes:
    - What: Stochastic freq/heatmap blend augmentation during VAE training.
    - Usage: Set ``synthetic_blend_prob`` in experiment ``config.yaml``; batch must include ``heatmap_norm_alt``, ``PI_freq_alt``.
    - Key symbol: ``maybe_apply_synthetic_blend(batch, cfg)``
''',
    "experiments/exp043/codes/metrics_csv_utils.py": '''Dedupe and sort exp043 training metrics CSVs by epoch.

Purpose:
    Remove duplicate epoch rows after training resume; keep last row per epoch, sorted.

Run:
    Import only — ``from experiments.exp043.codes.metrics_csv_utils import update_all_metrics_csv``.

Agent notes:
    - What: Post-training CSV hygiene for ``metrics/loss.csv`` and timing files after checkpoint resume.
    - Usage: Call ``update_all_metrics_csv(metrics_dir)`` after interrupted training or before replotting.
    - Key symbols: ``dedupe_csv_by_epoch``, ``update_all_metrics_csv``, ``insert_epoch_row``
''',
    "experiments/exp043/codes/__init__.py": '''Package marker for exp043 experiment code.

Purpose:
    Namespace for exp043 VAE training, inference, and evaluation modules.

Run:
    Import submodules — not executed directly.

Agent notes:
    - What: Python package root for ``experiments.exp043.codes``.
    - Usage: Import entry scripts by path; do not run this file.
    - Entry scripts: ``train_vae_simple.py``, ``inference_vae.py``, ``evaluate_vae.py``, ``visualize_latent.py``
''',
}


def _replace_docstring(text: str, new_body: str) -> str:
    """Replace first module docstring body."""
    m = re.match(r'(?s)([#!][^\n]*\n)?("""|\'\'\')(.*?)\2', text)
    if not m:
        raise ValueError("No module docstring found")
    prefix = m.group(1) or ""
    quote = m.group(2)
    return f'{prefix}{quote}{new_body}{quote}' + text[m.end():]


def main() -> None:
    for rel, body in DOCSTRINGS.items():
        path = REPO / rel
        if not path.is_file():
            print("skip (missing):", rel)
            continue
        text = path.read_text(encoding="utf-8")
        new_text = _replace_docstring(text, body)
        path.write_text(new_text, encoding="utf-8")
        print("updated", rel)

    # Fix known typo in regenerate_mhz_pebs.py
    peb = REPO / "pipelines/heatmaps/regenerate_mhz_pebs.py"
    t = peb.read_text(encoding="utf-8")
    if 'repo_path("data", "heatmaps")s[2]' in t:
        t = t.replace(
            'SCRIPT_DIR = repo_path("data", "heatmaps")s[2] / "data/heatmaps"',
            'SCRIPT_DIR = repo_path("data", "heatmaps")',
        )
        peb.write_text(t, encoding="utf-8")
        print("fixed SCRIPT_DIR typo in regenerate_mhz_pebs.py")


if __name__ == "__main__":
    main()
