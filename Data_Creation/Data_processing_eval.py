"""Create a held-out (evaluation) dataset from raw data.

Goal
----
Build an evaluation dataset using ONLY decap/occupancy combinations that are NOT
present in an already processed training dataset.

Workflow
--------
1) Load all occupancy vectors from the processed training dataset (Occ_map/).
2) Iterate raw samples (heatmap + impedance + decap vector).
3) Convert each raw decap vector -> occupancy vector.
4) Keep only samples whose occupancy combination is not in the training set.
5) Process & save heatmap/impedance/occupancy arrays for the kept samples.

Output structure (same as training dataset)
------------------------------------------
<out_root>/
  heatmap/   sample_*.npy
  Imp/       sample_*.npy
  Occ_map/   sample_*.npy
  manifest.csv

Run
---
Edit the "EVALUATION DATASET CONFIGURATION" block below (or set DATA_ROOT), then run:
    /home/ubuntu/venv-cgan/bin/python Data_Creation/Data_processing_eval.py

Notes
-----
- Combination check is based on the binary occupancy vector (52,) (values > 0.5).
- By default, this script keeps unique combinations only (one sample per combo).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from csv_to_occupancy import create_occupancy_vector
from heatmap import create_Heatmaps, load_mask_board
from impedance import EXPECTED_IMP_LENGTH, read_impedance_file

# Reuse the raw-folder iteration logic (including the newer Raw/ layout support)
from Data_processing import iter_dataset_samples


SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_DIR = SCRIPT_DIR.parent
FRAME_PATH = REPO_DIR / "configs" / "binary_mask.npy"

# ============================================================
# EVALUATION DATASET CONFIGURATION (edit these)
# ============================================================
# Raw dataset root (the "Raw" folder). Prefer the DATA_ROOT env var.
RAW_ROOT = next(
    (
        Path(p)
        for p in [
            os.getenv("DATA_ROOT"),
            "/mnt/c/Users/muthusamy/Desktop/Raw",
        ]
        if p and Path(p).exists()
    ),
    None,
)

# Already processed training dataset root (must contain heatmap/ Imp/ Occ_map/).
TRAIN_ROOT = REPO_DIR / "datasets" / "data"

# Output root for the held-out evaluation dataset.
OUT_ROOT = REPO_DIR / "datasets" / "data_eval"

# Limit how many held-out samples to save (None = all held-out found).
MAX_OUT = None

# Processing options
WORKERS = min(48, os.cpu_count() or 1)
OVERWRITE_OUTPUT = False
RESUME_IF_NONEMPTY = True  # if OUT_ROOT exists, skip already written sample_*.npy files
UNIQUE_BY_COMBO = True  # keep only one sample per unique occupancy combo
VERBOSE = True


_WORKER_MASK_BOARD = None


def _init_worker(frame_path: Path):
    global _WORKER_MASK_BOARD
    _WORKER_MASK_BOARD = load_mask_board(frame_path)


def _process_sample(task):
    """Process one held-out sample.

    Task tuple:
        (idx, sample_dict, hm_dir, imp_dir, occ_dir, verbose)

    Returns:
        (idx, success: bool, error: str | None)
    """
    idx, sample, hm_dir, imp_dir, occ_dir, verbose = task
    try:
        heatmap_path = Path(sample["heatmap_path"])
        if heatmap_path.is_dir():
            # Raw layout typically: PI-*/Power_GND/*.map
            search_dir = heatmap_path / "Power_GND"
            if not search_dir.is_dir():
                search_dir = heatmap_path

            candidates = sorted(search_dir.glob("*.map"), key=lambda p: p.name.lower())
            if not candidates:
                candidates = sorted(heatmap_path.rglob("*.map"), key=lambda p: str(p).lower())
            if not candidates:
                return (idx, False, f"No .map found in {heatmap_path}")
            heatmap_path = candidates[0]

        impedance_path = Path(sample["impedance_path"])
        if impedance_path.is_dir():
            # Raw layout typically: PI-*/Power_GND/*IC1*.csv
            search_dir = impedance_path / "Power_GND"
            if not search_dir.is_dir():
                search_dir = impedance_path

            candidates = [
                p for p in search_dir.glob("*.csv") if p.is_file() and "ic1" in p.name.lower()
            ]
            if not candidates:
                candidates = [p for p in impedance_path.rglob("*.csv") if p.is_file() and "ic1" in p.name.lower()]
            if not candidates:
                return (idx, False, f"No IC1*.csv found in {impedance_path}")
            impedance_path = sorted(candidates, key=lambda p: p.name.lower())[0]

        stacked = create_Heatmaps(heatmap_path, mask_board=_WORKER_MASK_BOARD, verbose=verbose)

        imp = read_impedance_file(impedance_path)
        if imp is None:
            return (idx, False, "Failed to read impedance")
        if len(imp) != EXPECTED_IMP_LENGTH:
            return (idx, False, f"Invalid impedance length: got {len(imp)}, expected {EXPECTED_IMP_LENGTH}")

        name = f"sample_{idx + 1}.npy"
        np.save(hm_dir / name, stacked)
        np.save(imp_dir / name, imp.reshape(-1, 1))

        # Prefer pre-computed occupancy vectors (to match the requested workflow).
        occ_vec = sample.get("occ_vector")
        if occ_vec is None:
            occ_vec = create_occupancy_vector(sample["decap_vector"])
        np.save(occ_dir / name, np.asarray(occ_vec, dtype=np.float32).reshape(-1))

        return (idx, True, None)
    except Exception as e:
        return (idx, False, str(e)[:200])


def _occ_to_key(vec: np.ndarray) -> bytes:
    """Convert occupancy vector (52,) to a compact hashable key."""
    arr = np.asarray(vec).reshape(-1)
    if arr.size != 52:
        raise ValueError(f"Expected occupancy vector length 52, got {arr.size}")
    bits = (arr > 0.5).astype(np.uint8)
    return np.packbits(bits).tobytes()  # 52 bits -> 7 bytes


def _load_training_combo_keys(train_root: Path, verbose: bool = True) -> set[bytes]:
    occ_dir = Path(train_root) / "Occ_map"
    if not occ_dir.exists():
        raise FileNotFoundError(f"Training Occ_map directory not found: {occ_dir}")

    occ_files = sorted(occ_dir.glob("sample_*.npy"))
    if not occ_files:
        raise FileNotFoundError(f"No occupancy samples found in: {occ_dir}")

    keys: set[bytes] = set()
    for p in occ_files:
        vec = np.load(p)
        keys.add(_occ_to_key(vec))

    if verbose:
        print(f"Loaded {len(occ_files)} training occupancy files")
        print(f"Unique training combinations: {len(keys)}")

    return keys


@dataclass
class SelectionStats:
    scanned: int = 0
    kept: int = 0
    skipped_in_train: int = 0
    skipped_duplicate_combo: int = 0
    errors: int = 0


def _select_heldout_samples(
    raw_root: Path,
    training_keys: set[bytes],
    *,
    max_out: int | None,
    unique_by_combo: bool,
    verbose: bool,
) -> tuple[list[dict], SelectionStats]:
    stats = SelectionStats()
    selected: list[dict] = []
    seen_keys: set[bytes] = set()

    # Iterate raw samples but do not process heatmap/impedance yet.
    for sample in iter_dataset_samples(raw_root, verbose=False, resolve_paths=False):
        stats.scanned += 1
        try:
            occ_vec = create_occupancy_vector(sample["decap_vector"])
            key = _occ_to_key(occ_vec)
        except Exception:
            stats.errors += 1
            continue

        if key in training_keys:
            stats.skipped_in_train += 1
            continue

        if unique_by_combo and key in seen_keys:
            stats.skipped_duplicate_combo += 1
            continue

        sample["occ_vector"] = occ_vec
        selected.append(sample)
        stats.kept += 1
        if unique_by_combo:
            seen_keys.add(key)

        if max_out is not None and stats.kept >= max_out:
            break

        if verbose and stats.scanned % 5000 == 0:
            print(
                f"Scanned {stats.scanned} raw samples | kept {stats.kept} | "
                f"skipped-in-train {stats.skipped_in_train} | errors {stats.errors}"
            )

    return selected, stats


def _write_manifest(out_root: Path, selected: list[dict]):
    import csv

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eval_sample_id",
                "source_folder",
                "pi_number",
                "decap_index",
                "heatmap_path",
                "impedance_path",
                "occ_key_hex",
            ]
        )
        for idx, s in enumerate(selected):
            occ_key_hex = _occ_to_key(s["occ_vector"]).hex()
            w.writerow(
                [
                    idx + 1,
                    s.get("source_folder"),
                    s.get("pi_number"),
                    s.get("decap_index"),
                    str(s.get("heatmap_path")),
                    str(s.get("impedance_path")),
                    occ_key_hex,
                ]
            )


def create_evaluation_dataset(
    *,
    raw_root: Path,
    train_root: Path,
    out_root: Path,
    max_out: int | None,
    workers: int,
    overwrite: bool,
    unique_by_combo: bool,
    verbose: bool,
) -> int:
    if not FRAME_PATH.exists():
        raise FileNotFoundError(f"Mask/frame file not found: {FRAME_PATH}")

    raw_root = Path(raw_root)
    train_root = Path(train_root)
    out_root = Path(out_root)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root not found: {raw_root}")
    if not train_root.exists():
        raise FileNotFoundError(f"Training dataset root not found: {train_root}")

    if overwrite and out_root.exists():
        shutil.rmtree(out_root)

    if out_root.exists() and any(out_root.iterdir()) and not overwrite:
        if not RESUME_IF_NONEMPTY:
            raise FileExistsError(
                f"Output root is not empty: {out_root}. Set OVERWRITE_OUTPUT=True, set RESUME_IF_NONEMPTY=True, or choose a new OUT_ROOT."
            )
        if verbose:
            print(f"Output root already exists; resuming in-place: {out_root}")

    print("=" * 60)
    print("EVALUATION DATASET CREATION")
    print("=" * 60)
    print(f"Raw root:        {raw_root}")
    print(f"Training root:   {train_root}")
    print(f"Output root:     {out_root}")
    print(f"Max out samples: {max_out if max_out is not None else 'All held-out'}")
    print(f"Workers:         {workers}")
    print(f"Unique by combo: {unique_by_combo}")
    print("=" * 60)

    print("Loading training combinations...")
    training_keys = _load_training_combo_keys(train_root, verbose=verbose)

    print("Selecting held-out raw samples (no heavy processing yet)...")
    selected, stats = _select_heldout_samples(
        raw_root,
        training_keys,
        max_out=max_out,
        unique_by_combo=unique_by_combo,
        verbose=verbose,
    )

    print("\nSelection summary")
    print(f"  Scanned raw samples:     {stats.scanned}")
    print(f"  Kept (held-out):         {stats.kept}")
    print(f"  Skipped (in training):   {stats.skipped_in_train}")
    print(f"  Skipped (dup combo):     {stats.skipped_duplicate_combo}")
    print(f"  Errors (bad rows/files): {stats.errors}")

    if not selected:
        print("No held-out samples found; nothing to do.")
        return 0

    hm_dir = out_root / "heatmap"
    imp_dir = out_root / "Imp"
    occ_dir = out_root / "Occ_map"

    # Create output folders only when we actually have something to write.
    hm_dir.mkdir(parents=True, exist_ok=True)
    imp_dir.mkdir(parents=True, exist_ok=True)
    occ_dir.mkdir(parents=True, exist_ok=True)

    # Always (re)write manifest for the full selected set.
    _write_manifest(out_root, selected)

    # If resuming, skip tasks whose output files already exist.
    tasks = []
    skipped_existing = 0
    for i, s in enumerate(selected):
        name = f"sample_{i + 1}.npy"
        if RESUME_IF_NONEMPTY and not overwrite:
            if (hm_dir / name).exists() and (imp_dir / name).exists() and (occ_dir / name).exists():
                skipped_existing += 1
                continue
        tasks.append((i, s, hm_dir, imp_dir, occ_dir, verbose))

    if verbose and skipped_existing:
        print(f"Resuming: skipping already processed samples: {skipped_existing}")

    print("\nProcessing held-out samples (heatmap + impedance + occupancy)...")

    if not tasks:
        print("Nothing to process (all selected samples already exist).")
        print(f"  Output saved to:        {out_root}")
        return 0

    results: list[bool] = []

    workers = max(1, int(workers))
    if workers == 1:
        _init_worker(FRAME_PATH)
        for idx, t in enumerate(tasks):
            out_idx, success, err = _process_sample(t)
            results.append(bool(success))
            if not success and verbose:
                print(f"  ✗ sample_{out_idx + 1}: {err}")
            if verbose and ((idx + 1) % max(1, len(tasks) // 10) == 0 or (idx + 1) == len(tasks)):
                print(f"  Progress: {idx + 1}/{len(tasks)}")
    else:
        with Pool(processes=workers, initializer=_init_worker, initargs=(FRAME_PATH,)) as pool:
            completed = 0
            for out_idx, success, err in pool.imap_unordered(_process_sample, tasks):
                results.append(bool(success))
                completed += 1
                if not success and verbose:
                    print(f"  ✗ sample_{out_idx + 1}: {err}")
                if verbose and (completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks)):
                    print(f"  Progress: {completed}/{len(tasks)}")

    success_count = sum(1 for r in results if r)
    print("\nDone")
    print(f"  Successfully processed: {success_count}/{len(selected)}")
    print(f"  Output saved to:        {out_root}")

    return success_count
if __name__ == "__main__":
    if RAW_ROOT is None:
        raise SystemExit(
            "Raw root not found. Set DATA_ROOT env var to your Raw folder, "
            "or edit RAW_ROOT in this script."
        )

    print("=" * 60)
    print("EVALUATION DATASET CONFIGURATION")
    print("=" * 60)
    print(f"Raw root:        {RAW_ROOT}")
    print(f"Training root:   {TRAIN_ROOT}")
    print(f"Output root:     {OUT_ROOT}")
    print(f"Max out samples: {MAX_OUT if MAX_OUT is not None else 'All held-out'}")
    print(f"Workers:         {WORKERS}")
    print(f"Overwrite:       {OVERWRITE_OUTPUT}")
    print(f"Unique by combo: {UNIQUE_BY_COMBO}")
    print(f"Verbose:         {VERBOSE}")
    print("=" * 60)

    create_evaluation_dataset(
        raw_root=RAW_ROOT,
        train_root=TRAIN_ROOT,
        out_root=OUT_ROOT,
        max_out=MAX_OUT,
        workers=WORKERS,
        overwrite=OVERWRITE_OUTPUT,
        unique_by_combo=UNIQUE_BY_COMBO,
        verbose=VERBOSE,
    )
