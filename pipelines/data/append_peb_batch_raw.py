#!/usr/bin/env python3
"""Append multifreq rows from ECADStar PEB-batch Raw (PI-1..PI-N) into an existing dataset.

Run:
    python pipelines/data/append_peb_batch_raw.py"""
from __future__ import annotations

import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

from repo_paths import REPO_ROOT as _REPO_ROOT, setup_path
setup_path()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from repo_paths import repo_path
from libs.dataset_meta import write_dataset_meta
from src_vae.others.multifreq_anchors import anchors_to_freq_labels
from src_vae.others.multifreq_layout_store import (
    layout_occ_path,
    load_manifest_rows,
    manifest_design_ids,
    manifest_path,
)

# Reuse Raw scanning and sample writers from the standard multifreq builder.
import pipelines.data.processing_multifreq as pmf  # noqa: E402

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/data/append_peb_batch_raw.py
# =============================================================================

OUTPUT_ROOT = repo_path("datasets") / "data_multifreq_train"

_DATA_ROOT_CANDIDATES = [
    os.getenv("DATA_ROOT"),
    r"C:\Users\muthusamy\Desktop\Raw",
    "/mnt/c/Users/muthusamy/Desktop/Raw",
]
DEFAULT_DATA_ROOT = next((Path(p) for p in _DATA_ROOT_CANDIDATES if p and Path(p).exists()), None)
DATA_ROOT: str | Path | None = None

# PEB batch map from filter_combinations.py (peb_row → design_id)
DECAP_INDEX_MAP = repo_path("data/heatmaps") / "decap_index_map.csv"

# Filtered decap combinations (row i = peb_row i in the PEB)
DECAP_CSV = repo_path("data/heatmaps") / "all_combinations.csv"

# MHz anchors present in Raw (heatmap_{MHz}/ subdirs)
ONLY_FREQS_MHZ: list[float] = [350.0, 370.0, 390.0, 420.0, 450.0]

# Skip (design_id, freq) pairs already in output manifest
SKIP_EXISTING_FREQ_ROWS = True

# False = dry-run (scan + report only); True = write heatmaps and manifest rows
EXECUTE = True

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "64"))

# =============================================================================


def load_peb_batch_map(map_csv: Path) -> dict[int, dict[str, int | str]]:
    """Map ECADStar batch PI number (1-based) → peb_row, design_id, original_decap_index."""
    if not map_csv.is_file():
        raise FileNotFoundError(f"Missing decap index map: {map_csv}")
    by_batch: dict[int, dict[str, int | str]] = {}
    with map_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            peb_row = int(row["peb_row"])
            batch_pi = peb_row + 1
            design_id = str(row["design_id"])
            if batch_pi in by_batch and by_batch[batch_pi]["design_id"] != design_id:
                raise ValueError(
                    f"Duplicate batch PI {batch_pi}: {by_batch[batch_pi]['design_id']} vs {design_id}",
                )
            by_batch[batch_pi] = {
                "peb_row": peb_row,
                "design_id": design_id,
                "original_decap_index": int(row["original_decap_index"]),
            }
    return by_batch


def load_layout_meta(data_dir: Path) -> dict[str, dict[str, int]]:
    """design_id → pi_number, decap_index from existing output manifest."""
    meta: dict[str, dict[str, int]] = {}
    for row in load_manifest_rows(data_dir):
        did = row.get("design_id")
        if not did or did in meta:
            continue
        try:
            meta[str(did)] = {
                "pi_number": int(row["pi_number"]),
                "decap_index": int(row["decap_index"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return meta


def load_decap_vectors_by_peb_row(decap_csv: Path, n_rows: int) -> np.ndarray:
    if not decap_csv.is_file():
        raise FileNotFoundError(f"Missing decap CSV: {decap_csv}")
    df = pd.read_csv(decap_csv, header=None)
    vectors = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    if len(vectors) < n_rows:
        raise ValueError(
            f"{decap_csv} has {len(vectors)} rows but decap_index_map expects {n_rows}",
        )
    return vectors


def existing_design_freq_pairs(data_dir: Path) -> set[tuple[str, str]]:
    return {
        (str(r["design_id"]), str(r["freq_label"]))
        for r in load_manifest_rows(data_dir)
        if r.get("design_id") and r.get("freq_label")
    }


def _sync_freq_tables(only_freq_mhz: list[float]) -> None:
    pmf._refresh_freq_tables(only_mhz=only_freq_mhz)


def iter_peb_batch_samples(
    data_root: Path,
    *,
    peb_map: dict[int, dict[str, int | str]],
    layout_meta: dict[str, dict[str, int]],
    decap_vectors: np.ndarray,
    allowed_design_ids: set[str],
    allowed_freq_labels: set[str],
    skip_existing: set[tuple[str, str]],
    verbose: bool = True,
) -> list[dict]:
    """Collect append candidates from batch-named PI folders."""
    imp_dir = pmf._find_subdir(data_root, ["imp", "Imp", "IMP"])
    if imp_dir is None:
        raise FileNotFoundError(f"No imp/ under {data_root}")

    imp_pi_dirs = pmf._list_pi_dirs(imp_dir)
    freq_indices: list[int] = []
    heatmap_pi_dirs: list[dict[int, Path]] = []

    for freq_idx, subdir_name in enumerate(pmf.HEATMAP_SUBDIR_NAMES):
        freq_label = pmf.FREQ_LABELS[freq_idx]
        if freq_label not in allowed_freq_labels:
            continue
        hm_root = data_root / subdir_name
        if not hm_root.is_dir():
            if verbose:
                print(f"  Warning: missing {subdir_name}/ ({freq_label})")
            continue
        freq_indices.append(freq_idx)
        heatmap_pi_dirs.append(pmf._list_pi_dirs(hm_root))

    if not freq_indices:
        raise FileNotFoundError(
            f"No heatmap subdirs for {sorted(allowed_freq_labels)} under {data_root}",
        )

    samples: list[dict] = []
    missing_imp = missing_hm = missing_meta = not_in_manifest = skipped = 0

    eligible_batch = sorted(set(peb_map) & set(imp_pi_dirs))
    if verbose:
        print(f"  Batch PI in imp/      : {len(eligible_batch):,} / {len(peb_map):,}")

    for batch_i, batch_pi in enumerate(eligible_batch):
        if verbose and batch_i > 0 and batch_i % 5000 == 0:
            print(f"  Scanning batch PI… {batch_i:,}/{len(eligible_batch):,}")
        entry = peb_map[batch_pi]
        design_id = str(entry["design_id"])
        peb_row = int(entry["peb_row"])

        if design_id not in allowed_design_ids:
            not_in_manifest += 1
            continue
        meta = layout_meta.get(design_id)
        if meta is None:
            missing_meta += 1
            continue

        imp_pi_dir = imp_pi_dirs[batch_pi]
        imp_path = pmf.resolve_impedance_ic1(imp_pi_dir)
        if imp_path is None:
            missing_imp += 1
            continue

        decap_vector = decap_vectors[peb_row]

        for local_i, freq_idx in enumerate(freq_indices):
            freq_label = pmf.FREQ_LABELS[freq_idx]
            if (design_id, freq_label) in skip_existing:
                skipped += 1
                continue

            pi_dir = heatmap_pi_dirs[local_i].get(batch_pi)
            if pi_dir is None:
                missing_hm += 1
                continue
            hm_path = pmf.resolve_heatmap_map(pi_dir, freq_label)
            if hm_path is None:
                missing_hm += 1
                continue

            samples.append(
                {
                    "design_id": design_id,
                    "source_folder": "peb_batch",
                    "pi_number": meta["pi_number"],
                    "decap_index": meta["decap_index"],
                    "batch_pi": batch_pi,
                    "peb_row": peb_row,
                    "freq_label": freq_label,
                    "freq_mhz": pmf.FREQ_HZ[freq_label] / 1e6,
                    "heatmap_path": hm_path,
                    "impedance_path": imp_path,
                    "decap_vector": decap_vector,
                },
            )

    if verbose:
        print(f"  Batch map entries     : {len(peb_map):,}")
        print(f"  Manifest layouts      : {len(allowed_design_ids):,}")
        print(f"  Rows collected        : {len(samples):,}")
        print(f"  Skipped existing freq : {skipped:,}")
        print(f"  Missing imp/PI-N      : {missing_imp:,}")
        print(f"  Missing heatmap/PI-N  : {missing_hm:,}")
        print(f"  design_id not in out  : {not_in_manifest:,}")
        print(f"  design_id no meta     : {missing_meta:,}")

    return samples


def validate_occ_alignment(
    output_root: Path,
    peb_map: dict[int, dict[str, int | str]],
    decap_vectors: np.ndarray,
    *,
    max_check: int = 50,
) -> int:
    """Spot-check decap CSV rows against layouts/*/occ.npy when present."""
    mismatches = 0
    checked = 0
    for batch_pi in sorted(peb_map):
        if checked >= max_check:
            break
        entry = peb_map[batch_pi]
        design_id = str(entry["design_id"])
        peb_row = int(entry["peb_row"])
        occ_path = layout_occ_path(output_root, design_id)
        if not occ_path.is_file():
            continue
        occ = np.load(occ_path).reshape(-1)
        if not np.allclose(occ, decap_vectors[peb_row]):
            mismatches += 1
        checked += 1
    return mismatches


def append_peb_batch_raw(
    *,
    output_root: Path,
    data_root: Path,
    decap_index_map: Path,
    decap_csv: Path,
    only_freq_mhz: list[float],
    execute: bool,
    skip_existing_freq_rows: bool = True,
    num_workers: int | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    output_root = Path(output_root)
    data_root = Path(data_root)

    if not manifest_path(output_root).is_file():
        raise FileNotFoundError(
            f"Output manifest missing: {manifest_path(output_root)} — run extract_subset first",
        )

    _sync_freq_tables(only_freq_mhz)
    allowed_freq_labels = set(anchors_to_freq_labels(only_freq_mhz))

    peb_map = load_peb_batch_map(decap_index_map)
    layout_meta = load_layout_meta(output_root)
    allowed_design_ids = manifest_design_ids(output_root)
    decap_vectors = load_decap_vectors_by_peb_row(decap_csv, n_rows=len(peb_map))

    missing_meta = sorted(allowed_design_ids - set(layout_meta))
    if missing_meta:
        raise ValueError(
            f"{len(missing_meta)} manifest design_id(s) lack pi_number/decap_index "
            f"(first: {missing_meta[0]})",
        )

    skip_existing = existing_design_freq_pairs(output_root) if skip_existing_freq_rows else set()
    occ_bad = validate_occ_alignment(output_root, peb_map, decap_vectors)
    if occ_bad:
        print(f"  Warning: {occ_bad} occ spot-check mismatch(es) vs {decap_csv.name}")

    if verbose:
        print(f"Raw root    : {data_root}")
        print(f"Output      : {output_root}")
        print(f"Map         : {decap_index_map} ({len(peb_map):,} batch PI entries)")
        print(f"Decap CSV   : {decap_csv}")
        print(f"MHz anchors : {only_freq_mhz}")
        print(f"Heatmap dirs: {pmf.HEATMAP_SUBDIR_NAMES}")

    all_samples = iter_peb_batch_samples(
        data_root,
        peb_map=peb_map,
        layout_meta=layout_meta,
        decap_vectors=decap_vectors,
        allowed_design_ids=allowed_design_ids,
        allowed_freq_labels=allowed_freq_labels,
        skip_existing=skip_existing,
        verbose=verbose,
    )

    unique_layouts = len({s["design_id"] for s in all_samples})
    freq_counts = Counter(s["freq_label"] for s in all_samples)
    expected_per_freq = unique_layouts

    if verbose:
        print(f"\nUnique layouts in batch: {unique_layouts:,}")
        print("Per-frequency row counts:")
        for fl in pmf.FREQ_LABELS:
            n = freq_counts.get(fl, 0)
            print(f"  {fl}: {n:,}")

    summary = {
        "batch_map_entries": len(peb_map),
        "rows_collected": len(all_samples),
        "unique_layouts": unique_layouts,
        "execute": int(execute),
    }

    if not all_samples:
        print("\nNo rows to append.")
        return summary

    if not execute:
        print("\n[DRY-RUN] Set EXECUTE = True to write heatmaps and manifest rows.")
        return summary

    hm_dir = output_root / "heatmap"
    layouts_dir = output_root / "layouts"
    pifreq_dir = output_root / "PI_freq"
    for d in (hm_dir, layouts_dir, pifreq_dir):
        d.mkdir(parents=True, exist_ok=True)

    start_idx = pmf._next_sample_index(hm_dir)
    if start_idx > 0 and verbose:
        print(f"\nContinuing sample index from {start_idx + 1}")

    random.seed(42)
    selected = list(all_samples)
    random.shuffle(selected)

    tasks = [
        (start_idx + i, s, hm_dir, layouts_dir, pifreq_dir)
        for i, s in enumerate(selected)
    ]
    n_workers = num_workers if num_workers and num_workers > 0 else min(48, os.cpu_count() or 1)
    print(f"\nRunning with {n_workers} workers...")

    manifest_rows: list[dict] = []
    ok = 0

    if n_workers == 1:
        pmf._init_worker(pmf.FRAME_PATH)
        for task in tasks:
            idx, success, error, row = pmf._process_sample(task)
            if success and row:
                manifest_rows.append(row)
                ok += 1
            elif verbose and not success:
                print(f"  x sample_{idx + 1}: {error}")
    else:
        try:
            from multiprocessing import Pool

            with Pool(processes=n_workers, initializer=pmf._init_worker, initargs=(str(pmf.FRAME_PATH),)) as pool:
                for i, out in enumerate(pool.imap_unordered(pmf._process_sample, tasks)):
                    idx, success, error, row = out
                    if success and row:
                        manifest_rows.append(row)
                        ok += 1
                    if verbose and (i + 1) % max(1, len(tasks) // 10) == 0:
                        print(f"  Progress: {i + 1}/{len(tasks)}")
        except Exception as e:
            print(f"Parallel failed ({e}); sequential fallback.")
            pmf._init_worker(pmf.FRAME_PATH)
            for task in tasks:
                idx, success, error, row = pmf._process_sample(task)
                if success and row:
                    manifest_rows.append(row)
                    ok += 1

    pmf._write_manifest(output_root, manifest_rows, append=True)
    write_dataset_meta(
        output_root,
        stage="raw",
        source_script="pipelines/data/append_peb_batch_raw.py",
        extra={
            "anchor_mhz_appended": list(only_freq_mhz),
            "peb_batch_map": str(decap_index_map),
        },
    )

    summary["rows_written"] = ok
    summary["manifest_rows_added"] = len(manifest_rows)
    print(f"\nWrote {ok:,} heatmap rows ({len(manifest_rows):,} manifest entries)")
    if expected_per_freq and any(
        freq_counts.get(fl, 0) != expected_per_freq for fl in pmf.FREQ_LABELS
    ):
        print("  Warning: not every MHz has the same layout count — check missing PI folders above.")
    print("Next: python pipelines/normalize/multifreq.py  # APPEND=True, USE_GLOBAL_MAX_HEATMAP=True")
    return summary


def main() -> None:
    data_root = Path(DATA_ROOT) if DATA_ROOT else DEFAULT_DATA_ROOT
    if data_root is None or not data_root.exists():
        raise SystemExit(
            "Raw data root not found. Set DATA_ROOT or place Raw at Desktop/Raw.",
        )

    print("=" * 60)
    print("APPEND PEB-BATCH RAW (PI-1..PI-N → manifest design_id)")
    print("=" * 60)
    print(f"Mode: {'EXECUTE' if EXECUTE else 'DRY-RUN'}")

    append_peb_batch_raw(
        output_root=OUTPUT_ROOT,
        data_root=data_root,
        decap_index_map=DECAP_INDEX_MAP,
        decap_csv=DECAP_CSV,
        only_freq_mhz=ONLY_FREQS_MHZ,
        execute=EXECUTE,
        skip_existing_freq_rows=SKIP_EXISTING_FREQ_ROWS,
        num_workers=NUM_WORKERS,
        verbose=True,
    )


if __name__ == "__main__":
    main()
