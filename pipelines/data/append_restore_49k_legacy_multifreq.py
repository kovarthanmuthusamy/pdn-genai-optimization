#!/usr/bin/env python3
"""Append missing legacy layout heatmaps from the 49k ECADStar restore into data_multifreq_train.

The restore folder (``Dataset_49k``) holds PI-1..PI-49106 heatmaps from the full
``all_combinations.csv.bak_49106`` run. This script maps the **19,499** subset via
``peb_with_19k/decap_index_map.csv`` and ``all_combinations.csv``, then appends only
MHz rows that are missing for ``layout_pi*`` layouts in ``data_multifreq_train``.

PI folder in Raw = PI-(original_decap_index + 1)  (same row index in the 49,106 CSV).

Run:
    python pipelines/data/append_restore_49k_legacy_multifreq.py --dry-run
    python pipelines/data/append_restore_49k_legacy_multifreq.py --mhz 10 --delete-raw-after
    python pipelines/heatmaps/run_restore_49k_stepwise.py   # one MHz → delete raw → next

After append:
    python pipelines/normalize/build_train_norm_unbounded.py   # or APPEND=True on multifreq.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import repo_path, setup_path

setup_path()
_logs = _REPO_BOOT / "logs"
if (_logs / "libs").is_dir() and str(_logs) not in sys.path:
    sys.path.insert(0, str(_logs))

import numpy as np

from libs.dataset_meta import write_dataset_meta
from pipelines.data import processing_multifreq as pmf
from pipelines.dataset_sim.combinations import load_combinations_csv
from pipelines.dataset_sim.ecadstar import resolve_windows_path
from pipelines.dataset_sim.paths import heatmap_raw_dir
from src_vae.others.multifreq_anchors import anchors_to_freq_hz, mhz_to_label
from src_vae.others.multifreq_layout_store import (
    invalidate_training_caches,
    load_manifest_rows,
    manifest_path,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

_PEB19K = repo_path("data", "heatmaps", "peb_with_19k")
LEGACY_CSV = _PEB19K / "all_combinations.csv"
DECAP_INDEX_MAP = _PEB19K / "decap_index_map.csv"
RAW_ROOT_WIN = r"C:\Users\muthusamy\Desktop\Raw"
OUTPUT_ROOT = repo_path("datasets", "data_multifreq_train")

# MHz missing for legacy layout_pi* (auto-filtered against Raw + existing manifest)
MISSING_LEGACY_MHZ = [400.0, 500.0]

START_ROW = 0
MAX_ROWS: int | None = None
MHZ_LIST: list[float] | None = None  # None → discover under Raw, intersect missing legacy MHz
APPEND_TAG: str | None = "restore_49k"
DRY_RUN = False
SKIP_EXISTING = True
SKIP_IMPEDANCE_IF_MISSING = True
COPY_LAYOUTS_FROM: Path | None = None  # optional restored data_multifreq/layouts
MANIFEST_FLUSH_EVERY = 5000
INDEX_THREADS = int(os.getenv("INDEX_THREADS", "16"))

_default_workers = "4" if str(resolve_windows_path(RAW_ROOT_WIN)).startswith("/mnt/") else "16"
NUM_WORKERS = int(os.getenv("NUM_WORKERS", _default_workers))

RAW_ROOT = resolve_windows_path(RAW_ROOT_WIN)
OUTPUT_ROOT = Path(OUTPUT_ROOT)

# =============================================================================


def _default_append_tag() -> str:
    return f"restore_49k_{datetime.now(timezone.utc).strftime('%Y%m%d')}"


def load_legacy_19k_index_map(path: Path) -> list[dict]:
    """Build index entries for the 19,499 legacy subset."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing decap index map: {path}")
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for peb_row, row in enumerate(csv.DictReader(f)):
            original = int(row["original_decap_index"])
            design_id = str(row["design_id"])
            batch_pi = original + 1
            rows.append(
                {
                    "global_peb_row": peb_row,
                    "pi_number": batch_pi,
                    "segment": "legacy",
                    "local_row": peb_row,
                    "decap_index": original,
                    "design_id": design_id,
                }
            )
    return rows


def _discover_mhz_from_raw(raw_root: Path) -> list[float]:
    mhz: list[float] = []
    pat = re.compile(r"^heatmaps?_(\d+)MHz$", re.IGNORECASE)
    for child in raw_root.iterdir():
        if child.is_dir() and (m := pat.match(child.name)):
            mhz.append(float(m.group(1)))
    return sorted(set(mhz))


def _missing_legacy_mhz(output_root: Path) -> set[float]:
    have: set[float] = set()
    for row in load_manifest_rows(output_root):
        did = str(row.get("design_id") or "")
        if not did.startswith("layout_pi"):
            continue
        try:
            have.add(float(row["freq_mhz"]))
        except (KeyError, TypeError, ValueError):
            continue
    return {float(m) for m in MISSING_LEGACY_MHZ} - have


def _resolve_mhz_list(raw_root: Path, output_root: Path, cli_mhz: list[float] | None) -> list[float]:
    if cli_mhz is not None:
        candidates = [float(m) for m in cli_mhz]
    elif MHZ_LIST is not None:
        candidates = [float(m) for m in MHZ_LIST]
    else:
        discovered = _discover_mhz_from_raw(raw_root)
        missing = _missing_legacy_mhz(output_root)
        candidates = sorted(m for m in discovered if m in missing)
        if not candidates and discovered:
            candidates = sorted(m for m in discovered if m in MISSING_LEGACY_MHZ)
    return candidates


def _map_path_for(pi_dir: Path, mhz: float) -> Path:
    tag = int(round(mhz))
    return pi_dir / "Power_GND" / f"Z_{tag:04d}.000MHz.map"


def delete_raw_heatmap_folder(raw_root: Path, mhz: float, *, dry_run: bool = False) -> Path | None:
    """Remove ``heatmap_{MHz}MHz`` / ``heatmaps_{MHz}MHz`` under the restore Raw root."""
    hm_dir = heatmap_raw_dir(raw_root, mhz)
    if not hm_dir.is_dir():
        print(f"  Raw folder already gone: {hm_dir}")
        return None
    if dry_run:
        print(f"  DRY RUN: would delete {hm_dir}")
        return hm_dir
    print(f"  Deleting raw folder: {hm_dir}", flush=True)
    shutil.rmtree(hm_dir)
    return hm_dir


def _index_hm_dir(args: tuple[float, Path]) -> tuple[float, dict[int, Path]]:
    mhz, hm_dir = args
    return mhz, pmf._list_pi_dirs(hm_dir)


def _resolve_imp_one(item: tuple[int, Path]) -> tuple[int, Path | None]:
    pi_num, pi_dir = item
    return pi_num, pmf.resolve_impedance_ic1(pi_dir)


def _index_imp_paths(imp_pi: dict[int, Path], *, threads: int) -> dict[int, Path]:
    print(f"  Resolving impedance CSV paths ({len(imp_pi):,} PI, {threads} threads)...", flush=True)
    out: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for pi_num, path in ex.map(_resolve_imp_one, list(imp_pi.items()), chunksize=200):
            if path is not None:
                out[pi_num] = path
    print(f"    impedance paths: {len(out):,}", flush=True)
    return out


def _load_raw_pi_index(
    raw_root: Path,
    mhz_list: list[float],
    *,
    threads: int = INDEX_THREADS,
) -> tuple[dict[int, Path], dict[float, dict[int, Path]]]:
    imp_dir = raw_root / "Impedance"
    imp_pi: dict[int, Path] = {}
    if imp_dir.is_dir():
        print("  Indexing Impedance/ PI folders...", flush=True)
        imp_pi = pmf._list_pi_dirs(imp_dir)
        print(f"    found {len(imp_pi):,} PI folders", flush=True)
    else:
        print("  Impedance/ not found — heatmap-only append", flush=True)

    hm_pi: dict[float, dict[int, Path]] = {}
    hm_jobs = [(mhz, heatmap_raw_dir(raw_root, mhz)) for mhz in mhz_list]
    for mhz, hm_dir in hm_jobs:
        if not hm_dir.is_dir():
            raise SystemExit(f"Missing heatmap folder for {mhz} MHz (tried heatmap_* / heatmaps_*)")

    print(f"  Indexing {len(mhz_list)} heatmap folder(s) ({threads} threads)...", flush=True)
    with ThreadPoolExecutor(max_workers=min(threads, len(mhz_list))) as ex:
        futs = {ex.submit(_index_hm_dir, job): job[0] for job in hm_jobs}
        for fut in as_completed(futs):
            mhz, pi_map = fut.result()
            hm_pi[mhz] = pi_map
            hm_dir = heatmap_raw_dir(raw_root, mhz)
            print(f"    {hm_dir.name}/: {len(pi_map):,} PI folders", flush=True)
    return imp_pi, hm_pi


def _existing_freq_keys(output_root: Path) -> set[tuple[str, str]]:
    return {
        (str(r["design_id"]), str(r["freq_label"]))
        for r in load_manifest_rows(output_root)
        if r.get("design_id") and r.get("freq_label")
    }


def _layout_store_complete(output_root: Path, design_id: str) -> bool:
    sub = output_root / "layouts" / design_id
    imp, occ = sub / "imp.npy", sub / "occ.npy"
    return imp.is_file() and occ.is_file() and imp.exists() and occ.exists()


def build_legacy_tasks(
    index_entries: list[dict],
    mhz_list: list[float],
    imp_paths: dict[int, Path],
    hm_pi: dict[float, dict[int, Path]],
    freq_hz: dict[str, float],
    output_root: Path,
    *,
    append_tag: str,
    skip_keys: set[tuple[str, str]],
    skip_impedance_if_missing: bool,
) -> list[dict]:
    samples: list[dict] = []
    for i, entry in enumerate(index_entries):
        if i % 2000 == 0:
            print(f"    tasks {i:,}/{len(index_entries):,}...", flush=True)

        batch_pi = int(entry["pi_number"])
        design_id = str(entry["design_id"])
        manifest_pi = batch_pi
        decap_index = int(entry["decap_index"])
        imp_path = imp_paths.get(batch_pi)

        for mhz in mhz_list:
            pi_dir = hm_pi[mhz].get(batch_pi)
            if pi_dir is None:
                continue
            fl = mhz_to_label(mhz)
            if (design_id, fl) in skip_keys:
                continue
            samples.append(
                {
                    "design_id": design_id,
                    "source_folder": "layout",
                    "segment": "legacy",
                    "append_tag": append_tag,
                    "pi_number": manifest_pi,
                    "decap_index": decap_index,
                    "batch_pi": batch_pi,
                    "global_peb_row": int(entry["global_peb_row"]),
                    "freq_label": fl,
                    "freq_mhz": float(mhz),
                    "freq_hz": freq_hz[fl],
                    "heatmap_path": _map_path_for(pi_dir, mhz),
                    "impedance_path": imp_path,
                }
            )
    return samples


def _append_manifest_rows(output_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    mp = manifest_path(output_root)
    fieldnames = [
        "sample_name", "design_id", "freq_label", "freq_mhz", "freq_hz",
        "source_folder", "append_tag", "pi_number", "decap_index",
    ]
    write_header = not mp.is_file()
    with mp.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _process_sample_fast(task: tuple) -> tuple[int, bool, str | None, dict | None]:
    idx, sample, hm_dir, layouts_dir, pifreq_dir = task
    try:
        from libs.data_creation.heatmap import create_Heatmaps

        stacked = create_Heatmaps(str(sample["heatmap_path"]), mask_board=pmf._WORKER_MASK_BOARD, verbose=False)
        name = f"sample_{idx + 1}.npy"
        np.save(hm_dir / name, stacked)
        hz = sample.get("freq_hz") or pmf.FREQ_HZ[sample["freq_label"]]
        np.save(pifreq_dir / name, np.array(hz, dtype=np.float64))

        if sample.get("impedance_path") is not None:
            err = pmf._write_layout_once(layouts_dir / sample["design_id"], sample)
            if err:
                return idx, False, err, None

        return idx, True, None, {
            "sample_name": name,
            "design_id": sample["design_id"],
            "freq_label": sample["freq_label"],
            "freq_mhz": sample["freq_mhz"],
            "freq_hz": float(hz),
            "source_folder": sample["source_folder"],
            "append_tag": sample.get("append_tag"),
            "pi_number": sample["pi_number"],
            "decap_index": sample["decap_index"],
        }
    except Exception as e:
        return idx, False, str(e)[:200], None


def copy_layouts_from_backup(src_root: Path, output_root: Path, design_ids: set[str]) -> int:
    """Copy imp.npy/occ.npy for layout_pi* from restored data_multifreq/layouts."""
    src_layouts = src_root / "layouts"
    if not src_layouts.is_dir():
        src_layouts = src_root
    if not src_layouts.is_dir():
        raise FileNotFoundError(f"No layouts directory under {src_root}")

    dst_layouts = output_root / "layouts"
    dst_layouts.mkdir(parents=True, exist_ok=True)
    copied = 0
    for did in sorted(design_ids):
        if not did.startswith("layout_pi"):
            continue
        if _layout_store_complete(output_root, did):
            continue
        src = src_layouts / did
        if not src.is_dir():
            continue
        dst = dst_layouts / did
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("imp.npy", "occ.npy"):
            sp, dp = src / name, dst / name
            if sp.is_file() and not sp.is_symlink():
                shutil.copy2(sp, dp)
        if _layout_store_complete(output_root, did):
            copied += 1
    return copied


def append_restore_49k_legacy(
    *,
    legacy_csv: Path,
    index_map: Path,
    raw_root: Path,
    output_root: Path,
    mhz_list: list[float] | None = None,
    start_row: int = 0,
    max_rows: int | None = None,
    num_workers: int = NUM_WORKERS,
    dry_run: bool = False,
    skip_existing: bool = SKIP_EXISTING,
    append_tag: str | None = None,
    skip_impedance_if_missing: bool = SKIP_IMPEDANCE_IF_MISSING,
    copy_layouts_from: Path | None = None,
) -> int:
    if not legacy_csv.is_file():
        raise SystemExit(f"Missing legacy CSV: {legacy_csv}")
    if not index_map.is_file():
        raise SystemExit(f"Missing index map: {index_map}")
    if not raw_root.is_dir():
        raise SystemExit(f"Missing raw restore root: {raw_root}")

    all_index = load_legacy_19k_index_map(index_map)
    end = len(all_index) if max_rows is None else min(len(all_index), start_row + max_rows)
    if start_row >= end:
        raise SystemExit(f"START_ROW={start_row} >= end={end}")
    index_entries = all_index[start_row:end]

    occupancy = load_combinations_csv(legacy_csv)
    if occupancy.shape[0] != len(all_index):
        raise SystemExit(
            f"Row count mismatch: {legacy_csv.name} has {occupancy.shape[0]:,} rows, "
            f"index map has {len(all_index):,}"
        )

    mhz_list = _resolve_mhz_list(raw_root, output_root, mhz_list)
    if not mhz_list:
        raise SystemExit("No MHz to append (check Raw folders and MISSING_LEGACY_MHZ)")

    pmf._refresh_freq_tables(only_mhz=mhz_list)
    freq_hz = anchors_to_freq_hz(mhz_list)
    tag = append_tag or _default_append_tag()
    missing_target = _missing_legacy_mhz(output_root)

    print("Append 49k restore -> data_multifreq_train (legacy 19,499 layouts)")
    print(f"  Legacy CSV  : {legacy_csv}")
    print(f"  Index map   : {index_map}")
    print(f"  Raw root    : {raw_root}")
    print(f"  Output      : {output_root}")
    print(f"  Rows        : {start_row}..{end - 1} ({len(index_entries):,} layouts)")
    print(f"  PI mapping  : PI-(original_decap_index+1) in 49,106 Raw")
    print(f"  MHz         : {mhz_list} ({len(mhz_list)} anchors)")
    print(f"  Missing target MHz for layout_pi*: {sorted(missing_target)}")
    print(f"  Expected    : up to {len(index_entries) * len(mhz_list):,} manifest rows")
    print(f"  Append tag  : {tag}")
    print(f"  Workers     : {num_workers}")

    design_ids = {str(e["design_id"]) for e in index_entries}
    if copy_layouts_from is not None:
        n = copy_layouts_from_backup(copy_layouts_from, output_root, design_ids)
        print(f"  Copied layouts imp/occ: {n:,} design_ids from {copy_layouts_from}")

    imp_pi, hm_pi = _load_raw_pi_index(raw_root, mhz_list)
    needed_pi = {int(e["pi_number"]) for e in index_entries}
    for mhz in mhz_list:
        missing_hm = sorted(pi for pi in needed_pi if pi not in hm_pi[mhz])
        if missing_hm:
            hm_name = heatmap_raw_dir(raw_root, mhz).name
            print(f"  WARNING: {hm_name} missing {len(missing_hm):,} PI folders (first PI-{missing_hm[0]})")

    imp_paths = _index_imp_paths(imp_pi, threads=INDEX_THREADS) if imp_pi else {}
    skip_keys = _existing_freq_keys(output_root) if skip_existing else set()
    if skip_existing:
        print(f"  Skip existing: {len(skip_keys):,} (design_id, freq) pairs", flush=True)

    print("  Building task list...", flush=True)
    all_samples = build_legacy_tasks(
        index_entries, mhz_list, imp_paths, hm_pi, freq_hz, output_root,
        append_tag=tag, skip_keys=skip_keys,
        skip_impedance_if_missing=skip_impedance_if_missing,
    )
    print(f"  Tasks to run: {len(all_samples):,}")

    if not all_samples:
        print("Nothing to append.")
        return 0

    if dry_run:
        freq_counts = Counter(s["freq_label"] for s in all_samples)
        print("\nDRY RUN:")
        for mhz in mhz_list:
            fl = mhz_to_label(mhz)
            print(f"  {fl}: {freq_counts.get(fl, 0):,}")
        print(f"  unique layouts: {len({s['design_id'] for s in all_samples}):,}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    hm_dir = output_root / "heatmap"
    layouts_dir = output_root / "layouts"
    pifreq_dir = output_root / "PI_freq"
    for d in (hm_dir, layouts_dir, pifreq_dir):
        d.mkdir(parents=True, exist_ok=True)

    start_idx = pmf._next_sample_index(hm_dir)
    print(f"  Continue sample index from sample_{start_idx + 1}.npy", flush=True)

    tasks = [(start_idx + i, s, hm_dir, layouts_dir, pifreq_dir) for i, s in enumerate(all_samples)]
    chunksize = max(1, len(tasks) // (num_workers * 8))
    ok = 0
    manifest_buf: list[dict] = []

    if num_workers <= 1:
        pmf._init_worker(pmf.FRAME_PATH)
        for task in tasks:
            idx, success, error, row = _process_sample_fast(task)
            if success and row:
                manifest_buf.append(row)
                ok += 1
            elif error:
                print(f"  x sample_{idx + 1}: {error}")
            if len(manifest_buf) >= MANIFEST_FLUSH_EVERY:
                _append_manifest_rows(output_root, manifest_buf)
                manifest_buf.clear()
    else:
        from multiprocessing import Pool

        with Pool(processes=num_workers, initializer=pmf._init_worker, initargs=(str(pmf.FRAME_PATH),)) as pool:
            for i, out in enumerate(pool.imap_unordered(_process_sample_fast, tasks, chunksize=chunksize)):
                idx, success, error, row = out
                if success and row:
                    manifest_buf.append(row)
                    ok += 1
                elif error:
                    print(f"  x sample_{idx + 1}: {error}")
                if len(manifest_buf) >= MANIFEST_FLUSH_EVERY:
                    _append_manifest_rows(output_root, manifest_buf)
                    manifest_buf.clear()
                if (i + 1) % max(1, len(tasks) // 20) == 0:
                    print(f"  Progress: {i + 1:,}/{len(tasks):,}", flush=True)

    if manifest_buf:
        _append_manifest_rows(output_root, manifest_buf)

    invalidate_training_caches(output_root)
    write_dataset_meta(
        output_root, stage="raw",
        source_script="pipelines/data/append_restore_49k_legacy_multifreq.py",
        extra={
            "legacy_csv": str(legacy_csv), "index_map": str(index_map),
            "raw_root": str(raw_root), "mhz_appended": mhz_list,
            "rows_appended": ok, "append_tag": tag,
        },
    )
    print(f"\nAppended {ok:,} sample row(s)")
    print(f"  manifest: {manifest_path(output_root)}")
    print("  Next: python pipelines/normalize/build_train_norm_unbounded.py")
    return ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhz", type=float, nargs="+", default=None, help="MHz to append")
    parser.add_argument("--append-tag", default=None, help="Manifest batch tag")
    parser.add_argument("--raw-root", default=None, help=f"Restore Raw root (default: {RAW_ROOT_WIN})")
    parser.add_argument(
        "--copy-layouts-from",
        default=None,
        help="Copy layout imp/occ from restored data_multifreq (or .../layouts)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview tasks only")
    parser.add_argument("--start-row", type=int, default=START_ROW)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--delete-raw-after",
        action="store_true",
        help="Delete heatmap_{MHz}MHz folder under --raw-root after a successful run",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_root = resolve_windows_path(args.raw_root or RAW_ROOT_WIN)
    copy_from = Path(args.copy_layouts_from).resolve() if args.copy_layouts_from else COPY_LAYOUTS_FROM
    mhz_list = _resolve_mhz_list(raw_root, OUTPUT_ROOT, args.mhz)

    ok = append_restore_49k_legacy(
        legacy_csv=LEGACY_CSV,
        index_map=DECAP_INDEX_MAP,
        raw_root=raw_root,
        output_root=OUTPUT_ROOT,
        mhz_list=mhz_list,
        start_row=args.start_row,
        max_rows=args.max_rows,
        num_workers=args.workers,
        dry_run=args.dry_run or DRY_RUN,
        append_tag=args.append_tag or APPEND_TAG,
        copy_layouts_from=copy_from,
    )

    if args.delete_raw_after and not (args.dry_run or DRY_RUN) and mhz_list:
        for mhz in mhz_list:
            delete_raw_heatmap_folder(raw_root, mhz, dry_run=False)


if __name__ == "__main__":
    main()
