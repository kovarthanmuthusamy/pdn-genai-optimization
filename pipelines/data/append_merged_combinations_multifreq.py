#!/usr/bin/env python3
"""Append multifreq rows from a **merged** ECADStar Raw run (PI-1..PI-N) into data_multifreq_train.

Use after simulating ``all_combinations_merged.csv`` (legacy + combinations in one PEB batch).
Row / PI alignment comes from ``merged_combinations_index_map.csv`` produced by
``pipelines/heatmaps/merge_combination_csvs.py``.

Run:
    python pipelines/heatmaps/merge_combination_csvs.py   # once, if not already merged
    # simulate merged PEB → Raw/Impedance + Raw/heatmaps_{MHz}MHz/
    python pipelines/data/append_merged_combinations_multifreq.py
    python pipelines/data/append_merged_combinations_multifreq.py --mhz 170 --append-tag merged_170
    python pipelines/data/append_merged_combinations_multifreq.py --mhz 170 280 430 470 --dry-run
    python pipelines/data/append_merged_combinations_multifreq.py --mhz 470 --append-tag merged_470_fill

Speed tips:
    - Copy Raw to local disk and set RAW_ROOT_WIN
    - NUM_WORKERS=16 on local ext4; 4-8 on /mnt/c
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from pipelines.dataset_sim.move_outputs import heatmaps_dir_name
from pipelines.dataset_sim.paths import RAW_ROOT_WIN as DEFAULT_RAW_ROOT_WIN, heatmap_raw_dir
from src_vae.others.multifreq_anchors import (
    anchors_to_freq_hz,
    anchors_to_freq_labels,
    load_anchors_mhz,
    mhz_to_label,
)
from src_vae.others.multifreq_layout_store import (
    invalidate_training_caches,
    load_manifest_rows,
    manifest_path,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

_MERGED_DIR = repo_path("data", "heatmaps", "peb_with_29k")
MERGED_CSV = _MERGED_DIR / "all_combinations_merged.csv"
MERGED_INDEX_MAP = _MERGED_DIR / "merged_combinations_index_map.csv"
RAW_ROOT_WIN = DEFAULT_RAW_ROOT_WIN
OUTPUT_ROOT = repo_path("datasets", "data_multifreq_train")

START_ROW = 0
MAX_ROWS: int | None = None
# Set explicitly for partial Raw (e.g. only 100/120/150 MHz); None = auto-discover under Raw
MHZ_LIST: list[float] | None = None

APPEND_TAG: str | None = None
DRY_RUN = False
SKIP_EXISTING = True
# When True (default with SKIP_EXISTING): validate Raw and build tasks only for manifest gaps.
MISSING_ONLY = True
# Full merged PEB (29,499 rows): False → PI-1..PI-29499 in Raw.
# Combinations-only Raw (10k PI): True → combinations segment uses PI-(local_row+1).
COMBINATIONS_USE_LOCAL_PI = False
# MHz-only append when layouts already exist in OUTPUT_ROOT and Raw has no Impedance/
SKIP_IMPEDANCE_IF_MISSING = True
MANIFEST_FLUSH_EVERY = 5000
INDEX_THREADS = int(os.getenv("INDEX_THREADS", "16"))

_default_workers = "4" if str(resolve_windows_path(RAW_ROOT_WIN)).startswith("/mnt/") else "16"
NUM_WORKERS = int(os.getenv("NUM_WORKERS", _default_workers))

RAW_ROOT = resolve_windows_path(RAW_ROOT_WIN)
OUTPUT_ROOT = Path(OUTPUT_ROOT)

# =============================================================================


def _default_append_tag() -> str:
    return f"merged_{datetime.now(timezone.utc).strftime('%Y%m%d')}"


def _progress_path() -> Path:
    return OUTPUT_ROOT / "append_merged_progress.json"


def _load_progress() -> dict:
    path = _progress_path()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"rows_appended": 0}


def _save_progress(progress: dict) -> None:
    path = _progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _append_batches_dir(output_root: Path) -> Path:
    return output_root / "append_batches"


def _batch_registry_path(output_root: Path, append_tag: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", append_tag)
    return _append_batches_dir(output_root) / f"{safe}.json"


def _load_batch_index(output_root: Path) -> dict:
    path = output_root / "merged_append_index.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"batches": []}


def _save_batch_registry(
    output_root: Path,
    *,
    append_tag: str,
    merged_csv: Path,
    index_map: Path,
    raw_root: Path,
    mhz_list: list[float],
    manifest_rows: list[dict],
) -> None:
    reg_dir = _append_batches_dir(output_root)
    reg_dir.mkdir(parents=True, exist_ok=True)
    sample_names = sorted({r["sample_name"] for r in manifest_rows})
    design_ids = sorted({r["design_id"] for r in manifest_rows})
    segments = sorted({r.get("segment", "") for r in manifest_rows if r.get("segment")})
    payload = {
        "append_tag": append_tag,
        "merged_csv": str(merged_csv),
        "index_map": str(index_map),
        "raw_root": str(raw_root),
        "mhz_list": mhz_list,
        "row_count": len(manifest_rows),
        "unique_layouts": len(design_ids),
        "segments": segments,
        "sample_names": sample_names,
        "design_ids": design_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    reg_path = _batch_registry_path(output_root, append_tag)
    reg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    index = _load_batch_index(output_root)
    batches = [b for b in index.get("batches", []) if b.get("append_tag") != append_tag]
    batches.append(
        {
            "append_tag": append_tag,
            "registry": str(reg_path.relative_to(output_root)),
            "row_count": len(manifest_rows),
            "unique_layouts": len(design_ids),
            "created_at": payload["created_at"],
        }
    )
    index["batches"] = sorted(batches, key=lambda b: b.get("created_at", ""))
    index["updated_at"] = payload["created_at"]
    (output_root / "merged_append_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    print(f"  Batch registry: {reg_path}")


def load_merged_index_map(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing merged index map: {path}")
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "global_peb_row": int(row["global_peb_row"]),
                    "pi_number": int(row["pi_number"]),
                    "segment": str(row["segment"]),
                    "local_row": int(row["local_row"]),
                    "decap_index": int(row["decap_index"]),
                    "design_id": str(row["design_id"]),
                }
            )
    rows.sort(key=lambda r: r["global_peb_row"])
    for i, row in enumerate(rows):
        if row["global_peb_row"] != i:
            raise ValueError(
                f"{path}: global_peb_row not contiguous at {i} (got {row['global_peb_row']})"
            )
    return rows


def load_layout_meta(output_root: Path) -> dict[str, dict[str, int]]:
    meta: dict[str, dict[str, int]] = {}
    for row in load_manifest_rows(output_root):
        did = row.get("design_id")
        if not did or did in meta or not str(did).startswith("layout_"):
            continue
        try:
            meta[str(did)] = {
                "pi_number": int(row["pi_number"]),
                "decap_index": int(row["decap_index"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return meta


def _source_folder_for_segment(segment: str) -> str:
    if segment == "legacy":
        return "peb_batch"
    if segment == "combinations":
        return "combinations"
    raise ValueError(f"Unknown segment: {segment!r}")


def _manifest_pi_decap(entry: dict, layout_meta: dict[str, dict[str, int]]) -> tuple[int, int]:
    design_id = entry["design_id"]
    if entry["segment"] == "combinations":
        local = int(entry["local_row"])
        return local + 1, local
    meta = layout_meta.get(design_id)
    if meta is not None:
        return int(meta["pi_number"]), int(meta["decap_index"])
    return int(entry["decap_index"]) + 1, int(entry["decap_index"])


def _heatmap_raw_dir(raw_root: Path, mhz: float) -> Path:
    """Resolve Raw heatmap folder (``heatmap_{MHz}MHz`` or ``heatmaps_{MHz}MHz``)."""
    return heatmap_raw_dir(raw_root, mhz)


def _discover_mhz_from_raw(raw_root: Path) -> list[float]:
    mhz: list[float] = []
    pat = re.compile(r"^heatmaps?_(\d+)MHz$", re.IGNORECASE)
    for child in raw_root.iterdir():
        if child.is_dir() and (m := pat.match(child.name)):
            mhz.append(float(m.group(1)))
    return sorted(set(mhz))


def _filter_mhz_with_raw(raw_root: Path, mhz_list: list[float]) -> list[float]:
    """Keep only MHz values that have a heatmap folder under Raw."""
    available: list[float] = []
    for mhz in mhz_list:
        if _heatmap_raw_dir(raw_root, mhz).is_dir():
            available.append(mhz)
        else:
            print(f"  WARNING: no Raw folder for {mhz:g} MHz — skipped", flush=True)
    return available


def _map_path_for(pi_dir: Path, mhz: float) -> Path:
    tag = int(round(mhz))
    return pi_dir / "Power_GND" / f"Z_{tag:04d}.000MHz.map"


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
        print("  Impedance/ not found — will skip impedance read if layouts exist in dataset", flush=True)

    hm_pi: dict[float, dict[int, Path]] = {}
    hm_jobs = [(mhz, _heatmap_raw_dir(raw_root, mhz)) for mhz in mhz_list]
    for mhz, hm_dir in hm_jobs:
        if not hm_dir.is_dir():
            raise SystemExit(f"Missing heatmap folder for {mhz} MHz (tried heatmap_* / heatmaps_*)")

    print(f"  Indexing {len(mhz_list)} heatmap folder(s) ({threads} threads)...", flush=True)
    with ThreadPoolExecutor(max_workers=min(threads, len(mhz_list))) as ex:
        futs = {ex.submit(_index_hm_dir, job): job[0] for job in hm_jobs}
        for fut in as_completed(futs):
            mhz, pi_map = fut.result()
            hm_pi[mhz] = pi_map
            hm_dir = _heatmap_raw_dir(raw_root, mhz)
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
    return (sub / "imp.npy").is_file() and (sub / "occ.npy").is_file()


def build_merged_tasks(
    index_entries: list[dict],
    occupancy: np.ndarray,
    mhz_list: list[float],
    imp_paths: dict[int, Path],
    hm_pi: dict[float, dict[int, Path]],
    freq_hz: dict[str, float],
    layout_meta: dict[str, dict[str, int]],
    output_root: Path,
    *,
    append_tag: str,
    skip_keys: set[tuple[str, str]] | None = None,
    combinations_use_local_pi: bool = False,
    skip_impedance_if_missing: bool = False,
) -> list[dict]:
    skip_keys = skip_keys or set()
    samples: list[dict] = []

    for i, entry in enumerate(index_entries):
        if i % 2000 == 0:
            print(f"    tasks {i:,}/{len(index_entries):,}...", flush=True)

        global_row = int(entry["global_peb_row"])
        batch_pi = int(entry["pi_number"])
        if combinations_use_local_pi and entry["segment"] == "combinations":
            batch_pi = int(entry["local_row"]) + 1
        design_id = str(entry["design_id"])
        segment = str(entry["segment"])
        manifest_pi, decap_index = _manifest_pi_decap(entry, layout_meta)
        imp_path = imp_paths.get(batch_pi)
        if imp_path is None and not skip_impedance_if_missing:
            continue

        decap_vector = occupancy[global_row]
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
                    "source_folder": _source_folder_for_segment(segment),
                    "segment": segment,
                    "append_tag": append_tag,
                    "pi_number": manifest_pi,
                    "decap_index": decap_index,
                    "batch_pi": batch_pi,
                    "global_peb_row": global_row,
                    "freq_label": fl,
                    "freq_mhz": float(mhz),
                    "freq_hz": freq_hz[fl],
                    "heatmap_path": _map_path_for(pi_dir, mhz),
                    "impedance_path": imp_path,
                    "decap_vector": decap_vector,
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
            "segment": sample.get("segment"),
        }
    except Exception as e:
        return idx, False, str(e)[:200], None


def _batch_pi_for_entry(entry: dict, *, combinations_use_local_pi: bool) -> int:
    pi = int(entry["pi_number"])
    if combinations_use_local_pi and entry["segment"] == "combinations":
        return int(entry["local_row"]) + 1
    return pi


def existing_manifest_freq_keys(output_root: Path) -> set[tuple[str, str]]:
    """(design_id, freq_label) pairs already present in manifest.csv."""
    return _existing_freq_keys(output_root)


def missing_manifest_freq_keys(
    index_entries: list[dict],
    mhz_list: list[float],
    output_root: Path,
) -> set[tuple[str, str]]:
    """(design_id, freq_label) pairs in the merged index that are not in the manifest."""
    have = existing_manifest_freq_keys(output_root)
    missing: set[tuple[str, str]] = set()
    for entry in index_entries:
        design_id = str(entry["design_id"])
        for mhz in mhz_list:
            fl = mhz_to_label(mhz)
            key = (design_id, fl)
            if key not in have:
                missing.add(key)
    return missing


def summarize_missing_append(
    index_entries: list[dict],
    mhz_list: list[float],
    output_root: Path,
) -> dict[str, int]:
    """Count missing manifest rows per freq_label (for dry-run / logging)."""
    missing = missing_manifest_freq_keys(index_entries, mhz_list, output_root)
    counts: Counter = Counter()
    for _did, fl in missing:
        counts[fl] += 1
    return dict(counts)


def needed_pi_for_missing_append(
    index_entries: list[dict],
    mhz_list: list[float],
    skip_keys: set[tuple[str, str]],
    *,
    combinations_use_local_pi: bool,
) -> set[int]:
    """PI folders required in Raw to append only manifest gaps."""
    needed_pi: set[int] = set()
    for entry in index_entries:
        design_id = str(entry["design_id"])
        batch_pi = _batch_pi_for_entry(entry, combinations_use_local_pi=combinations_use_local_pi)
        for mhz in mhz_list:
            fl = mhz_to_label(mhz)
            if (design_id, fl) in skip_keys:
                continue
            needed_pi.add(batch_pi)
    return needed_pi


def _validate_raw_slice(
    imp_pi: dict[int, Path],
    hm_pi: dict[float, dict[int, Path]],
    index_entries: list[dict],
    mhz_list: list[float],
    raw_root: Path,
    *,
    combinations_use_local_pi: bool = False,
    skip_impedance_if_missing: bool = False,
    skip_keys: set[tuple[str, str]] | None = None,
    missing_only: bool = False,
) -> None:
    if missing_only and skip_keys is not None:
        needed_pi = needed_pi_for_missing_append(
            index_entries,
            mhz_list,
            skip_keys,
            combinations_use_local_pi=combinations_use_local_pi,
        )
    else:
        needed_pi = {
            _batch_pi_for_entry(e, combinations_use_local_pi=combinations_use_local_pi)
            for e in index_entries
        }
    if not needed_pi:
        print("  Raw validation: nothing missing to append", flush=True)
        return
    if not skip_impedance_if_missing or imp_pi:
        missing_imp = sorted(pi for pi in needed_pi if pi not in imp_pi)
        if missing_imp:
            raise SystemExit(
                f"Impedance missing {len(missing_imp):,} PI folder(s) for slice "
                f"(first missing PI-{missing_imp[0]})"
            )
    for mhz in mhz_list:
        missing_hm = sorted(pi for pi in needed_pi if pi not in hm_pi[mhz])
        if missing_hm:
            hm_name = _heatmap_raw_dir(raw_root, mhz).name
            raise SystemExit(
                f"{hm_name} missing {len(missing_hm):,} PI folder(s) "
                f"(first missing PI-{missing_hm[0]})"
            )


def append_merged_combinations_dataset(
    *,
    merged_csv: Path,
    index_map: Path,
    raw_root: Path,
    output_root: Path,
    mhz_list: list[float] | None = None,
    start_row: int = 0,
    max_rows: int | None = None,
    num_workers: int = NUM_WORKERS,
    dry_run: bool = False,
    skip_existing: bool = SKIP_EXISTING,
    missing_only: bool = MISSING_ONLY,
    append_tag: str | None = None,
    combinations_use_local_pi: bool = COMBINATIONS_USE_LOCAL_PI,
    skip_impedance_if_missing: bool = SKIP_IMPEDANCE_IF_MISSING,
) -> int:
    if not merged_csv.is_file():
        raise SystemExit(f"Missing merged CSV: {merged_csv}")
    if not index_map.is_file():
        raise SystemExit(f"Missing index map: {index_map}")
    if not raw_root.is_dir():
        raise SystemExit(f"Missing raw simulation root: {raw_root}")

    all_index = load_merged_index_map(index_map)
    end = len(all_index) if max_rows is None else min(len(all_index), start_row + max_rows)
    if start_row >= end:
        raise SystemExit(f"START_ROW={start_row} >= end={end}")
    index_entries = all_index[start_row:end]

    occupancy = load_combinations_csv(merged_csv)
    if occupancy.shape[0] != len(all_index):
        raise SystemExit(
            f"Row count mismatch: {merged_csv.name} has {occupancy.shape[0]:,} rows, "
            f"index map has {len(all_index):,}"
        )

    mhz_list = mhz_list or _discover_mhz_from_raw(raw_root)
    if not mhz_list:
        raise SystemExit(f"No heatmaps_*MHz folders under {raw_root}")
    mhz_list = _filter_mhz_with_raw(raw_root, mhz_list)
    if not mhz_list:
        raise SystemExit(f"None of the requested MHz folders exist under {raw_root}")

    pmf._refresh_freq_tables(only_mhz=mhz_list)
    freq_hz = anchors_to_freq_hz(mhz_list)
    tag = append_tag or _default_append_tag()
    seg_counts = Counter(e["segment"] for e in index_entries)

    print("Append merged combinations -> multifreq dataset")
    print(f"  Merged CSV  : {merged_csv}")
    print(f"  Index map   : {index_map}")
    print(f"  Raw root    : {raw_root}")
    print(f"  Output      : {output_root}")
    print(f"  Rows        : {start_row}..{end - 1} ({len(index_entries):,} layouts)")
    print(f"    legacy={seg_counts.get('legacy', 0):,}  combinations={seg_counts.get('combinations', 0):,}")
    print(f"  PI range    : PI-{index_entries[0]['pi_number']}..PI-{index_entries[-1]['pi_number']}")
    print(f"  MHz         : {mhz_list} ({len(mhz_list)} anchors)")
    print(f"  Expected    : {len(index_entries) * len(mhz_list):,} manifest rows")
    print(f"  Append tag  : {tag}")
    print(f"  Workers     : {num_workers}")
    if combinations_use_local_pi:
        print("  Combinations Raw PI: local_row+1 (combinations-only sim)")

    layout_meta = load_layout_meta(output_root)
    print(f"  Legacy meta : {len(layout_meta):,} layout design_ids in manifest")

    if skip_impedance_if_missing:
        print("  Impedance: skip if missing (MHz-only append on existing layouts)", flush=True)

    missing_only = bool(missing_only and skip_existing)
    skip_keys = existing_manifest_freq_keys(output_root) if skip_existing else set()
    if skip_existing:
        print(f"  Skip existing: {len(skip_keys):,} (design_id, freq) pairs", flush=True)
    if missing_only:
        miss_counts = summarize_missing_append(index_entries, mhz_list, output_root)
        total_missing = sum(miss_counts.values())
        print(f"  Missing-only: {total_missing:,} manifest row(s) to append", flush=True)
        for fl in anchors_to_freq_labels(mhz_list):
            n = miss_counts.get(fl, 0)
            if n:
                print(f"    {fl}: {n:,} missing")

    imp_pi, hm_pi = _load_raw_pi_index(raw_root, mhz_list)
    _validate_raw_slice(
        imp_pi, hm_pi, index_entries, mhz_list, raw_root,
        combinations_use_local_pi=combinations_use_local_pi,
        skip_impedance_if_missing=skip_impedance_if_missing,
        skip_keys=skip_keys,
        missing_only=missing_only,
    )
    print("  Raw validation OK", flush=True)

    imp_paths = _index_imp_paths(imp_pi, threads=INDEX_THREADS) if imp_pi else {}

    print("  Building task list...", flush=True)
    all_samples = build_merged_tasks(
        index_entries, occupancy, mhz_list, imp_paths, hm_pi, freq_hz,
        layout_meta, output_root, append_tag=tag, skip_keys=skip_keys,
        combinations_use_local_pi=combinations_use_local_pi,
        skip_impedance_if_missing=skip_impedance_if_missing,
    )
    print(f"  Tasks to run: {len(all_samples):,}")

    if not all_samples:
        print("Nothing to append.")
        return 0

    if dry_run:
        freq_counts = Counter(s["freq_label"] for s in all_samples)
        print("\nDRY RUN:")
        for fl in anchors_to_freq_labels(mhz_list):
            print(f"  {fl}: {freq_counts.get(fl, 0):,} task(s)")
        print(f"  segments: {dict(Counter(s['segment'] for s in all_samples))}")
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
    all_written_rows: list[dict] = []

    if num_workers <= 1:
        pmf._init_worker(pmf.FRAME_PATH)
        for i, task in enumerate(tasks):
            idx, success, error, row = _process_sample_fast(task)
            if success and row:
                manifest_buf.append(row)
                all_written_rows.append(row)
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
                    all_written_rows.append(row)
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

    if all_written_rows:
        _save_batch_registry(
            output_root, append_tag=tag, merged_csv=merged_csv, index_map=index_map,
            raw_root=raw_root, mhz_list=mhz_list, manifest_rows=all_written_rows,
        )

    invalidate_training_caches(output_root)
    write_dataset_meta(
        output_root, stage="raw",
        source_script="pipelines/data/append_merged_combinations_multifreq.py",
        extra={
            "merged_csv": str(merged_csv), "index_map": str(index_map),
            "raw_root": str(raw_root), "mhz_appended": mhz_list,
            "rows_appended": ok, "append_tag": tag,
            "start_row": start_row, "end_row": end,
        },
    )
    progress = _load_progress()
    progress["rows_appended"] = int(progress.get("rows_appended", 0)) + ok
    _save_progress(progress)

    print(f"\nAppended {ok:,} sample row(s)")
    print(f"  manifest: {manifest_path(output_root)}")
    print("  Next: python pipelines/normalize/multifreq.py  (set APPEND=True)")
    return ok


def _resolve_raw_root(cli_raw_root: str | None) -> Path:
    return resolve_windows_path(cli_raw_root or RAW_ROOT_WIN)


def _resolve_mhz_list(cli_mhz: list[float] | None) -> list[float]:
    mhz = cli_mhz if cli_mhz is not None else MHZ_LIST
    if mhz is None and RAW_ROOT.is_dir():
        return _discover_mhz_from_raw(RAW_ROOT)
    if mhz is None:
        return [float(m) for m in load_anchors_mhz()]
    return [float(m) for m in mhz]


def _resolve_append_tag(cli_tag: str | None, cli_mhz: list[float] | None) -> str | None:
    if cli_tag is not None:
        return cli_tag
    if cli_mhz is not None and len(cli_mhz) == 1:
        return f"merged_{int(round(cli_mhz[0]))}"
    return APPEND_TAG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mhz",
        type=float,
        nargs="+",
        default=None,
        help="MHz values to append (default: MHZ_LIST config or auto-discover under Raw)",
    )
    parser.add_argument(
        "--append-tag",
        default=None,
        help="Manifest batch tag (default: merged_<mhz> when one --mhz, else APPEND_TAG config)",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help=f"ECADStar Raw root (default: {RAW_ROOT_WIN})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview missing tasks only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--missing-only",
        action="store_true",
        default=None,
        help="Validate/append only manifest gaps (default with skip-existing)",
    )
    mode.add_argument(
        "--all-rows",
        action="store_true",
        help="Validate all index-map PI rows in Raw (even when skipping existing manifest rows)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip (design_id, freq) pairs already in manifest (default: True)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mhz = _resolve_mhz_list(args.mhz)
    append_tag = _resolve_append_tag(args.append_tag, args.mhz)
    raw_root = _resolve_raw_root(args.raw_root)
    skip_existing = SKIP_EXISTING if args.skip_existing is None else args.skip_existing
    if args.all_rows:
        missing_only = False
    elif args.missing_only:
        missing_only = True
    else:
        missing_only = MISSING_ONLY
    dry_run = DRY_RUN or args.dry_run

    append_merged_combinations_dataset(
        merged_csv=MERGED_CSV,
        index_map=MERGED_INDEX_MAP,
        raw_root=raw_root,
        output_root=OUTPUT_ROOT,
        mhz_list=mhz,
        start_row=START_ROW,
        max_rows=MAX_ROWS,
        num_workers=NUM_WORKERS,
        dry_run=dry_run,
        skip_existing=skip_existing,
        missing_only=missing_only,
        append_tag=append_tag,
        combinations_use_local_pi=COMBINATIONS_USE_LOCAL_PI,
        skip_impedance_if_missing=SKIP_IMPEDANCE_IF_MISSING,
    )


if __name__ == "__main__":
    main()
