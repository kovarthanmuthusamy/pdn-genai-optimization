#!/usr/bin/env python3
"""Append or replace legacy 19k layout heatmaps from Dataset_19k into data_multifreq_train.

Maps ``all_combinations.csv`` row *i* → ``PI-(i+1)`` under ``heatmap_{MHz}MHz/``.
Uses ``decap_index_map.csv`` for ``design_id`` / ``decap_index``.

Run:
    python pipelines/heatmaps/append_legacy_19k_multifreq.py              # dry-run
    python pipelines/heatmaps/append_legacy_19k_multifreq.py --execute
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
from src_vae.others.multifreq_anchors import anchors_to_freq_hz, mhz_to_label
from src_vae.others.multifreq_layout_store import (
    invalidate_training_caches,
    load_manifest_rows,
    manifest_path,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

_PEB_19K = repo_path("data", "heatmaps", "peb_with_19k")
DECAP_CSV = _PEB_19K / "all_combinations.csv"
INDEX_MAP = _PEB_19K / "decap_index_map.csv"
RAW_ROOT_WIN = r"C:\Users\muthusamy\Desktop\Dataset_19k"
OUTPUT_ROOT = repo_path("datasets", "data_multifreq_train")

MHZ_LIST: list[float] = [350.0, 370.0, 390.0, 420.0, 450.0]
APPEND_TAG = "legacy_19k_restore"
SOURCE_FOLDER = "layout"
REPLACE_EXISTING = True
SKIP_IMPEDANCE_IF_MISSING = True
MANIFEST_FLUSH_EVERY = 5000
INDEX_THREADS = int(os.getenv("INDEX_THREADS", "16"))

_default_workers = "4" if str(resolve_windows_path(RAW_ROOT_WIN)).startswith("/mnt/") else "16"
NUM_WORKERS = int(os.getenv("NUM_WORKERS", _default_workers))

RAW_ROOT = resolve_windows_path(RAW_ROOT_WIN)
OUTPUT_ROOT = Path(OUTPUT_ROOT)

# =============================================================================


def load_19k_index_map(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing index map: {path}")
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            peb_row = int(row["peb_row"])
            rows.append(
                {
                    "peb_row": peb_row,
                    "pi_number": peb_row + 1,
                    "decap_index": int(row["original_decap_index"]),
                    "design_id": str(row["design_id"]),
                }
            )
    rows.sort(key=lambda r: r["peb_row"])
    for i, row in enumerate(rows):
        if row["peb_row"] != i:
            raise ValueError(f"{path}: peb_row not contiguous at {i} (got {row['peb_row']})")
    return rows


def _heatmap_raw_dir(raw_root: Path, mhz: float) -> Path:
    tag = int(round(float(mhz)))
    for name in (f"heatmap_{tag}MHz", f"heatmaps_{tag}MHz"):
        candidate = raw_root / name
        if candidate.is_dir():
            return candidate
    return raw_root / f"heatmap_{tag}MHz"


def _map_path_for(pi_dir: Path, mhz: float) -> Path:
    tag = int(round(mhz))
    return pi_dir / "Power_GND" / f"Z_{tag:04d}.000MHz.map"


def _index_hm_dir(args: tuple[float, Path]) -> tuple[float, dict[int, Path]]:
    mhz, hm_dir = args
    return mhz, pmf._list_pi_dirs(hm_dir)


def _load_hm_pi(raw_root: Path, mhz_list: list[float], *, threads: int) -> dict[float, dict[int, Path]]:
    hm_pi: dict[float, dict[int, Path]] = {}
    jobs = [(mhz, _heatmap_raw_dir(raw_root, mhz)) for mhz in mhz_list]
    for mhz, hm_dir in jobs:
        if not hm_dir.is_dir():
            raise FileNotFoundError(f"Missing heatmap folder for {mhz} MHz: {hm_dir}")
    print(f"  Indexing {len(mhz_list)} heatmap folder(s) ({threads} threads)...", flush=True)
    with ThreadPoolExecutor(max_workers=min(threads, len(mhz_list))) as ex:
        futs = {ex.submit(_index_hm_dir, job): job[0] for job in jobs}
        for fut in as_completed(futs):
            mhz, pi_map = fut.result()
            hm_pi[mhz] = pi_map
            hm_dir = _heatmap_raw_dir(raw_root, mhz)
            print(f"    {hm_dir.name}/: {len(pi_map):,} PI folders", flush=True)
    return hm_pi


def _existing_manifest_index(output_root: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for row in load_manifest_rows(output_root):
        did = row.get("design_id")
        fl = row.get("freq_label")
        if did and fl:
            out[(str(did), str(fl))] = dict(row)
    return out


def build_19k_tasks(
    index_entries: list[dict],
    mhz_list: list[float],
    hm_pi: dict[float, dict[int, Path]],
    freq_hz: dict[str, float],
    existing: dict[tuple[str, str], dict],
    *,
    append_tag: str,
    replace_existing: bool,
) -> tuple[list[dict], int, int]:
    samples: list[dict] = []
    n_replace = 0
    n_new = 0

    for i, entry in enumerate(index_entries):
        if i % 2000 == 0:
            print(f"    tasks {i:,}/{len(index_entries):,}...", flush=True)

        pi_number = int(entry["pi_number"])
        design_id = str(entry["design_id"])
        decap_index = int(entry["decap_index"])

        for mhz in mhz_list:
            pi_dir = hm_pi[mhz].get(pi_number)
            if pi_dir is None:
                continue
            fl = mhz_to_label(mhz)
            key = (design_id, fl)
            prev = existing.get(key)
            if prev is not None:
                if not replace_existing:
                    continue
                n_replace += 1
                sample_name = str(prev["sample_name"])
            else:
                n_new += 1
                sample_name = None

            samples.append(
                {
                    "design_id": design_id,
                    "source_folder": SOURCE_FOLDER,
                    "append_tag": append_tag,
                    "pi_number": pi_number,
                    "decap_index": decap_index,
                    "freq_label": fl,
                    "freq_mhz": float(mhz),
                    "freq_hz": freq_hz[fl],
                    "heatmap_path": _map_path_for(pi_dir, mhz),
                    "sample_name": sample_name,
                    "replace": prev is not None,
                }
            )
    return samples, n_replace, n_new


def _process_sample(task: tuple) -> tuple[int, bool, str | None, dict | None]:
    idx, sample, hm_dir, pifreq_dir = task
    try:
        from libs.data_creation.heatmap import create_Heatmaps

        stacked = create_Heatmaps(
            str(sample["heatmap_path"]), mask_board=pmf._WORKER_MASK_BOARD, verbose=False
        )
        name = sample.get("sample_name") or f"sample_{idx + 1}.npy"
        out_hm = hm_dir / name
        out_pf = pifreq_dir / name
        if out_hm.is_symlink():
            out_hm.unlink()
        if out_pf.is_symlink():
            out_pf.unlink()
        np.save(out_hm, stacked)
        hz = sample.get("freq_hz") or pmf.FREQ_HZ[sample["freq_label"]]
        np.save(out_pf, np.array(hz, dtype=np.float64))

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
            "replaced": bool(sample.get("replace")),
        }
    except Exception as e:
        return idx, False, str(e)[:200], None


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


def _update_manifest_rows(output_root: Path, updated: dict[str, dict]) -> int:
    """Update existing manifest rows by sample_name (in-place rewrite)."""
    mp = manifest_path(output_root)
    if not mp.is_file():
        return 0
    rows = list(csv.DictReader(mp.open(newline="", encoding="utf-8")))
    fieldnames = rows[0].keys() if rows else [
        "sample_name", "design_id", "freq_label", "freq_mhz", "freq_hz",
        "source_folder", "append_tag", "pi_number", "decap_index",
    ]
    n = 0
    for row in rows:
        sn = row.get("sample_name")
        if sn in updated:
            row.update(updated[sn])
            n += 1
    with mp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return n


def append_legacy_19k_dataset(
    *,
    decap_csv: Path,
    index_map: Path,
    raw_root: Path,
    output_root: Path,
    mhz_list: list[float],
    num_workers: int = NUM_WORKERS,
    dry_run: bool = False,
    replace_existing: bool = REPLACE_EXISTING,
    append_tag: str = APPEND_TAG,
) -> int:
    if not decap_csv.is_file():
        raise SystemExit(f"Missing decap CSV: {decap_csv}")
    if not index_map.is_file():
        raise SystemExit(f"Missing index map: {index_map}")
    if not raw_root.is_dir():
        raise SystemExit(f"Missing raw root: {raw_root}")

    index_entries = load_19k_index_map(index_map)
    occupancy = load_combinations_csv(decap_csv)
    if occupancy.shape[0] != len(index_entries):
        raise SystemExit(
            f"Row mismatch: {decap_csv.name} has {occupancy.shape[0]:,} rows, "
            f"index map has {len(index_entries):,}"
        )

    pmf._refresh_freq_tables(only_mhz=mhz_list)
    freq_hz = anchors_to_freq_hz(mhz_list)

    print("Append legacy 19k heatmaps -> data_multifreq_train (real files, no symlinks)")
    print(f"  Decap CSV   : {decap_csv}")
    print(f"  Index map   : {index_map}")
    print(f"  Raw root    : {raw_root}")
    print(f"  Output      : {output_root}")
    print(f"  Layouts     : {len(index_entries):,}")
    print(f"  MHz         : {mhz_list}")
    print(f"  Replace     : {replace_existing}")
    print(f"  Append tag  : {append_tag}")
    print(f"  Workers     : {num_workers}")

    hm_pi = _load_hm_pi(raw_root, mhz_list, threads=INDEX_THREADS)
    for mhz in mhz_list:
        needed = {int(e["pi_number"]) for e in index_entries}
        missing = sorted(pi for pi in needed if pi not in hm_pi[mhz])
        if missing:
            raise SystemExit(
                f"{_heatmap_raw_dir(raw_root, mhz).name} missing {len(missing):,} PI "
                f"(first PI-{missing[0]})"
            )
    print("  Raw validation OK", flush=True)

    existing = _existing_manifest_index(output_root)
    print(f"  Existing manifest keys: {len(existing):,}", flush=True)

    print("  Building task list...", flush=True)
    all_samples, n_replace, n_new = build_19k_tasks(
        index_entries, mhz_list, hm_pi, freq_hz, existing,
        append_tag=append_tag, replace_existing=replace_existing,
    )
    print(f"  Tasks: {len(all_samples):,}  (replace={n_replace:,}, new={n_new:,})")

    if not all_samples:
        print("Nothing to do.")
        return 0

    if dry_run:
        freq_counts = Counter(s["freq_label"] for s in all_samples)
        print("\nDRY RUN:")
        for fl in sorted(freq_counts):
            print(f"  {fl}: {freq_counts[fl]:,}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    hm_dir = output_root / "heatmap"
    pifreq_dir = output_root / "PI_freq"
    for d in (hm_dir, pifreq_dir):
        d.mkdir(parents=True, exist_ok=True)

    start_idx = pmf._next_sample_index(hm_dir)
    print(f"  New sample index from sample_{start_idx + 1}.npy", flush=True)

    tasks: list[tuple] = []
    next_idx = start_idx
    for s in all_samples:
        if s.get("sample_name"):
            idx = int(str(s["sample_name"]).replace("sample_", "").replace(".npy", "")) - 1
        else:
            idx = next_idx
            next_idx += 1
            s["sample_name"] = f"sample_{idx + 1}.npy"
        tasks.append((idx, s, hm_dir, pifreq_dir))

    chunksize = max(1, len(tasks) // (num_workers * 8))
    ok = 0
    new_rows: list[dict] = []
    manifest_updates: dict[str, dict] = {}

    if num_workers <= 1:
        pmf._init_worker(pmf.FRAME_PATH)
        for task in tasks:
            idx, success, error, row = _process_sample(task)
            if success and row:
                ok += 1
                if row.pop("replaced", False):
                    manifest_updates[row["sample_name"]] = {
                        k: row[k]
                        for k in (
                            "source_folder", "append_tag", "pi_number", "decap_index"
                        )
                    }
                else:
                    new_rows.append(row)
            elif error:
                print(f"  x sample_{idx + 1}: {error}")
    else:
        from multiprocessing import Pool

        with Pool(
            processes=num_workers,
            initializer=pmf._init_worker,
            initargs=(str(pmf.FRAME_PATH),),
        ) as pool:
            for i, out in enumerate(pool.imap_unordered(_process_sample, tasks, chunksize=chunksize)):
                idx, success, error, row = out
                if success and row:
                    ok += 1
                    if row.pop("replaced", False):
                        manifest_updates[row["sample_name"]] = {
                            k: row[k]
                            for k in (
                                "source_folder", "append_tag", "pi_number", "decap_index"
                            )
                        }
                    else:
                        new_rows.append(row)
                elif error:
                    print(f"  x sample_{idx + 1}: {error}")
                if (i + 1) % max(1, len(tasks) // 20) == 0:
                    print(f"  Progress: {i + 1:,}/{len(tasks):,}", flush=True)

    if new_rows:
        for j in range(0, len(new_rows), MANIFEST_FLUSH_EVERY):
            _append_manifest_rows(output_root, new_rows[j : j + MANIFEST_FLUSH_EVERY])
    if manifest_updates:
        n = _update_manifest_rows(output_root, manifest_updates)
        print(f"  Updated {n:,} existing manifest row(s)", flush=True)

    reg_dir = output_root / "append_batches"
    reg_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", append_tag)
    reg_path = reg_dir / f"{safe}.json"
    reg_path.write_text(
        json.dumps(
            {
                "append_tag": append_tag,
                "decap_csv": str(decap_csv),
                "index_map": str(index_map),
                "raw_root": str(raw_root),
                "mhz_list": mhz_list,
                "row_count": ok,
                "replaced": n_replace,
                "new_rows": n_new,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    invalidate_training_caches(output_root)
    write_dataset_meta(
        output_root,
        stage="raw",
        source_script="pipelines/heatmaps/append_legacy_19k_multifreq.py",
        extra={
            "decap_csv": str(decap_csv),
            "index_map": str(index_map),
            "raw_root": str(raw_root),
            "mhz_appended": mhz_list,
            "rows_written": ok,
            "append_tag": append_tag,
        },
    )

    print(f"\nWrote {ok:,} heatmap + PI_freq file(s) (real .npy, no symlinks)")
    print(f"  manifest: {manifest_path(output_root)}")
    print(f"  registry: {reg_path}")
    print("  Next: python pipelines/normalize/build_train_norm_unbounded.py")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run append (default: dry-run)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--mhz",
        type=float,
        nargs="*",
        default=MHZ_LIST,
        help="MHz list (default: 350 370 390 420 450)",
    )
    args = parser.parse_args()

    append_legacy_19k_dataset(
        decap_csv=DECAP_CSV,
        index_map=INDEX_MAP,
        raw_root=RAW_ROOT,
        output_root=OUTPUT_ROOT,
        mhz_list=list(args.mhz),
        num_workers=args.workers,
        dry_run=not args.execute,
        replace_existing=REPLACE_EXISTING,
        append_tag=APPEND_TAG,
    )


if __name__ == "__main__":
    main()
