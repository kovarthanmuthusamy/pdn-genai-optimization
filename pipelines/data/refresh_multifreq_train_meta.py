#!/usr/bin/env python3
"""Lightweight JSON metadata refresh for data_multifreq_train.

Reads manifest.csv only — no full heatmap/layout disk scan.

Run:
    python pipelines/data/refresh_multifreq_train_meta.py
    python pipelines/data/refresh_multifreq_train_meta.py --norm
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT, setup_path

setup_path()

from src_vae.others.multifreq_layout_store import invalidate_training_caches

TRAIN_ROOT = REPO_ROOT / "datasets" / "data_multifreq_train"
NORM_ROOT = REPO_ROOT / "datasets" / "data_multifreq_train_norm_robust"
APPEND_TAG = "New_data_10000"


def _mhz_counts(rows: list[dict]) -> dict[str, int]:
    c = Counter()
    for r in rows:
        if r.get("freq_mhz"):
            mhz = float(r["freq_mhz"])
            key = str(int(mhz) if mhz == int(mhz) else mhz)
            c[key] += 1
    return dict(sorted(c.items(), key=lambda x: float(x[0])))


def _load_rows(root: Path) -> list[dict]:
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def refresh_train_meta(root: Path, *, now: str) -> dict:
    rows = _load_rows(root)
    comb = [r for r in rows if r.get("source_folder") == "combinations"]
    orig = [r for r in rows if r.get("source_folder") != "combinations"]
    comb_layouts = len({r["design_id"] for r in comb})
    orig_layouts = len({r["design_id"] for r in orig})
    mhz_list = sorted({float(r["freq_mhz"]) for r in rows if r.get("freq_mhz")})

    meta_path = root / "dataset_meta.json"
    prior = {}
    if meta_path.is_file():
        try:
            prior = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}

    dataset_meta = {
        "schema_version": 1,
        "stage": prior.get("stage", "raw"),
        "source_script": "pipelines/data/refresh_multifreq_train_meta.py",
        "generated_at_utc": now,
        "dataset_root": str(root),
        "storage": "layout_store",
        "counts": {
            "manifest_rows": len(rows),
            "unique_layouts": comb_layouts + orig_layouts,
            "heatmap_files": len(rows),
            "pi_freq_files": len(rows),
            "layout_directories": comb_layouts + orig_layouts,
            "original_rows": len(orig),
            "combinations_rows": len(comb),
            "original_layouts": orig_layouts,
            "combinations_layouts": comb_layouts,
        },
        "size": prior.get("size", {}),
        "pi_frequencies_mhz": mhz_list,
        "samples_per_mhz": _mhz_counts(rows),
        "append_tag": APPEND_TAG,
        "combinations_csv": str(REPO_ROOT / "data" / "heatmaps" / "combinations.csv"),
        "raw_root": "/mnt/c/Users/muthusamy/Desktop/Raw",
        "manifest_columns": list(rows[0].keys()) if rows else [],
        "notes": [
            "Lightweight metadata refresh from manifest.csv only",
            "Counts assume manifest rows match on-disk heatmap/PI_freq files",
        ],
    }
    meta_path.write_text(json.dumps(dataset_meta, indent=2) + "\n", encoding="utf-8")

    index = {
        "batches": [
            {
                "append_tag": APPEND_TAG,
                "registry": "append_batches/New_data_10000.json",
                "row_count": len(comb),
                "unique_layouts": comb_layouts,
                "sample_range": {
                    "first": comb[0]["sample_name"] if comb else None,
                    "last": comb[-1]["sample_name"] if comb else None,
                },
                "created_at": "2026-06-29T12:56:40.058568+00:00",
                "manifest_synced_at": now,
            }
        ],
        "updated_at": now,
        "dataset_manifest_rows": len(rows),
    }
    (root / "combinations_append_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    progress = {
        "status": "complete",
        "append_tag": APPEND_TAG,
        "rows_appended": len(comb),
        "layouts_appended": comb_layouts,
        "mhz_count": len({r["freq_label"] for r in comb}),
        "updated_at": now,
    }
    (root / "append_combinations_progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

    subset_meta = {
        "dataset": str(root),
        "updated_at": now,
        "description": "data_multifreq_train = original subset + combinations append (New_data_10000)",
        "total_manifest_rows": len(rows),
        "total_layouts": comb_layouts + orig_layouts,
        "original": {
            "rows": len(orig),
            "layouts": orig_layouts,
            "heatmap_source": str(REPO_ROOT / "datasets" / "data_multifreq"),
            "mhz_in_subset": _mhz_counts(orig),
        },
        "combinations": {
            "append_tag": APPEND_TAG,
            "rows": len(comb),
            "layouts": comb_layouts,
            "mhz_in_subset": _mhz_counts(comb),
            "registry": "append_batches/New_data_10000.json",
        },
        "next_steps": [
            "Train from datasets/data_multifreq_train_norm_robust",
        ],
    }
    (root / "subset_meta.json").write_text(json.dumps(subset_meta, indent=2) + "\n", encoding="utf-8")

    reg_path = root / "append_batches" / f"{APPEND_TAG}.json"
    if reg_path.is_file():
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        reg["manifest_synced_at"] = now
        reg["manifest_row_count"] = len(comb)
        reg["manifest_has_append_tag"] = True
        reg_path.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")

    invalidate_training_caches(root)
    return {
        "manifest_rows": len(rows),
        "combinations_rows": len(comb),
        "layouts": comb_layouts + orig_layouts,
    }


def refresh_norm_meta(root: Path, *, now: str) -> dict:
    rows = _load_rows(root)
    meta_path = root / "dataset_meta.json"
    prior = {}
    if meta_path.is_file():
        try:
            prior = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}

    meta = {
        **prior,
        "generated_at_utc": now,
        "source_script": "pipelines/data/refresh_multifreq_train_meta.py",
        "stage": prior.get("stage", "normalized"),
        "dataset_root": str(root),
        "counts": {
            "manifest_rows": len(rows),
            "unique_layouts": len({r["design_id"] for r in rows if r.get("design_id")}),
            "heatmap_files": len(rows),
            "pi_freq_files": len(rows),
        },
        "notes": [
            "Lightweight metadata touch — counts from manifest.csv only",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    invalidate_training_caches(root)
    return {"manifest_rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm", action="store_true", help="also refresh normalized dataset meta")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    print(f"Refreshing train metadata: {TRAIN_ROOT}")
    stats = refresh_train_meta(TRAIN_ROOT, now=now)
    print(f"  manifest_rows={stats['manifest_rows']:,} combinations={stats['combinations_rows']:,} layouts={stats['layouts']:,}")

    if args.norm and NORM_ROOT.is_dir():
        print(f"Refreshing norm metadata: {NORM_ROOT}")
        nstats = refresh_norm_meta(NORM_ROOT, now=now)
        print(f"  manifest_rows={nstats['manifest_rows']:,}")

    print("Done.")


if __name__ == "__main__":
    main()
