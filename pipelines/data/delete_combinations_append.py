#!/usr/bin/env python3
"""Delete combinations-appended samples from a multifreq dataset.

Rows are matched by manifest ``append_tag`` and/or ``source_folder=combinations``.
Batch registries live under ``append_batches/`` and are listed in
``combinations_append_index.json``.

Run:
    python pipelines/data/delete_combinations_append.py          # dry-run
    python pipelines/data/delete_combinations_append.py --execute
    python pipelines/data/delete_combinations_append.py --list-tags
    python pipelines/data/delete_combinations_append.py --tag combinations_20260628 --execute
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT, setup_path

setup_path()

from src_vae.others.multifreq_layout_store import (  # noqa: E402
    invalidate_training_caches,
    manifest_path,
    prune_orphan_layouts,
)

# =============================================================================
# CONFIGURATION — edit when not using CLI flags
# =============================================================================

DATA_DIR = REPO_ROOT / "datasets/data_multifreq_train"
APPEND_TAG: str | None = None  # None = all combinations rows (any append_tag)
EXECUTE = False

# =============================================================================

MANIFEST_FIELDS = [
    "sample_name",
    "design_id",
    "freq_label",
    "freq_mhz",
    "freq_hz",
    "source_folder",
    "append_tag",
    "pi_number",
    "decap_index",
]


def _load_index(data_dir: Path) -> dict:
    path = data_dir / "combinations_append_index.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"batches": []}


def list_tags(data_dir: Path) -> None:
    index = _load_index(data_dir)
    batches = index.get("batches", [])
    if not batches:
        print(f"No append batches registered under {data_dir}")
        return
    print(f"Registered combinations append batches ({len(batches)}):")
    for b in batches:
        print(
            f"  {b.get('append_tag')}  rows={b.get('row_count')}  "
            f"layouts={b.get('unique_layouts')}  created={b.get('created_at')}"
        )


def _manifest_fieldnames(rows: list[dict]) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    ordered = [f for f in MANIFEST_FIELDS if f in keys]
    for k in sorted(keys):
        if k not in ordered:
            ordered.append(k)
    return ordered


def delete_combinations_append(
    data_dir: Path,
    *,
    append_tag: str | None,
    execute: bool,
) -> dict[str, int]:
    data_dir = data_dir.resolve()
    mf = manifest_path(data_dir)
    if not mf.is_file():
        raise FileNotFoundError(f"Missing manifest: {mf}")

    hm_dir = data_dir / "heatmap"
    pf_dir = data_dir / "PI_freq"
    layouts_dir = data_dir / "layouts"

    with mf.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    registry_samples: set[str] = set()
    registry_designs: set[str] = set()
    if append_tag:
        reg_path = data_dir / "append_batches" / f"{append_tag}.json"
        if reg_path.is_file():
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            registry_samples = set(reg.get("sample_names") or [])
            registry_designs = set(reg.get("design_ids") or [])

    def is_combinations_row(r: dict) -> bool:
        sf = (r.get("source_folder") or "").strip()
        did = (r.get("design_id") or "").strip()
        if sf == "combinations" or did.startswith("combinations_"):
            return True
        return False

    def should_drop(r: dict) -> bool:
        if append_tag:
            tag = (r.get("append_tag") or "").strip()
            if tag == append_tag:
                return True
            name = (r.get("sample_name") or "").strip()
            did = (r.get("design_id") or "").strip()
            if name in registry_samples or did in registry_designs:
                return True
            return False
        return is_combinations_row(r)

    drop = [r for r in rows if should_drop(r)]
    keep = [r for r in rows if r not in drop]

    extra_samples: set[str] = set()
    if append_tag is not None and registry_samples:
        extra_samples = registry_samples - {r.get("sample_name", "") for r in drop}

    counts = {
        "manifest_before": len(rows),
        "manifest_after": len(keep),
        "rows_removed": len(drop),
        "heatmap_removed": 0,
        "pi_freq_removed": 0,
        "layouts_removed": 0,
        "registry_removed": 0,
        "errors": 0,
    }

    if append_tag is None and layouts_dir.is_dir():
        orphan_layouts = sum(
            1
            for p in layouts_dir.iterdir()
            if p.is_dir() and p.name.startswith("combinations_")
        )
        counts["combinations_layout_dirs"] = orphan_layouts

    if not drop and not extra_samples and not (append_tag is None and counts.get("combinations_layout_dirs")):
        return counts

    if not execute:
        return counts

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag_suffix = append_tag or "all_combinations"
    backup = data_dir / f"manifest.csv.bak_delete_{tag_suffix}_{ts}"
    shutil.copy2(mf, backup)

    names_to_delete = {r.get("sample_name", "") for r in drop if r.get("sample_name")}
    names_to_delete |= extra_samples
    for name in sorted(names_to_delete):
        hm = hm_dir / name
        pf = pf_dir / name
        try:
            if hm.is_file():
                hm.unlink()
                counts["heatmap_removed"] += 1
            if pf.is_file():
                pf.unlink()
                counts["pi_freq_removed"] += 1
        except OSError:
            counts["errors"] += 1

    design_ids = sorted(
        {r.get("design_id", "") for r in drop if r.get("design_id")} | registry_designs
    )
    layout_targets = set(design_ids)
    if append_tag is None and layouts_dir.is_dir():
        layout_targets |= {
            p.name for p in layouts_dir.iterdir()
            if p.is_dir() and p.name.startswith("combinations_")
        }
    for did in sorted(layout_targets):
        layout_path = layouts_dir / did
        if layout_path.is_dir():
            try:
                shutil.rmtree(layout_path)
                counts["layouts_removed"] += 1
            except OSError:
                counts["errors"] += 1

    with mf.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_manifest_fieldnames(keep))
        writer.writeheader()
        writer.writerows(keep)

    prune_orphan_layouts(data_dir, allowed_design_ids={r["design_id"] for r in keep}, dry_run=False)

    index = _load_index(data_dir)
    batches = index.get("batches", [])
    if append_tag is None:
        for b in batches:
            reg = data_dir / b.get("registry", "")
            if reg.is_file():
                reg.unlink()
                counts["registry_removed"] += 1
        index["batches"] = []
    else:
        kept_batches = []
        for b in batches:
            if b.get("append_tag") == append_tag:
                reg = data_dir / b.get("registry", "")
                if reg.is_file():
                    reg.unlink()
                    counts["registry_removed"] += 1
            else:
                kept_batches.append(b)
        index["batches"] = kept_batches
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    (data_dir / "combinations_append_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )

    invalidate_training_caches(data_dir)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tag", type=str, default=APPEND_TAG, help="append_tag to delete")
    parser.add_argument(
        "--all-combinations",
        action="store_true",
        help="delete every combinations row (ignore --tag)",
    )
    parser.add_argument("--execute", action="store_true", default=EXECUTE)
    parser.add_argument("--list-tags", action="store_true", help="list registered append batches")
    args = parser.parse_args()

    if args.list_tags:
        list_tags(args.data_dir)
        return

    tag = None if args.all_combinations else args.tag
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    scope = tag or "all combinations rows"
    print(f"{mode}: delete combinations append [{scope}]")
    print(f"  dataset: {args.data_dir}")

    counts = delete_combinations_append(args.data_dir, append_tag=tag, execute=args.execute)
    print(
        f"  manifest: {counts['manifest_before']} -> {counts['manifest_after']} "
        f"(-{counts['rows_removed']})"
    )
    if counts.get("combinations_layout_dirs"):
        print(f"  combinations layout dirs on disk: {counts['combinations_layout_dirs']}")
    if args.execute:
        print(
            f"  removed heatmap={counts['heatmap_removed']}  "
            f"PI_freq={counts['pi_freq_removed']}  "
            f"layouts={counts['layouts_removed']}  "
            f"registry={counts['registry_removed']}  "
            f"errors={counts['errors']}"
        )
    elif counts["rows_removed"]:
        print("  Re-run with --execute to delete files.")
    else:
        print("  Nothing matched.")


if __name__ == "__main__":
    main()
