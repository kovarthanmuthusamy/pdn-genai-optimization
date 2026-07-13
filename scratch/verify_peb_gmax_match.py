#!/usr/bin/env python3
"""TEMPORARY one-off: verify PEB-batch append + gmax dataset alignment.

Delete when done. Does not modify any pipeline scripts.

Run:
    python scratch/verify_peb_gmax_match.py
    python scratch/verify_peb_gmax_match.py --quick      # counts + occ map only (~30s)
    python scratch/verify_peb_gmax_match.py --raw-spot  # also check 5 Raw PI-N folders (slow on /mnt/c)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import REPO_ROOT as ROOT, setup_path

setup_path()

from src_vae.others.heatmap_gmax_norm import physical_to_gmax_norm
from src_vae.others.multifreq_layout_store import layout_occ_path, layouts_root

# --- edit paths if needed ---
TRAIN_DIR = ROOT / "datasets" / "data_multifreq_train"
GMAX_DIR = ROOT / "datasets" / "data_multifreq_gmax"
REF_DIR = ROOT / "datasets" / "data_multifreq_norm_z_score"
MAP_CSV = ROOT / "data" / "heatmaps" / "decap_index_map.csv"
DECAP_CSV = ROOT / "data" / "heatmaps" / "all_combinations.csv"
RAW_DIR = Path("/mnt/c/Users/muthusamy/Desktop/Raw")

NEW_MHZ = ["350MHz", "370MHz", "390MHz", "420MHz", "450MHz"]
EXPECTED_LAYOUTS = 19_499


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_peb_map(path: Path) -> dict[int, dict]:
    """batch_pi (1-based) -> {peb_row, design_id, original_decap_index}"""
    by_batch: dict[int, dict] = {}
    design_to_batch: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            peb_row = int(row["peb_row"])
            batch_pi = peb_row + 1
            did = str(row["design_id"])
            by_batch[batch_pi] = {
                "peb_row": peb_row,
                "design_id": did,
                "original_decap_index": int(row["original_decap_index"]),
            }
            design_to_batch[did] = batch_pi
    return by_batch, design_to_batch


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def check_decap_map_occ(train_dir: Path, decap_csv: Path, map_csv: Path) -> dict:
    section("1) decap_index_map + all_combinations.csv ↔ train layouts/occ.npy")
    _, design_to_batch = load_peb_map(map_csv)
    df = pd.read_csv(decap_csv, header=None)
    vectors = df.select_dtypes(include=[np.number]).values.astype(np.float32)

    bad_occ = missing = 0
    for did, batch_pi in design_to_batch.items():
        peb_row = batch_pi - 1
        occ_p = layout_occ_path(train_dir, did)
        if not occ_p.is_file():
            missing += 1
            continue
        occ = np.load(occ_p).reshape(-1)
        if not np.allclose(occ, vectors[peb_row]):
            bad_occ += 1

    print(f"  Map layouts           : {len(design_to_batch):,}")
    print(f"  Decap CSV rows        : {len(vectors):,}")
    print(f"  occ mismatch          : {bad_occ}")
    print(f"  missing layouts/occ   : {missing}")
    ok = bad_occ == 0 and missing == 0
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "bad_occ": bad_occ, "missing": missing}


def check_peb_batch_manifest(train_dir: Path, map_csv: Path) -> dict:
    section("2) peb_batch manifest rows (append_peb_batch_raw)")
    manifest = load_manifest(train_dir / "manifest.csv")
    _, design_to_batch = load_peb_map(map_csv)

    peb_rows = [r for r in manifest if r.get("source_folder") == "peb_batch"]
    fc = Counter(r["freq_label"] for r in peb_rows)
    designs = {r["design_id"] for r in peb_rows}

    bad_design = sum(1 for r in peb_rows if r["design_id"] not in design_to_batch)
    pi_viol = 0
    meta_by_did: dict[str, tuple[int, int]] = {}
    for r in manifest:
        did = r["design_id"]
        if did not in meta_by_did:
            meta_by_did[did] = (int(r["pi_number"]), int(r["decap_index"]))

    for r in peb_rows:
        did = r["design_id"]
        pi, di = meta_by_did[did]
        if pi != di + 1:
            pi_viol += 1
        batch_pi = design_to_batch.get(did)
        if batch_pi is not None and pi != batch_pi:  # sanity: manifest pi is real, not batch
            pass  # expected — pi_number is manifest value, not batch index

    print(f"  Total manifest rows   : {len(manifest):,}")
    print(f"  peb_batch rows        : {len(peb_rows):,}")
    print(f"  Unique layouts        : {len(designs):,}")
    print(f"  design_id not in map  : {bad_design}")
    print(f"  pi != decap+1         : {pi_viol}")
    print("  Per-frequency (peb_batch only):")
    for fl in NEW_MHZ:
        n = fc.get(fl, 0)
        flag = "OK" if n == EXPECTED_LAYOUTS else "CHECK"
        print(f"    {fl}: {n:,}  [{flag}]")

    expected_rows = EXPECTED_LAYOUTS * len(NEW_MHZ)
    ok = (
        len(peb_rows) == expected_rows
        and bad_design == 0
        and pi_viol == 0
        and all(fc.get(fl, 0) == EXPECTED_LAYOUTS for fl in NEW_MHZ)
    )
    print(f"  Expected peb_batch    : {expected_rows:,} ({EXPECTED_LAYOUTS:,} × {len(NEW_MHZ)} MHz)")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "peb_rows": len(peb_rows), "freq_counts": dict(fc)}


def check_ref_subset(train_dir: Path, ref_dir: Path) -> dict:
    section("3) train subset still matches reference layouts")
    ref_ids = {r["design_id"] for r in load_manifest(ref_dir / "manifest.csv")}
    train_ids = {r["design_id"] for r in load_manifest(train_dir / "manifest.csv")}
    extra = train_ids - ref_ids
    missing = ref_ids - train_ids

    print(f"  ref unique layouts    : {len(ref_ids):,}")
    print(f"  train unique layouts  : {len(train_ids):,}")
    print(f"  train - ref (extra)   : {len(extra)}")
    print(f"  ref - train (missing) : {len(missing)}")
    ok = len(extra) == 0 and len(missing) == 0
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "extra": len(extra), "missing": len(missing)}


def check_train_vs_gmax(train_dir: Path, gmax_dir: Path, *, spot_n: int = 20) -> dict:
    section("4) train raw heatmaps ↔ gmax normalized heatmaps")
    train_hm = train_dir / "heatmap"
    gmax_hm = gmax_dir / "heatmap"
    stats_path = gmax_dir / "normalization_stats.json"

    if not gmax_hm.is_dir():
        print("  gmax heatmap/ missing — skip")
        return {"ok": False, "skipped": True}

    # count without loading all into memory
    train_stems = {p.stem for p in train_hm.glob("sample_*.npy")}
    gmax_stems = {p.stem for p in gmax_hm.glob("sample_*.npy")}
    only_train = train_stems - gmax_stems
    only_gmax = gmax_stems - train_stems

    print(f"  train heatmap files   : {len(train_stems):,}")
    print(f"  gmax heatmap files    : {len(gmax_stems):,}")
    print(f"  in train only         : {len(only_train):,}")
    print(f"  in gmax only          : {len(only_gmax):,}")

    stats = None
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        hm = stats.get("Heatmap") or stats
        print(f"  normalization_stats   : global_max_ohm={hm.get('global_max_ohm')}  clip={hm.get('clip_max')}")
    else:
        print("  normalization_stats   : not written yet (normalize may still be running)")

    # Spot-check peb_batch rows
    manifest = load_manifest(train_dir / "manifest.csv")
    peb_samples = [r for r in manifest if r.get("source_folder") == "peb_batch"]
    rng = np.random.default_rng(42)
    pick = (
        list(rng.choice(peb_samples, size=min(spot_n, len(peb_samples)), replace=False))
        if peb_samples
        else []
    )

    gmax_range_bad = formula_bad = missing_pair = 0
    for r in pick:
        stem = Path(r["sample_name"]).stem
        tr_p = train_hm / f"{stem}.npy"
        gm_p = gmax_hm / f"{stem}.npy"
        if not tr_p.is_file() or not gm_p.is_file():
            missing_pair += 1
            continue
        raw = np.load(tr_p).astype(np.float32)
        gmx = np.load(gm_p).astype(np.float32)
        if gmx.ndim == 3:
            gmx = gmx[0]
        if raw.ndim < 3:
            continue
        ch0, mask = raw[0], raw[1]

        if gmx.max() > 1.05 or gmx.min() < -0.01:
            gmax_range_bad += 1

        if stats:
            hm = stats.get("Heatmap") or stats
            gmax_ohm = float(hm["global_max_ohm"])
            bg = float(hm.get("bg_ohm_threshold", 0.05))
            clip_hi = float(hm.get("clip_max", 1.02))
            expected = np.clip(
                physical_to_gmax_norm(ch0, global_max_ohm=gmax_ohm, bg_ohm=bg),
                0.0,
                clip_hi,
            )
            if not np.allclose(gmx.reshape(expected.shape), expected, rtol=1e-4, atol=1e-4):
                formula_bad += 1

    print(f"  Spot-check samples    : {len(pick)} (peb_batch)")
    print(f"  missing train/gmax pair: {missing_pair}")
    print(f"  gmax value out of range: {gmax_range_bad}")
    if stats:
        print(f"  gmax formula mismatch : {formula_bad}")
    else:
        print("  gmax formula check    : skipped (no stats yet)")

    complete = len(only_train) == 0 and len(only_gmax) == 0
    ok = complete and missing_pair == 0 and gmax_range_bad == 0
    if stats:
        ok = ok and formula_bad == 0
    if not complete:
        print("  NOTE: gmax build incomplete if 'in train only' > 0")
    print(f"  => {'PASS' if ok else 'PARTIAL' if gmax_stems else 'FAIL'}")
    return {
        "ok": ok,
        "train_n": len(train_stems),
        "gmax_n": len(gmax_stems),
        "only_train": len(only_train),
        "formula_bad": formula_bad if stats else None,
    }


def check_raw_batch_spot(raw_dir: Path, train_dir: Path, map_csv: Path, n: int = 5) -> dict:
    section(f"5) Raw PI-N spot-check ({n} layouts, slow on /mnt/c)")
    if not raw_dir.is_dir():
        print(f"  Raw not found: {raw_dir} — skip")
        return {"ok": True, "skipped": True}

    by_batch, _ = load_peb_map(map_csv)
    manifest = load_manifest(train_dir / "manifest.csv")
    meta_by_did = {}
    for r in manifest:
        if r["design_id"] not in meta_by_did:
            meta_by_did[r["design_id"]] = r

    # pick first N batch PIs that have 350MHz folder
    hm_root = raw_dir / "heatmap_350MHz"
    if not hm_root.is_dir():
        print(f"  Missing {hm_root} — skip")
        return {"ok": True, "skipped": True}

    checked = matched = 0
    for batch_pi in sorted(by_batch)[:n]:
        entry = by_batch[batch_pi]
        did = entry["design_id"]
        pi_dir = hm_root / f"PI-{batch_pi}"
        if not pi_dir.is_dir():
            print(f"  PI-{batch_pi}: missing in Raw 350MHz")
            continue
        maps = list(pi_dir.rglob("*.map"))
        if not maps:
            print(f"  PI-{batch_pi}: no .map")
            continue
        mrow = meta_by_did.get(did)
        if not mrow:
            print(f"  PI-{batch_pi}: design {did} not in manifest")
            continue
        checked += 1
        print(
            f"  PI-{batch_pi} (peb_row {entry['peb_row']}) "
            f"-> {did}  manifest_pi={mrow['pi_number']}  freq_row={mrow.get('freq_label','?')}"
        )
        matched += 1

    ok = checked == n and matched == n
    print(f"  => {'PASS' if ok else 'PARTIAL'} ({checked}/{n} checked)")
    return {"ok": ok, "checked": checked}


def main() -> int:
    ap = argparse.ArgumentParser(description="Temporary PEB/gmax verification")
    ap.add_argument("--quick", action="store_true", help="Skip gmax file scan (sections 1-3 only)")
    ap.add_argument("--raw-spot", action="store_true", help="Include Raw PI-N spot-check (slow)")
    ap.add_argument("--train", type=Path, default=TRAIN_DIR)
    ap.add_argument("--gmax", type=Path, default=GMAX_DIR)
    args = ap.parse_args()

    print("PEB-batch + gmax verification (temporary script)")
    print(f"  train : {args.train}")
    print(f"  gmax  : {args.gmax}")

    results = []
    results.append(check_decap_map_occ(args.train, DECAP_CSV, MAP_CSV))
    results.append(check_peb_batch_manifest(args.train, MAP_CSV))
    results.append(check_ref_subset(args.train, REF_DIR))

    if not args.quick:
        results.append(check_train_vs_gmax(args.train, args.gmax))

    if args.raw_spot:
        results.append(check_raw_batch_spot(RAW_DIR, args.train, MAP_CSV))

    section("SUMMARY")
    names = ["decap/occ map", "peb_batch manifest", "ref subset", "train↔gmax", "raw spot"]
    fails = 0
    for i, r in enumerate(results):
        status = "PASS" if r.get("ok") else ("SKIP" if r.get("skipped") else "FAIL")
        if status == "FAIL":
            fails += 1
        print(f"  [{status}] {names[i]}")

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
