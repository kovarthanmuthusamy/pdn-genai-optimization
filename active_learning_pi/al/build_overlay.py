"""Build layout_store overlay dataset from AL iterations (Option B: heatmap-only labels).

Run:
    python -m active_learning_pi.al.build_overlay --config active_learning_pi/config/exp057.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from active_learning_pi.al.config import load_config
from active_learning_pi.al.paths import iteration_dir, run_dir
from active_learning_pi.al.robust_normalize import load_heatmap_stats, normalize_heatmap_raw
from active_learning_pi.al.normalize_labels import denormalize_impedance_from_model, normalize_impedance_raw
from src_vae.others.multifreq_anchors import mhz_to_label
from src_vae.others.multifreq_layout_store import invalidate_training_caches, manifest_path


def _load_imp_stats(stats_json: Path) -> dict[str, Any]:
    raw = json.loads(stats_json.read_text(encoding="utf-8"))
    return dict(raw["Impedance"])


def _next_sample_index(heatmap_dir: Path) -> int:
    max_idx = 0
    for p in heatmap_dir.glob("sample_*.npy"):
        m = re.match(r"sample_(\d+)\.npy$", p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_manifest_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _candidate_lookup(scored_path: Path) -> dict[int, dict[str, Any]]:
    if not scored_path.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in json.loads(scored_path.read_text(encoding="utf-8")):
        out[int(row["candidate_id"])] = row
    return out


def collect_iteration_samples(
    cfg: dict,
    groot: Path,
    iteration: int,
) -> list[dict[str, Any]]:
    it_dir = iteration_dir(cfg, iteration, groot)
    manifest_path_local = it_dir / "ingest_manifest.json"
    if not manifest_path_local.is_file():
        return []

    scored = _candidate_lookup(it_dir / "scored_candidates.json")
    imp_stats = _load_imp_stats(groot / cfg["normalization_stats_json"])
    rows: list[dict[str, Any]] = []

    for meta in json.loads(manifest_path_local.read_text(encoding="utf-8")):
        if not meta.get("ingest_ok"):
            continue
        sample_dir = Path(meta["sample_dir"])
        hm_raw_path = sample_dir / "heatmap.npy"
        if not hm_raw_path.is_file():
            continue
        cid = int(meta["candidate_id"])
        pred = scored.get(cid, meta)
        rows.append(
            {
                "iteration": iteration,
                "candidate_id": cid,
                "mhz": float(meta.get("mhz", pred.get("mhz", 200.0))),
                "k": int(meta.get("k", pred.get("k", 0))),
                "occupancy": list(meta.get("occupancy", pred.get("occupancy", []))),
                "heatmap_raw_path": str(hm_raw_path),
                "pred_impedance_norm": pred.get("pred_impedance_norm"),
            }
        )
    if rows and not imp_stats:
        raise RuntimeError("Missing Impedance stats in normalization_stats.json")
    return rows


def build_overlay_from_iterations(
    cfg: dict,
    groot: Path,
    *,
    iterations: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Merge completed AL iterations into a layout_store overlay dataset."""
    overlay_rel = cfg.get("overlay_data_dir") or cfg.get("finetune", {}).get("overlay_data_dir")
    if not overlay_rel:
        raise ValueError("overlay_data_dir is required in AL config")
    overlay_root = groot / overlay_rel
    stats_json = groot / cfg["normalization_stats_json"]
    hm_stats = load_heatmap_stats(stats_json)
    imp_stats = _load_imp_stats(stats_json)

    run = run_dir(cfg, groot)
    if iterations is None:
        iterations = sorted(
            int(p.name.split("_")[1])
            for p in run.glob("iter_*")
            if (p / "ingest_manifest.json").is_file()
        )

    collected: list[dict[str, Any]] = []
    for it in iterations:
        collected.extend(collect_iteration_samples(cfg, groot, it))

    hm_dir = overlay_root / "heatmap"
    pf_dir = overlay_root / "PI_freq"
    layouts_dir = overlay_root / "layouts"
    manifest_rows = _read_manifest_rows(manifest_path(overlay_root))
    existing_keys = {
        (r["design_id"], r.get("freq_label", ""))
        for r in manifest_rows
        if r.get("design_id")
    }
    heatmap_only_ids = {
        str(x)
        for x in json.loads((overlay_root / "al_heatmap_only_design_ids.json").read_text(encoding="utf-8"))
    } if (overlay_root / "al_heatmap_only_design_ids.json").is_file() else set()

    next_idx = _next_sample_index(hm_dir) if hm_dir.is_dir() else 1
    run_name = str(cfg.get("run_name", "al"))
    append_tag = f"al_{run_name}"
    added = 0
    skipped = 0
    errors: list[str] = []

    for item in collected:
        mhz = float(item["mhz"])
        freq_label = mhz_to_label(mhz)
        design_id = f"al_{run_name}_i{item['iteration']:04d}_c{item['candidate_id']:03d}"
        key = (design_id, freq_label)
        if key in existing_keys:
            skipped += 1
            continue

        occ = np.asarray(item["occupancy"], dtype=np.float32).reshape(-1)
        if occ.size != 52:
            errors.append(f"{design_id}: occupancy length {occ.size}")
            continue

        hm_raw = np.load(item["heatmap_raw_path"])
        try:
            hm_norm = normalize_heatmap_raw(hm_raw, hm_stats, mhz=mhz, groot=groot)
        except Exception as exc:
            errors.append(f"{design_id}: heatmap normalize failed: {exc}")
            continue

        pred_z = item.get("pred_impedance_norm")
        if not pred_z:
            errors.append(f"{design_id}: missing pred_impedance_norm for layout imp placeholder")
            continue
        imp_phys = denormalize_impedance_from_model(np.asarray(pred_z, dtype=np.float32), imp_stats)
        imp_norm = normalize_impedance_raw(imp_phys, imp_stats)[0]

        sample_name = f"sample_{next_idx}.npy"
        row = {
            "sample_name": sample_name,
            "design_id": design_id,
            "freq_label": freq_label,
            "freq_mhz": f"{mhz:g}",
            "freq_hz": str(int(round(mhz * 1e6))),
            "source_folder": "active_learning",
            "append_tag": append_tag,
            "pi_number": str(-(item["iteration"] * 1000 + item["candidate_id"])),
            "decap_index": str(next_idx),
        }

        if dry_run:
            added += 1
            existing_keys.add(key)
            heatmap_only_ids.add(design_id)
            next_idx += 1
            continue

        hm_dir.mkdir(parents=True, exist_ok=True)
        pf_dir.mkdir(parents=True, exist_ok=True)
        layout_sub = layouts_dir / design_id
        layout_sub.mkdir(parents=True, exist_ok=True)

        np.save(hm_dir / sample_name, hm_norm.astype(np.float32))
        np.save(pf_dir / sample_name, np.float64(mhz * 1e6))
        np.save(layout_sub / "occ.npy", occ.astype(np.float32))
        np.save(layout_sub / "imp.npy", imp_norm.astype(np.float32))

        manifest_rows.append(row)
        existing_keys.add(key)
        heatmap_only_ids.add(design_id)
        added += 1
        next_idx += 1

    report = {
        "overlay_root": str(overlay_root),
        "iterations": iterations,
        "candidates_seen": len(collected),
        "added": added,
        "skipped_existing": skipped,
        "errors": errors,
        "dry_run": dry_run,
    }

    if dry_run:
        return report

    if added > 0:
        _write_manifest_rows(manifest_path(overlay_root), manifest_rows)
        (overlay_root / "al_heatmap_only_design_ids.json").write_text(
            json.dumps(sorted(heatmap_only_ids), indent=2) + "\n",
            encoding="utf-8",
        )
        shutil_copy_stats(stats_json, overlay_root / "normalization_stats.json")
        invalidate_training_caches(overlay_root)

    (run / "overlay_build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def shutil_copy_stats(src: Path, dst: Path) -> None:
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build AL layout_store overlay for exp057 finetune")
    parser.add_argument("--config", default="active_learning_pi/config/exp057.json")
    parser.add_argument("--iteration", type=int, action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    groot = Path(cfg["repo_root"])
    report = build_overlay_from_iterations(
        cfg,
        groot,
        iterations=args.iteration,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))
    return 0 if not report.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
