"""Package and normalize ingested ECADSTAR labels for VAE fine-tuning.

Run:
    python active_learning_pi/al/normalize_labels.py"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from active_learning_pi.al.robust_normalize import is_robust_per_mhz_stats, normalize_heatmap_raw as _normalize_hm

EXPECTED_IMP_LEN = 231


def _load_training_stats(stats_json: Path) -> tuple[dict, dict, float]:
    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    hm = dict(stats["Heatmap"])
    imp = dict(stats["Impedance"])
    bg = float(stats.get("background_value", hm.get("background_value", -2.0)))
    hm["background_value"] = bg
    return hm, imp, bg


def denormalize_impedance_from_model(z: np.ndarray, imp_stats: dict) -> np.ndarray:
    """Invert log z-score used in pipelines/normalize/multifreq.py → raw |Z| (231,)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.size != EXPECTED_IMP_LEN:
        raise ValueError(f"Expected {EXPECTED_IMP_LEN} impedance points, got {z.size}")
    log_mean = float(imp_stats["log_mean"])
    log_std = float(imp_stats["log_std"])
    log_imp = z * log_std + log_mean
    return np.exp(log_imp).astype(np.float32)


def normalize_heatmap_raw(
    raw: np.ndarray,
    hm_stats: dict,
    *,
    mhz: float | None = None,
    groot: Path | None = None,
) -> np.ndarray:
    """Normalize raw heatmap using training stats (robust per-MHz or legacy log-z)."""
    if is_robust_per_mhz_stats(hm_stats) and mhz is None:
        raise ValueError("mhz is required for robust per-MHz heatmap normalization")
    return _normalize_hm(raw, hm_stats, mhz=mhz, groot=groot)


def normalize_impedance_raw(raw: np.ndarray, imp_stats: dict) -> np.ndarray:
    """Same as pipelines/normalize/multifreq.py — raw (231,) → (1, 231) log z-score."""
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw.size != EXPECTED_IMP_LEN:
        raise ValueError(f"Unexpected impedance length: {raw.size}")
    log_mean = float(imp_stats["log_mean"])
    log_std = float(imp_stats["log_std"])
    log_data = np.log(np.maximum(raw, 1e-10))
    z = (log_data - log_mean) / (log_std if log_std != 0.0 else 1.0)
    return z[np.newaxis].astype(np.float32, copy=False)


def _resolve_impedance_csv(sample_dir: Path) -> Path | None:
    for p in sorted(sample_dir.rglob("*.csv")):
        if "ic1" in p.name.lower():
            return p
    return None


def package_raw_dataset(
    labels_dir: Path,
    raw_root: Path,
    manifest: list[dict[str, Any]],
    groot: Path,
    imp_stats: dict,
    *,
    heatmap_only_labels: bool = False,
    hm_stats: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Layout under raw_root/ (matches Data_Creation multifreq before Normalization.py):
      heatmap/sample_N.npy   (2, 64, 64)
      Imp/sample_N.npy       (231,) physical
      Occ_map/sample_N.npy   (52,)
      PI_freq/sample_N.npy   float64 Hz
    """
    if str(groot) not in sys.path:
        sys.path.insert(0, str(groot))
    from libs.data_creation.impedance import read_impedance_file  # noqa: E402

    hm_out = raw_root / "heatmap"
    imp_out = raw_root / "Imp"
    occ_out = raw_root / "Occ_map"
    pf_out = raw_root / "PI_freq"
    for d in (hm_out, imp_out, occ_out, pf_out):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, meta in enumerate(manifest):
        if not meta.get("ingest_ok"):
            rows.append({**meta, "raw_packaged": False})
            continue

        sample_dir = Path(meta["sample_dir"])
        sid = f"sample_{i:03d}"

        hm_path = sample_dir / "heatmap.npy"
        if not hm_path.is_file():
            rows.append({**meta, "raw_packaged": False, "error": "missing heatmap.npy"})
            continue

        hm_raw = np.load(hm_path)
        np.save(hm_out / f"{sid}.npy", hm_raw.astype(np.float32))

        occ = np.load(sample_dir / "occupancy.npy").astype(np.float32).reshape(-1)
        np.save(occ_out / f"{sid}.npy", occ)

        mhz = float(meta.get("mhz", np.load(sample_dir / "pi_freq_mhz.npy")))
        np.save(pf_out / f"{sid}.npy", np.float64(mhz * 1e6))

        imp_source = "skipped_heatmap_only" if heatmap_only_labels else "missing"
        if heatmap_only_labels:
            rows.append(
                {
                    **meta,
                    "raw_packaged": True,
                    "raw_sample_id": sid,
                    "imp_source": imp_source,
                    "heatmap_only_label": True,
                }
            )
            continue

        imp_source = "simulation_csv"
        imp_raw = None
        csv_p = _resolve_impedance_csv(sample_dir)
        if csv_p is not None:
            imp_raw = read_impedance_file(str(csv_p))

        if imp_raw is None and meta.get("pred_impedance_norm"):
            imp_source = "model_denorm"
            imp_raw = denormalize_impedance_from_model(
                np.array(meta["pred_impedance_norm"], dtype=np.float32),
                imp_stats,
            )

        if imp_raw is None:
            rows.append({**meta, "raw_packaged": False, "error": "no impedance (csv or model)"})
            continue

        np.save(imp_out / f"{sid}.npy", imp_raw.reshape(-1).astype(np.float32))
        rows.append({**meta, "raw_packaged": True, "raw_sample_id": sid, "imp_source": imp_source})

    return rows


def normalize_raw_tree(
    raw_root: Path,
    norm_root: Path,
    stats_json: Path,
    *,
    overwrite: bool = False,
    heatmap_only_labels: bool = False,
    groot: Path | None = None,
) -> dict[str, Any]:
    """Apply training stats (pipelines/normalize/multifreq.py rules) to raw_root → norm_root."""
    if not stats_json.is_file():
        raise FileNotFoundError(f"Training stats not found: {stats_json}")

    hm_stats, imp_stats, _bg = _load_training_stats(stats_json)

    if overwrite and norm_root.exists():
        shutil.rmtree(norm_root)

    hm_out = norm_root / "heatmap"
    imp_out = norm_root / "Imp"
    occ_out = norm_root / "Occ_map"
    pf_out = norm_root / "PI_freq"
    for d in (hm_out, imp_out, occ_out, pf_out):
        d.mkdir(parents=True, exist_ok=True)

    occ_dir = raw_root / "Occ_map"
    pf_dir = raw_root / "PI_freq"
    stems = sorted(p.stem for p in occ_dir.glob("sample_*.npy"))
    done = 0
    errors: list[str] = []

    for stem in stems:
        try:
            hm_raw = np.load(raw_root / "heatmap" / f"{stem}.npy")
            occ_raw = np.load(occ_dir / f"{stem}.npy")
            pf_raw = np.load(pf_dir / f"{stem}.npy")
            mhz = float(np.asarray(pf_raw).reshape(-1)[0] / 1e6)

            np.save(
                hm_out / f"{stem}.npy",
                normalize_heatmap_raw(hm_raw, hm_stats, mhz=mhz, groot=groot),
            )
            if not heatmap_only_labels:
                imp_raw = np.load(raw_root / "Imp" / f"{stem}.npy")
                np.save(imp_out / f"{stem}.npy", normalize_impedance_raw(imp_raw, imp_stats))
            np.save(occ_out / f"{stem}.npy", occ_raw.astype(np.float32))
            np.save(pf_out / f"{stem}.npy", pf_raw)
            done += 1
        except Exception as exc:
            errors.append(f"{stem}: {exc}")

    prov = {
        "stats_json": str(stats_json),
        "raw_root": str(raw_root),
        "normalized_root": str(norm_root),
        "n_samples": done,
        "errors": errors,
    }
    (norm_root / "normalization_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(stats_json, norm_root / "normalization_stats.json")
    return prov


def normalize_iteration_labels(
    cfg: dict,
    iteration_dir: Path,
    groot: Path,
) -> dict[str, Any]:
    """Package labels/ → raw_dataset/ → dataset_norm/ using training normalization stats."""
    manifest_path = iteration_dir / "ingest_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run ingest first — missing {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats_rel = cfg.get(
        "normalization_stats_json",
        "datasets/data_multifreq_norm/normalization_stats.json",
    )
    stats_json = groot / stats_rel
    hm_stats, imp_stats, _ = _load_training_stats(stats_json)
    heatmap_only_labels = bool(cfg.get("heatmap_only_labels", False))

    raw_root = iteration_dir / cfg.get("raw_dataset_subdir", "raw_dataset")
    norm_root = iteration_dir / cfg.get("normalized_dataset_subdir", "dataset_norm")

    print(f"\n=== Package raw dataset ===")
    print(f"  Stats: {stats_json}")
    if heatmap_only_labels:
        print("  Option B: heatmap-only labels (impedance packaging skipped)")
    packaged = package_raw_dataset(
        iteration_dir / "labels",
        raw_root,
        manifest,
        groot,
        imp_stats,
        heatmap_only_labels=heatmap_only_labels,
        hm_stats=hm_stats,
    )
    ok_pack = sum(1 for r in packaged if r.get("raw_packaged"))
    print(f"  Packaged {ok_pack}/{len(packaged)} samples → {raw_root}")

    print(f"\n=== Normalize (training stats) ===")
    report = normalize_raw_tree(
        raw_root,
        norm_root,
        stats_json,
        overwrite=bool(cfg.get("normalize_overwrite", False)),
        heatmap_only_labels=heatmap_only_labels,
        groot=groot,
    )
    print(f"  Normalized {report['n_samples']} samples → {norm_root}")
    if report["errors"]:
        print(f"  Errors ({len(report['errors'])}):")
        for e in report["errors"][:5]:
            print(f"    {e}")

    out = {"packaged": packaged, "normalize_report": report, "normalized_root": str(norm_root)}
    (iteration_dir / "normalize_report.json").write_text(
        json.dumps(out, indent=2) + "\n",
        encoding="utf-8",
    )
    return out
