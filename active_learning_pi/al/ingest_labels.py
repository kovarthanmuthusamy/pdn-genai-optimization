"""Ingest ECADSTAR PI-Distribution outputs into per-sample label directories.

Run:
    python active_learning_pi/al/ingest_labels.py"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pipelines.dataset_sim.ecadstar import parse_pi_number, resolve_windows_path
from pipelines.dataset_sim.paths import heatmap_map_path


def _find_pi_folders(emc_dir: Path) -> dict[int, Path]:
    """Map PI number → folder under EMC (PI-1, PI-2, …)."""
    out: dict[int, Path] = {}
    if not emc_dir.is_dir():
        return out
    for item in emc_dir.iterdir():
        if not item.is_dir():
            continue
        pi_num = parse_pi_number(item.name)
        if pi_num is not None:
            out[pi_num] = item
    return out


def ingest_simulation_outputs(
    cfg: dict,
    selected: list[dict[str, Any]],
    labels_dir: Path,
    groot: Path,
) -> list[dict[str, Any]]:
    """
    Read PI-1..PI-N folders from ECADStar .emc directory; extract .map per candidate MHz.
    """
    if str(groot) not in sys.path:
        sys.path.insert(0, str(groot))

    from libs.data_creation.heatmap import create_Heatmaps, load_mask_board  # noqa: E402

    emc_dir = resolve_windows_path(str(cfg["ecadstar"]["emc_output_dir"]))
    mask_path = groot / cfg.get("mask_board", "configs/binary_mask.npy")
    mask_board = load_mask_board(str(mask_path)) if mask_path.is_file() else None

    pi_folders = _find_pi_folders(emc_dir)
    if len(pi_folders) < len(selected):
        raise RuntimeError(
            f"Expected at least {len(selected)} PI folders in {emc_dir}, found {len(pi_folders)}"
        )

    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for i, sel in enumerate(selected):
        pi_num = i + 1
        pi_dir = pi_folders.get(pi_num)
        sample_dir = labels_dir / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        mhz = float(sel.get("mhz", 200.0))
        meta: dict[str, Any] = {
            **sel,
            "pi_number": pi_num,
            "sample_dir": str(sample_dir),
            "pred_impedance_norm": sel.get("pred_impedance_norm"),
        }

        if pi_dir is None:
            meta["ingest_ok"] = False
            meta["ingest_error"] = f"PI-{pi_num} folder missing under {emc_dir}"
            manifest.append(meta)
            continue

        map_src = heatmap_map_path(pi_dir, mhz)
        if not map_src.is_file():
            alt_maps = sorted(pi_dir.rglob("Z_*MHz.map"))
            map_src = alt_maps[0] if alt_maps else map_src

        if not map_src.is_file():
            meta["ingest_ok"] = False
            meta["ingest_error"] = f"No .map for {mhz:g} MHz under {pi_dir.name}"
            manifest.append(meta)
            continue

        map_dst = sample_dir / map_src.name
        if map_dst.exists():
            map_dst.unlink()
        shutil.copy2(map_src, map_dst)

        occ = np.array(sel["occupancy"], dtype=np.int8)
        np.save(sample_dir / "occupancy.npy", occ)
        np.save(sample_dir / "pi_freq_mhz.npy", np.float32(mhz))

        meta["pi_folder"] = str(pi_dir)
        meta["map_file"] = map_src.name
        try:
            hm = create_Heatmaps(
                str(map_dst),
                grid_size=int(cfg.get("heatmap_grid_size", 64)),
                mask_board=mask_board,
            )
            np.save(sample_dir / "heatmap.npy", hm)
            meta["heatmap_shape"] = list(hm.shape)
            meta["ingest_ok"] = True
        except Exception as exc:
            meta["ingest_ok"] = False
            meta["ingest_error"] = str(exc)

        manifest.append(meta)

    return manifest
