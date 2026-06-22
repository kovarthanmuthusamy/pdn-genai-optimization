"""Ingest ECADSTAR PI-Distribution .map outputs into per-sample label directories.

Run:
    python active_learning_pi/al/ingest_labels.py"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


_PI_RE = re.compile(r"PI[-_]?([0-9]+)", re.IGNORECASE)


def _find_pi_files(emc_dir: Path) -> list[Path]:
    if not emc_dir.is_dir():
        return []
    out = []
    for p in emc_dir.iterdir():
        if p.is_file() and _PI_RE.search(p.name):
            out.append(p)
    return sorted(out, key=lambda p: int(_PI_RE.search(p.name).group(1)))


def ingest_simulation_outputs(
    cfg: dict,
    selected: list[dict[str, Any]],
    labels_dir: Path,
    groot: Path,
) -> list[dict[str, Any]]:
    """
    Move PI-* from ECADStar .emc folder into labels_dir/sample_{i}/ and build heatmaps from .map.
    """
    if str(groot) not in sys.path:
        sys.path.insert(0, str(groot))

    from libs.data_creation.heatmap import create_Heatmaps, load_mask_board  # noqa: E402

    emc_dir = Path(cfg["ecadstar"]["emc_output_dir"])
    mask_path = groot / cfg.get("mask_board", "configs/binary_mask.npy")
    mask_board = load_mask_board(str(mask_path)) if mask_path.is_file() else None

    pi_files = _find_pi_files(emc_dir)
    if len(pi_files) < len(selected):
        raise RuntimeError(
            f"Expected at least {len(selected)} PI outputs in {emc_dir}, found {len(pi_files)}"
        )

    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for i, (sel, pi_src) in enumerate(zip(selected, pi_files[: len(selected)])):
        sample_dir = labels_dir / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        map_dst = sample_dir / pi_src.name
        if map_dst.exists():
            map_dst.unlink()
        shutil.move(str(pi_src), str(map_dst))

        occ = np.array(sel["occupancy"], dtype=np.int8)
        np.save(sample_dir / "occupancy.npy", occ)
        np.save(sample_dir / "pi_freq_mhz.npy", np.float32(sel["mhz"]))

        meta = {
            **sel,
            "pi_file": pi_src.name,
            "sample_dir": str(sample_dir),
            "pred_impedance_norm": sel.get("pred_impedance_norm"),
        }
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
