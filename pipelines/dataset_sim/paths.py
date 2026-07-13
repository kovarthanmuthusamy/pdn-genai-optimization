"""Shared Windows paths and heatmap location helpers for sim → append pipeline."""
from __future__ import annotations

from pathlib import Path

from pipelines.dataset_sim.ecadstar import resolve_windows_path
from pipelines.dataset_sim.move_outputs import heatmaps_dir_name

# Single source of truth — sim move destination == append Raw input
RAW_ROOT_WIN = r"C:\Users\muthusamy\Desktop\Raw"
PEB_DIR_WIN = r"C:\Users\muthusamy\Desktop\Raw\peb"


def resolve_raw_root(win_path: str | None = None) -> Path:
    return resolve_windows_path(win_path or RAW_ROOT_WIN)


def heatmap_raw_dir(raw_root: Path, mhz: float) -> Path:
    """Folder append reads and sim writes (``heatmaps_{MHz}MHz``, legacy ``heatmap_*`` ok)."""
    tag = int(round(float(mhz)))
    for name in (f"heatmap_{tag}MHz", f"heatmaps_{tag}MHz"):
        candidate = raw_root / name
        if candidate.is_dir():
            return candidate
    return raw_root / heatmaps_dir_name(mhz)


def heatmap_map_path(pi_dir: Path, mhz: float) -> Path:
    tag = int(round(float(mhz)))
    return pi_dir / "Power_GND" / f"Z_{tag:04d}.000MHz.map"


def verify_heatmap_ready(
    mhz: float,
    *,
    raw_root: Path | None = None,
    raw_root_win: str | None = None,
) -> Path:
    """Raise if sim move output is not ready for append at this MHz."""
    root = raw_root or resolve_raw_root(raw_root_win)
    hm_dir = heatmap_raw_dir(root, mhz)

    if not hm_dir.is_dir():
        raise FileNotFoundError(
            f"Heatmap folder missing: {hm_dir} "
            f"(expected {heatmaps_dir_name(mhz)} under {root})"
        )

    pi1 = hm_dir / "PI-1"
    if not pi1.is_dir():
        raise FileNotFoundError(f"PI-1 missing under {hm_dir}")

    sample_map = heatmap_map_path(pi1, mhz)
    if not sample_map.is_file():
        raise FileNotFoundError(f"Sample .map missing (append would fail): {sample_map}")

    return hm_dir
