"""Compute per-anchor foreground max (physical Ω) from data_multifreq_norm and write metrics JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.exp038_true_multi.codes.dataloader_multifreq import ANCHOR_MHZ  # noqa: E402

DATA = ROOT / "datasets" / "data_multifreq_norm"
OUT = Path(__file__).resolve().parents[1] / "metrics" / "anchor_hm_fg_max.json"


def main() -> None:
    stats_path = DATA / "normalization_stats.json"
    if not stats_path.is_file():
        raise SystemExit(f"Missing {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    hm_mean = float(stats["heatmap_log_mean"])
    hm_std = float(stats["heatmap_log_std"])
    mask_path = DATA / "binary_mask.npy"
    mask = np.load(mask_path) if mask_path.is_file() else None

    table: dict[str, float] = {}
    for mhz in ANCHOR_MHZ:
        mhz_int = int(round(mhz))
        tag = f"freq_{mhz_int}MHz"
        fg_maxes: list[float] = []
        for k_dir in sorted(DATA.glob(f"{tag}/K*")):
            for hm_path in k_dir.rglob("heatmap_physical.npy"):
                hm = np.load(hm_path)
                if mask is not None and mask.shape == hm.shape[-2:]:
                    fg = hm[..., mask] if hm.ndim == 3 else hm[mask]
                else:
                    fg = hm.reshape(-1)
                if fg.size:
                    fg_maxes.append(float(fg.max()))
            for hz_path in k_dir.rglob("heatmap_zscore.npy"):
                z = np.load(hz_path)
                phys = np.exp(z * hm_std + hm_mean) - 1e-6
                phys = np.clip(phys, 0, None)
                if mask is not None and mask.shape == phys.shape[-2:]:
                    fg = phys[..., mask] if phys.ndim == 3 else phys[mask]
                else:
                    fg = phys.reshape(-1)
                if fg.size:
                    fg_maxes.append(float(fg.max()))
        if not fg_maxes:
            print(f"  {mhz} MHz: no samples found under {DATA / tag}")
            continue
        key = str(int(mhz)) if mhz == mhz_int else str(mhz)
        table[key] = float(np.median(fg_maxes))
        print(f"  {mhz} MHz: median FG max = {table[key]:.3f} (n={len(fg_maxes)})")

    if not table:
        raise SystemExit("No heatmaps found; keeping existing JSON if any.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
