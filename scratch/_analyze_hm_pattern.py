#!/usr/bin/env python3
"""Quick spatial analysis of sweep heatmaps vs real."""
import sys
from pathlib import Path
import numpy as np

from repo_paths import REPO_ROOT, setup_path

setup_path()

root = Path("experiments/exp043/multifreq_heatmap_sweep_30/freq_400MHz/K30")
mask = np.load("configs/binary_mask.npy").astype(bool)
if mask.ndim == 3:
    mask = mask[0]

for i in range(2):
    gen = np.load(root / f"data_sample_{i}" / "heatmap_physical.npy")
    if gen.ndim == 3:
        gen = gen[0]
    fg = gen * mask
    total = float(fg.sum())
    right = float(fg[:, -4:].sum())
    col_prof = fg.sum(axis=0)
    row_prof = fg.sum(axis=1)
    r = gen[mask]
    # load real from compare pipeline - heatmap from Real folder via create_Heatmaps path
    real_maps = sorted((root / "Real").rglob("Z_0400*.map"))
    if not real_maps:
        real_maps = sorted((root / "Real").rglob("*.map"))
    real = None
    if i < len(real_maps):
        from libs.data_creation.heatmap import create_Heatmaps
        real = create_Heatmaps(str(real_maps[i]), mask_board=mask, verbose=False)[0]

    print(f"sample_{i}: gen fg max={r.max():.3f} mean={r.mean():.3f} p95={np.percentile(r,95):.3f}")
    print(f"  energy right 4 cols: {right/max(total,1e-9):.1%}  peak col={int(col_prof.argmax())}/{gen.shape[1]}")
    if real is not None:
        rv = real[mask]
        corr = float(np.corrcoef(rv, r)[0, 1])
        print(f"  real fg max={rv.max():.3f} mean={rv.mean():.3f}  pearson r={corr:.3f}")
        rn = (rv - rv.min()) / max(rv.max() - rv.min(), 1e-12)
        gn = (r - r.min()) / max(r.max() - r.min(), 1e-12)
        print(f"  pattern MAE (minmax norm): {np.mean(np.abs(rn-gn)):.3f}")
