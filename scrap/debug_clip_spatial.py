#!/usr/bin/env python3
"""How much does z-clip flatten spatial structure per MHz?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders
from experiments.exp048.codes.exp048_eval_common import pi_norm_to_mhz
from src_vae.others.norm_stats import load_norm_stats


def spatial_stats(data_dir: str, target_mhz: float, max_maps: int = 200) -> dict:
    bundle = load_norm_stats(data_dir)
    hm = bundle.heatmap
    _, val_ld = create_multifreq_data_loaders(
        data_dir=data_dir,
        batch_size=64,
        num_workers=0,
        train_split=0.9,
        seed=42,
        balance_k=False,
        balance_freq=False,
        cache_in_ram=False,
    )
    b = hm.bin_stats(target_mhz)
    clip_hi = b.clip_max
    at_clip_fracs: list[float] = []
    spatial_grads: list[float] = []
    maps_at_clip: list[float] = []
    top_decile_grads: list[float] = []
    n_maps = 0
    for batch in val_ld:
        hm_t = batch["heatmap_norm"]
        pi = batch["PI_freq"].numpy()
        mhz = pi_norm_to_mhz(pi)
        sel = np.abs(mhz - target_mhz) < 2.0
        if not sel.any():
            continue
        for j in np.where(sel)[0]:
            z = hm_t[j, 0].numpy()
            fg = z > b.background_value + 0.5
            if fg.sum() < 10:
                continue
            zfg = z[fg]
            at_clip_fracs.append(float((zfg >= clip_hi - 1e-4).mean()))
            maps_at_clip.append(float(zfg.max() >= clip_hi - 1e-4))
            gy = np.abs(np.diff(z, axis=0)).mean()
            gx = np.abs(np.diff(z, axis=1)).mean()
            spatial_grads.append(float((gy + gx) / 2))
            thr = np.percentile(zfg, 90)
            top = zfg[zfg >= thr]
            if top.size > 4:
                # local gradient proxy on peak region: std/mean (higher = more spatial contrast)
                top_decile_grads.append(float(np.std(top) / max(np.mean(top), 1e-6)))
            n_maps += 1
            if n_maps >= max_maps:
                break
        if n_maps >= max_maps:
            break
    return {
        "n": n_maps,
        "clip_hi": clip_hi,
        "pct_pixels_at_clip": float(np.mean(at_clip_fracs) * 100),
        "pct_maps_max_at_clip": float(np.mean(maps_at_clip) * 100),
        "mean_spatial_grad_z": float(np.mean(spatial_grads)),
        "top_decile_contrast": float(np.mean(top_decile_grads)) if top_decile_grads else 0.0,
    }


def main() -> None:
    print("=== data_multi_norm_robust (exp048) ===")
    for m in [10, 270, 400]:
        s = spatial_stats("data_multi_norm_robust", m)
        print(
            f"{m} MHz: maps_max@clip={s['pct_maps_max_at_clip']:.1f}%  "
            f"fg_px@clip={s['pct_pixels_at_clip']:.1f}%  "
            f"|grad|_z={s['mean_spatial_grad_z']:.4f}  "
            f"top10%_contrast={s['top_decile_contrast']:.4f}  clip_hi={s['clip_hi']:.3f}"
        )


if __name__ == "__main__":
    main()
