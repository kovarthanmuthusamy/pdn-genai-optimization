#!/usr/bin/env python3
"""Inspect robust per-MHz stats and clip ceilings."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_vae.others.norm_stats import load_norm_stats

STATS = ROOT / "data_multi_norm_robust" / "normalization_stats.json"


def main() -> None:
    if not STATS.is_file():
        raise SystemExit(f"Missing {STATS}")
    raw = json.loads(STATS.read_text(encoding="utf-8"))
    hm = raw["Heatmap"]
    print("norm_mode:", hm.get("norm_mode"))
    print("global clip:", hm.get("clip_min"), hm.get("clip_max"))
    print()
    for mhz in ["10.0", "63.0", "270.0", "330.0", "400.0", "500.0"]:
        b = hm["by_mhz"].get(mhz)
        if not b:
            print(f"{mhz} MHz: MISSING")
            continue
        zmax = float(b["clip_max"])
        med, iqr = float(b["median"]), float(b["iqr"])
        ceil = float(np.expm1(zmax * iqr + med))

        def z_at(ohm: float) -> float:
            return (np.log1p(ohm) - med) / iqr

        print(
            f"{mhz} MHz: med={med:.4f} iqr={iqr:.4f} "
            f"clip=[{b['clip_min']:.3f},{zmax:.3f}] ceiling={ceil:.2f}Ω  "
            f"z@10Ω={z_at(10):.2f} z@15Ω={z_at(15):.2f} z@20Ω={z_at(20):.2f}"
        )

    bundle = load_norm_stats(ROOT / "data_multi_norm_robust")
    print()
    for m in [10, 70, 200, 270, 400]:
        print(f"physical_ceiling_ohm({m}) = {bundle.heatmap.physical_ceiling_ohm(m):.2f}Ω")

    # Val-set real max distribution at anchor MHz (quick sanity)
    try:
        import torch
        from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders
        from experiments.exp048.codes.exp048_eval_common import pi_norm_to_mhz

        _, val_ld = create_multifreq_data_loaders(
            data_dir=str(ROOT / "data_multi_norm_robust"),
            batch_size=128,
            num_workers=0,
            train_split=0.9,
            seed=42,
            balance_k=False,
            balance_freq=False,
            cache_in_ram=False,
        )
        bundle = load_norm_stats(ROOT / "data_multi_norm_robust")
        for target in [270.0, 400.0]:
            vals = []
            for batch in val_ld:
                hm = batch["heatmap_norm"]
                pi = batch["PI_freq"]
                mhz = pi_norm_to_mhz(pi.numpy())
                sel = np.abs(mhz - target) < 2.0
                if not sel.any():
                    continue
                hm_s = hm[sel]
                pi_s = pi[sel]
                phys = bundle.heatmap.norm_to_physical(hm_s, pi_norm=pi_s, apply_clip=True)
                vals.append(phys.amax(dim=(1, 2, 3)).numpy())
            if vals:
                v = np.concatenate(vals)
                ceil = bundle.heatmap.physical_ceiling_ohm(target)
                print(
                    f"\nval real_max @ {target:.0f} MHz: "
                    f"p50={np.percentile(v,50):.2f} p95={np.percentile(v,95):.2f} "
                    f"p99={np.percentile(v,99):.2f} max={v.max():.2f}  ceiling={ceil:.2f}Ω"
                )
    except Exception as e:
        print(f"\n(val scan skipped: {e})")


if __name__ == "__main__":
    main()
