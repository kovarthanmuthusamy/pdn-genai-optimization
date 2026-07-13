#!/usr/bin/env python3
"""Check exp044 normalization consistency and encode-native reconstruction quality."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from repo_paths import REPO_ROOT, setup_path

setup_path()

from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders
from experiments.exp044.codes.inference_vae import VAEInference, load_experiment_config
from src_vae.others.heatmap_z_clip import heatmap_z_to_physical


def main() -> None:
    cfg = load_experiment_config(_ROOT / "experiments/exp044/config.yaml")
    stats = json.loads(
        (_ROOT / "datasets/data_multifreq_norm/normalization_stats.json").read_text()
    )
    hm = stats["Heatmap"]

    print("=== Config vs dataset stats ===")
    pairs = [
        ("background_value", "background_value", stats.get("background_value")),
        ("heatmap_z_clip_min", "clip_min", hm.get("clip_min")),
        ("heatmap_z_clip_max", "clip_max", hm.get("clip_max")),
    ]
    for cfg_k, stat_k, sv in pairs:
        cv = cfg.get(cfg_k)
        ok = cv is not None and sv is not None and abs(float(cv) - float(sv)) < 1e-4
        print(f"  {cfg_k}: cfg={cv}  stats={sv}  {'OK' if ok else 'MISMATCH'}")

    z_hi = float(hm["clip_max"])
    phys_hi = math.exp(z_hi * hm["log_std"] + hm["log_mean"]) - 1.0
    print(f"\n  z_clip_max -> physical ceiling: {phys_hi:.4f} Ω  (sweep gen_max=33.753)")
    print(f"  full data z_max={hm['z_max']:.4f} -> {math.exp(hm['z_max']*hm['log_std']+hm['log_mean'])-1:.1f} Ω uncapped")

    eng = VAEInference(str(_ROOT / "experiments/exp044/checkpoints/last_model.pt"), device="cuda")
    base = getattr(eng.model, "_orig_mod", eng.model)
    _, val = create_multifreq_data_loaders(
        str(_ROOT / "datasets/data_multifreq_norm"), batch_size=64, num_workers=0, cache_in_ram=False,
    )
    bg = float(eng.background_value)
    clip = eng.hm_z_clip

    mae_norm, mae_phys, sat_frac, n = [], [], [], 0
    for batch in val:
        hm = batch["heatmap_norm"].cuda()
        if hm.dim() == 3:
            hm = hm.unsqueeze(1)
        occ = batch["occupancy"].cuda()
        imp = batch["impedance"].cuda()
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        K = batch["K"].cuda()
        pi = batch["PI_freq"].cuda()
        hm_enc = hm.masked_fill(hm < bg, 0.0)
        with torch.no_grad():
            rh, _, _, _, _, _ = base(hm_enc, occ, imp, K, pi)
        if clip:
            sat_frac.append(float((rh >= clip[1] - 1e-4).float().mean().item()))
        mae_norm.append(float((rh - hm).abs().mean().item()))
        rp = heatmap_z_to_physical(rh, eng.hm_log_mean, eng.hm_log_std, clip_lo=clip[0], clip_hi=clip[1])
        tp = heatmap_z_to_physical(hm, eng.hm_log_mean, eng.hm_log_std)
        thr = bg + 0.5
        fg = (hm > thr).float()
        mae_phys.append(float(((rp - tp).abs() * fg).sum().item() / fg.sum().clamp(min=1).item()))
        n += 1
        if n >= 50:
            break

    print(f"\n=== Encode @ native π (val, {n} batches) ===")
    print(f"  MAE norm space:     {np.mean(mae_norm):.4f}  (val heatmap_loss ~1.17 is Huber+weighted)")
    print(f"  MAE physical Ω fg:  {np.mean(mae_phys):.4f}")
    print(f"  pixels at z_clip_hi: {np.mean(sat_frac)*100:.2f}%")


if __name__ == "__main__":
    main()
