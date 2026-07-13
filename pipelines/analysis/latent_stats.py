#!/usr/bin/env python3
"""Per-modality latent statistics from a trained VAE checkpoint.

Run: python pipelines/analysis/latent_stats.py"""
from __future__ import annotations

from repo_paths import setup_path

setup_path()

import json
import sys
from pathlib import Path

import numpy as np
import torch


from experiments.exp021.codes.vae_multi_input_simple import MultiInputVAE
from source.others.dataloader import create_data_loaders

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/analysis/latent_stats.py
# =============================================================================

CHECKPOINT_PATH = "experiments/exp021/checkpoints/last_model.pt"
DATA_ROOT = "datasets/data_norm"
DEVICE = "cuda"

# =============================================================================

_MODALITIES = ("heatmap", "impedance", "shared")


def compute_latent_stats(checkpoint_path: str, data_root: str = DATA_ROOT, device: str = DEVICE) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MultiInputVAE(
        latent_dim=116, heatmap_private_dim=32, occupancy_private_dim=0,
        impedance_private_dim=16, shared_dim=68,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_loader, _ = create_data_loaders(data_root, batch_size=256, num_workers=4, normalize=False, train_split=0.9, seed=42)
    buckets_mu = {n: [] for n in _MODALITIES}
    buckets_std = {n: [] for n in _MODALITIES}

    print(f"Computing latent stats over {len(train_loader)} batches...")
    with torch.no_grad():
        for batch in train_loader:
            hm, occ, imp = batch["heatmap_norm"].to(device), batch["occupancy"].to(device), batch["impedance"].to(device)
            _, _, _, mu_gauss, logvar_gauss, mod_stats = model(hm, occ, imp)
            for name in ("heatmap", "impedance"):
                mu_m, lv_m = mod_stats[name]
                buckets_mu[name].append(mu_m.cpu())
                buckets_std[name].append(torch.exp(0.5 * lv_m).cpu())
            sd = model.shared_dim
            buckets_mu["shared"].append(mu_gauss[:, -sd:].cpu())
            buckets_std["shared"].append(torch.exp(0.5 * logvar_gauss[:, -sd:]).cpu())

    stats = {}
    for name in _MODALITIES:
        mu_all = torch.cat(buckets_mu[name], dim=0)
        std_all = torch.cat(buckets_std[name], dim=0)
        mu_per = mu_all.mean(0)
        std_per = mu_all.std(0)
        agg = torch.sqrt(mu_all.var(0) + (std_all**2).mean(0))
        stats[name] = {
            "mu_mean": mu_per.mean().item(), "mu_std": std_per.mean().item(),
            "mu_mean_per_dim": mu_per.tolist(), "mu_std_per_dim": std_per.tolist(),
            "sigma_mean": std_all.mean().item(), "agg_std": agg.mean().item(),
            "agg_std_per_dim": agg.tolist(), "n_samples": mu_all.shape[0], "n_dims": mu_all.shape[1],
        }
        print(f"\n{name} (dim={mu_all.shape[1]}): mu={stats[name]['mu_mean']:.4f} σ_agg={stats[name]['agg_std']:.4f}")
    return stats


if __name__ == "__main__":
    out = Path(CHECKPOINT_PATH).parent.parent / "metrics" / "latent_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compute_latent_stats(CHECKPOINT_PATH, DATA_ROOT, DEVICE), indent=2), encoding="utf-8")
    print(f"\nSaved latent stats to: {out}")
