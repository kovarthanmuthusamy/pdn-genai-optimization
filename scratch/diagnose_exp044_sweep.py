#!/usr/bin/env python3
"""Quick encode vs layout diagnostic for exp044 at off-anchor MHz."""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import torch

from repo_paths import REPO_ROOT, setup_path

setup_path()

from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders
from experiments.exp044.codes.inference_vae import VAEInference
from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm

CKPT = _ROOT / "experiments/exp044/checkpoints/last_model.pt"
TARGET_MHZ = 330.0
K_FILTER = 30
MAX_ROWS = 32


def fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float) -> float:
    thr = bg + 0.5
    fg = (target > thr).float()
    err = (recon - target).pow(2)
    n = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return float(((err * fg).sum(dim=(1, 2, 3)) / n).mean().item())


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eng = VAEInference(checkpoint_path=str(CKPT), device=device)
    base = getattr(eng.model, "_orig_mod", eng.model)
    bg = float(eng.background_value)
    pi_alt = torch.tensor([pi_freq_mhz_to_norm(TARGET_MHZ)], device=device)

    _, val_ld = create_multifreq_data_loaders(
        data_dir=str(_ROOT / "datasets/data_multifreq_norm"),
        batch_size=64,
        num_workers=0,
        cache_in_ram=False,
    )

    rows: list[dict[str, float]] = []
    for batch in val_ld:
        hm = batch["heatmap_norm"]
        if hm.dim() == 3:
            hm = hm.unsqueeze(1)
        occ = batch["occupancy"].to(device)
        imp = batch["impedance"].to(device)
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        K = batch["K"].to(device)
        pi = batch["PI_freq"].to(device)

        for j in range(hm.shape[0]):
            if int(K[j].item()) != K_FILTER:
                continue
            hm_j = hm[j : j + 1].to(device)
            hm_enc = hm_j.masked_fill(hm_j < bg, 0.0)
            occ_j, imp_j = occ[j : j + 1], imp[j : j + 1]
            K_j, pi_j = K[j : j + 1], pi[j : j + 1]
            pi_t = pi_alt.expand_as(pi_j)
            with torch.no_grad():
                z_lay = base.encode_layout_latent(occ_j, imp_j, K_j, pi_j)
                rh_lay_nat, _, _ = base.decode(z_lay, K_j, pi_j)
                rh_lay_x, _, _ = base.decode(z_lay, K_j, pi_t)
                rh_enc_nat, _, _, _, _, _ = base(hm_enc, occ_j, imp_j, K_j, pi_j)
                rh_enc_x, _, _, _, _, _ = base(hm_enc, occ_j, imp_j, K_j, pi_t)
            rows.append(
                {
                    "layout@native": fg_mse(rh_lay_nat, hm_j, bg),
                    f"layout@{int(TARGET_MHZ)}": fg_mse(rh_lay_x, hm_j, bg),
                    "encode@native": fg_mse(rh_enc_nat, hm_j, bg),
                    f"encode@{int(TARGET_MHZ)}": fg_mse(rh_enc_x, hm_j, bg),
                }
            )
            if len(rows) >= MAX_ROWS:
                break
        if len(rows) >= MAX_ROWS:
            break

    print(f"exp044 val K={K_FILTER}  n={len(rows)}  (fg MSE in train norm space)")
    for k in rows[0]:
        vals = [r[k] for r in rows]
        print(f"  {k:18s} mean={st.mean(vals):.4f}  median={st.median(vals):.4f}")


if __name__ == "__main__":
    main()
