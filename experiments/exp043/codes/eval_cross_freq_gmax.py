"""Off-anchor eval for exp043 global-max + optional log1p train space."""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from experiments.exp043.codes.gmax_training_patch import transform_disk_heatmap
from src_vae.others.heatmap_gmax_norm import heatmap_fg_threshold
from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm


def _fg_mse(recon: torch.Tensor, target: torch.Tensor, fg_thr: float) -> torch.Tensor:
    fg = (target > fg_thr).float()
    err = (recon - target).pow(2)
    n = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (err * fg).sum(dim=(1, 2, 3)) / n


@torch.inference_mode()
def run_off_anchor_eval_gmax(
    model: torch.nn.Module,
    val_loader,
    *,
    c,
    off_anchor_mhz: tuple[float, ...],
    max_batches: int = 30,
    device: str | torch.device = "cuda",
    out_csv: Path | str | None = None,
) -> list[dict]:
    """Layout-z / encode-cross at off-anchor MHz; targets in train space."""
    model.eval()
    base = getattr(model, "_orig_mod", model)
    fg_thr = heatmap_fg_threshold(c)
    rows: list[dict] = []
    buckets: dict[tuple[str, float], list[float]] = {}

    for bi, batch in enumerate(val_loader):
        if max_batches > 0 and bi >= max_batches:
            break
        hm = transform_disk_heatmap(batch["heatmap_norm"].to(device), c)
        occ = batch["occupancy"].to(device)
        imp = batch["impedance"].to(device)
        K = batch["K"].to(device)
        pi_native = batch["PI_freq"].to(device)
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        elif imp.dim() == 3:
            imp = imp[:, :1]
        hm_enc = hm

        pi_ref_mhz = getattr(c, "cross_freq_layout_pi_ref_mhz", None)
        if pi_ref_mhz is not None:
            pi_layout = torch.full(
                (hm.shape[0],), pi_freq_mhz_to_norm(float(pi_ref_mhz)), device=device,
            )
        else:
            pi_layout = pi_native
        z_layout = base.encode_layout_latent(occ, imp, K, pi_layout)

        for mhz in off_anchor_mhz:
            pi_t = torch.full((hm.shape[0],), pi_freq_mhz_to_norm(mhz), device=device)
            rh_enc, _, _, _, _, _ = base(hm_enc, occ, imp, K, pi_t)
            rh_lay, _, _ = base.decode(z_layout, K, pi_t, occupancy=occ)
            mse_enc = _fg_mse(rh_enc, hm, fg_thr)
            mse_lay = _fg_mse(rh_lay, hm, fg_thr)
            for j in range(mse_enc.shape[0]):
                buckets.setdefault(("encode_cross", float(mhz)), []).append(float(mse_enc[j].item()))
                buckets.setdefault(("layout_cross", float(mhz)), []).append(float(mse_lay[j].item()))

    for (kind, mhz), vals in sorted(buckets.items()):
        if not vals:
            continue
        rows.append({
            "mhz": mhz,
            "kind": kind,
            "n": len(vals),
            "hm_fg_mse_mean": sum(vals) / len(vals),
        })

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["mhz", "kind", "n", "hm_fg_mse_mean"])
            w.writeheader()
            w.writerows(rows)

    for r in rows:
        print(
            f"    {r['kind']:14s} @ {r['mhz']:6.0f} MHz  "
            f"MSE={r['hm_fg_mse_mean']:.4f}  n={r['n']}",
        )
    return rows
