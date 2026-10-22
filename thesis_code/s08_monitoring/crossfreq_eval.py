"""Cross-frequency check run during training; picks ``best_off_anchor_model.pt``.

Encode a held-out placement at one frequency (real map + placement + spectrum), decode at a
different real frequency of the same placement (its cross-frequency partner), and score against the
real map there -- what maps at a peak frequency rely on. Encoder and decoder get the typed
(B, 52, T) placement as in training.

Usage (import only):
run_crossfreq_eval(model, val_loader, max_batches, device, epoch, out_csv, n_decap_types)
    -> {"epoch", "n", "pearson_at_target", "pearson_vs_source", "fg_mse_at_target"}; appends to out_csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from thesis_code.common.board import board_fg, zero_offboard
from thesis_code.common.placement import ensure_batch_onehot
from thesis_code.s06_losses.heatmap import pearson_fg
from thesis_code.s07_training.distributed import unwrap_model

CROSSFREQ_FIELDS = ("epoch", "n", "pearson_at_target", "pearson_vs_source", "fg_mse_at_target")


def _fg_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fg = board_fg(target)
    return ((recon - target).pow(2) * fg).sum(dim=(1, 2, 3)) / fg.sum(dim=(1, 2, 3)).clamp(min=1.0)


def _as_map(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(1) if x.dim() == 3 else x


@torch.inference_mode()
def run_crossfreq_eval(model: torch.nn.Module, val_loader, *, max_batches: int = 12,
                       device: str | torch.device = "cuda", epoch: int = 0, out_csv: Path | str | None = None,
                       n_decap_types: int = 3) -> dict:
    base = unwrap_model(model)
    model.eval()
    at_target: list[float] = []
    vs_source: list[float] = []
    mse_target: list[float] = []
    for bi, batch in enumerate(val_loader):
        if max_batches > 0 and bi >= max_batches:
            break
        hm, hm_alt = _as_map(batch["heatmap_norm"].to(device)), _as_map(batch["heatmap_norm_alt"].to(device))
        imp = batch["impedance"].to(device)
        imp = imp.unsqueeze(1) if imp.dim() == 2 else imp[:, :1]
        K = batch["K"].to(device)
        occ = ensure_batch_onehot(batch["occupancy"].to(device), n_decap_types)
        z = base.encode(zero_offboard(hm), occ, imp, K, batch["PI_freq"].to(device))[1]
        recon = base.decode(z, K, batch["PI_freq_alt"].to(device), occupancy=occ)[0]
        at_target += pearson_fg(recon, hm_alt).tolist()
        vs_source += pearson_fg(recon, hm).tolist()
        mse_target += _fg_mse(recon, hm_alt).tolist()
    if not at_target:
        return {}

    mean = lambda v: sum(v) / len(v)  # noqa: E731
    row = {"epoch": epoch, "n": len(at_target), "pearson_at_target": mean(at_target),
           "pearson_vs_source": mean(vs_source), "fg_mse_at_target": mean(mse_target)}
    if out_csv is not None:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.is_file()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CROSSFREQ_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    print(f"  Cross-freq ep {epoch}: pearson@target {row['pearson_at_target']:.4f}  "
          f"(vs source map {row['pearson_vs_source']:.4f})  n={row['n']}", flush=True)
    return row
