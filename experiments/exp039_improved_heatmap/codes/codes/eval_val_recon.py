"""Validation reconstruction metrics for exp038 (occ accuracy + impedance peaks).

Run:
    python experiments/exp039_improved_heatmap/codes/eval_val_recon.py

Run:
    python experiments/exp039_improved_heatmap/codes/eval_val_recon.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve()
PROJECT_ROOT = next(
    (str(p) for p in _ROOT.parents if (p / "src_vae").is_dir() and (p / "experiments").is_dir()),
    str(_ROOT.parents[3]),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders
from experiments.exp038_true_multi.codes.impedance_spectrum_loss import local_peak_indices
from experiments.exp038_true_multi.codes.vae_multi_input_simple import MultiInputVAE

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

CHECKPOINT_NAME = "last_model.pt"
MAX_BATCHES = 0  # 0 = full val set

# =============================================================================


def _imp_ch0(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] if x.dim() == 3 else x


def _decode_occ_topk(logits: torch.Tensor, k_per_sample: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(logits)
    for i in range(logits.shape[0]):
        k = int(k_per_sample[i].item())
        if k > 0:
            out[i].scatter_(0, logits[i].topk(min(k, 52)).indices, 1.0)
    return out


def _predict_k(logits: torch.Tensor) -> torch.Tensor:
    """K from sigmoid mass (rounded), clamped to [1, 52]."""
    k = torch.sigmoid(logits).sum(dim=-1).round().long()
    return k.clamp(1, 52)


@torch.inference_mode()
def main() -> None:
    exp = Path(PROJECT_ROOT) / "experiments/exp039_improved_heatmap"
    ckpt_path = exp / "checkpoints" / CHECKPOINT_NAME
    cfg = json.loads((exp / "config.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    imp_std = float(json.loads((Path(cfg["data_dir"]) / "normalization_stats.json").read_text())
                         ["Impedance"]["log_std"]) or 1.0

    model = MultiInputVAE(
        latent_dim=int(cfg["latent_dim"]),
        cond_dim=int(cfg["cond_dim"]),
        heatmap_private_dim=int(cfg["heatmap_private_dim"]),
        modality_dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    _, val_ld = create_multifreq_data_loaders(
        data_dir=cfg["data_dir"], batch_size=64, num_workers=2,
        train_split=float(cfg["train_split"]), seed=42, pin_memory=device.type == "cuda",
        split_by_design=cfg.get("split_by_design", True),
        stratify_by_k=cfg.get("stratify_by_k", True),
        balance_k=False, balance_freq=False,
        cache_in_ram=cfg.get("cache_in_ram", True),
    )

    n_b = 0
    occ_bce = occ_k_match = occ_k_pred_match = slot_acc = 0.0
    imp_mse = peak_mse = under_peak = max_peak_err = peak_idx_err = 0.0
    num_peaks = int(cfg.get("impedance_num_peaks", 8))
    topk_k = int(cfg.get("impedance_topk_k", 15))

    for batch in val_ld:
        hm = batch["heatmap_norm"].to(device)
        occ = batch["occupancy"].to(device)
        imp = batch["impedance"].to(device)
        K = batch["K"].to(device)
        pi = batch.get("PI_freq", torch.full((hm.shape[0],), 0.828, device=device)).to(device)
        if hm.dim() == 3:
            hm = hm.unsqueeze(1)
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        elif imp.dim() == 3:
            imp = imp[:, :1]
        hm_enc = hm.masked_fill(hm < float(cfg.get("background_value", -3.62)) + 0.5, 0.0)

        _, occ_logits, imp_pred, _, _, _ = model(hm_enc, occ, imp, K, pi)
        occ_prob = torch.sigmoid(occ_logits)
        K_pred = _predict_k(occ_logits)
        occ_bin_gt = _decode_occ_topk(occ_logits, K)
        occ_bin_pred = _decode_occ_topk(occ_logits, K_pred)

        B = occ.shape[0]
        occ_bce += F.binary_cross_entropy(occ_prob, occ, reduction="mean").item() * B
        occ_k_match += (occ_bin_gt.sum(1).round() == K.float()).float().sum().item()
        occ_k_pred_match += (K_pred.float() == K.float()).float().sum().item()
        slot_acc += ((occ_bin_pred == (occ > 0.5).float()).float().mean(1)).sum().item()

        r = _imp_ch0(imp_pred) * imp_std
        t = _imp_ch0(imp) * imp_std
        imp_mse += F.mse_loss(r, t, reduction="sum").item()
        idx = t.topk(min(topk_k, t.shape[-1]), dim=-1).indices
        peak_mse += F.mse_loss(r.gather(1, idx), t.gather(1, idx), reduction="sum").item()
        under_peak += (r.gather(1, idx) < t.gather(1, idx)).float().mean().item() * B
        max_peak_err += (r.max(1).values - t.max(1).values).abs().mean().item() * B
        denom = max(t.shape[-1] - 1, 1)
        ti = local_peak_indices(t, num_peaks).float()
        ri = local_peak_indices(r, num_peaks).float()
        ti_s, _ = ti.sort(dim=-1)
        ri_s, _ = ri.sort(dim=-1)
        peak_idx_err += ((ti_s / denom) - (ri_s / denom)).abs().mean().item() * B

        n_b += B
        if MAX_BATCHES > 0 and n_b >= MAX_BATCHES * 64:
            break

    n = max(n_b, 1)
    print(f"Checkpoint: {ckpt_path.name}  (epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss', 0):.4f})")
    print(f"Val samples: {n_b}")
    print("--- Occupancy ---")
    print(f"  BCE (prob):     {occ_bce / n:.4f}   (focal train loss ~0.23–0.25 is same order)")
    print(f"  Slot accuracy:  {slot_acc / n:.4f}   (top-K decode with predicted K)")
    print(f"  K count match:  {occ_k_pred_match / n:.4f}   (sigmoid sum vs true K)")
    print(f"  K match (GT K decode): {occ_k_match / n:.4f}   (oracle K for slot set only)")
    print("--- Impedance (log-z norm × std) ---")
    print(f"  Full MSE:       {imp_mse / n:.6f}")
    print(f"  Top-{topk_k} MSE:     {peak_mse / n:.6f}   (val loss ~0.12 includes Huber+topk mix)")
    print(f"  Under-peak frac:{under_peak / n:.4f}   (recon < target at peaks)")
    print(f"  Max peak |err|: {max_peak_err / n:.4f}")
    print(f"  Peak index |err| (norm freq, top-{num_peaks}): {peak_idx_err / n:.4f}")


if __name__ == "__main__":
    main()
