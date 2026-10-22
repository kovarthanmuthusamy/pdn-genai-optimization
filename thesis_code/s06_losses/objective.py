"""The full training objective of the VAE.

    total = w_hm * heatmap + occupancy_weight * occupancy + w_imp * impedance
          + occ_k_consistency_weight * occ_count
          + beta * KL + mu_hinge + mu_bias + per_expert_kl + sigma_reg
          + heatmap_peak_loc_weight * peak_location             (also folded into "heatmap_loss")
          + full_encode_cycle_consistency_weight * cycle        (train only)
          + w_cf * cross_freq + anticopy_weight * anticopy      (train only, from their start epochs)

``w_hm``, ``w_imp``, ``w_cf`` are the phase weights (s07_training/schedule.py). Every term is always on.
The decap (occupancy) loss is the same for every K.

Usage (import only):
LOSS_KEYS                          names of the per-epoch averaged terms (checkpoint val_metrics)
vae_loss(...)                      reconstruction + latent terms for one batch -> dict of scalars
cycle_consistency_loss(model, z, K, pi)            encode(decode(z)) ~= z through the full encoder
cross_freq_loss(model, z, K, occ, pi_alt, hm_alt, c)  decode z at another frequency vs its real map
anticopy_loss(rh_cf, hm_alt, hm_src)               the cross-frequency map may not copy the source
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from thesis_code.common.board import board_fg, zero_offboard
from thesis_code.common.placement import hard_onehot_from_logits
from thesis_code.s06_losses.guards import sanitize_loss_dict
from thesis_code.s06_losses.heatmap import heatmap_loss, pearson_fg, peak_location_loss
from thesis_code.s06_losses.impedance import impedance_spectrum_loss, impedance_weights

LOSS_KEYS = (
    "total_loss", "recon_loss", "kl_loss",
    "heatmap_loss", "heatmap_loss_tier_a", "heatmap_peak_loc_loss",
    "occupancy_loss", "occ_k_consistency_loss",
    "impedance_loss", "impedance_legacy_loss", "impedance_peak_loss",
    "expert_kl_loss", "sigma_reg_loss",
    "cross_freq_heatmap_loss", "anticopy_loss", "cycle_consistency_loss",
)


# ── Decap layout ─────────────────────────────────────────────────────────────

def occupancy_focal_loss(recon_occ: torch.Tensor, tgt_occ: torch.Tensor, gamma: float) -> torch.Tensor:
    """Per-sample (B,) focal BCE over 52 slots x T parts."""
    bce = F.binary_cross_entropy_with_logits(recon_occ, tgt_occ.clamp(0, 1), reduction="none")
    with torch.no_grad():
        pt = torch.exp(-bce)
    return ((1.0 - pt).pow(gamma) * bce).mean(dim=(1, 2))


def occupancy_count_loss(recon_occ: torch.Tensor, tgt_occ: torch.Tensor) -> torch.Tensor:
    """Mean |summed part probability - presence| per slot: keeps the decap count at K."""
    p_occ = torch.sigmoid(recon_occ).sum(dim=-1).clamp(0, 1)
    t_pres = tgt_occ.sum(dim=-1).clamp(0, 1)
    return (p_occ - t_pres).abs().mean(dim=1).mean()


# ── Latent ───────────────────────────────────────────────────────────────────

def expert_kl(expert_stats: dict) -> torch.Tensor:
    """KL of each expert to N(0, I), averaged over the experts."""
    return sum((-0.5 * (1.0 + lv - m.pow(2) - lv.exp())).mean() for m, lv in expert_stats.values()) / len(expert_stats)


def sigma_band_loss(expert_stats: dict, target: float) -> torch.Tensor:
    """Keep the fused posterior sigma (expert precisions + unit prior) in [target, 1.3 * target]."""
    prec = torch.stack([1.0 / (lv.exp() + 1e-8) for _, lv in expert_stats.values()]).sum(0) + 1.0
    sig = (1.0 / (prec + 1e-8)).sqrt()
    d_lo = (target - sig).clamp(min=0)
    d_hi = (sig - target * 1.3).clamp(min=0)
    return (d_lo.pow(2) * (target / sig.detach().clamp(min=0.05)) + d_hi.pow(2)).mean()


# ── Objective ────────────────────────────────────────────────────────────────

def vae_loss(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp, mu, logvar, expert_stats, K, *,
             c, beta: float, imp_log_std: float, weights: dict[str, float],
             penalty_scale: float, peak_scale: float) -> dict[str, torch.Tensor]:
    """Reconstruction, KL and latent-regularizer terms for one batch (see module docstring).

    ``weights``: phase weights {"heatmap_weight", "impedance_weight", ...}; ``penalty_scale`` /
    ``peak_scale``: impedance curriculum ramps. ``tgt_occ`` is the (B, 52, T) one-hot layout.
    """
    loss_hm = heatmap_loss(recon_hm, tgt_hm, c).mean()
    loss_occ = occupancy_focal_loss(recon_occ, tgt_occ, c.occupancy_focal_gamma).mean()
    imp_ps, imp_parts = impedance_spectrum_loss(recon_imp, tgt_imp, imp_log_std=imp_log_std,
                                                penalty_scale=penalty_scale, peak_scale=peak_scale,
                                                w=impedance_weights(c))
    loss_imp = imp_ps.mean()
    occ_k = occupancy_count_loss(recon_occ, tgt_occ)

    recon = (weights["heatmap_weight"] * loss_hm + c.occupancy_weight * loss_occ
             + weights["impedance_weight"] * loss_imp + c.occ_k_consistency_weight * occ_k)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).clamp(min=c.free_bits).mean()
    total = recon + beta * kl + c.mu_hinge_weight * F.relu(mu.abs() - c.mu_hinge_threshold).pow(2).mean()
    total = total + c.mu_bias_weight * mu.mean(0).pow(2).mean()
    exp_kl = expert_kl(expert_stats)
    total = total + c.per_expert_kl_weight * exp_kl
    sigma_reg = sigma_band_loss(expert_stats, c.sigma_reg_target)
    total = total + c.sigma_reg_weight * sigma_reg

    # soft-argmax hotspot location, counted in the total and in "heatmap_loss"
    peak = peak_location_loss(recon_hm, tgt_hm, board_fg(tgt_hm), float(c.background_value),
                              temperature=float(c.heatmap_peak_loc_temperature)).reshape(-1).mean()
    add = c.heatmap_peak_loc_weight * peak
    losses = {
        "total_loss": total + add, "recon_loss": recon, "kl_loss": kl,
        "heatmap_loss": loss_hm + add, "heatmap_loss_tier_a": loss_hm.detach(), "heatmap_peak_loc_loss": peak.detach(),
        "occupancy_loss": loss_occ, "occ_k_consistency_loss": occ_k,
        "impedance_loss": loss_imp, "impedance_legacy_loss": imp_parts["legacy"].mean(),
        "impedance_peak_loss": imp_parts["peak_extra"].mean(),
        "expert_kl_loss": exp_kl, "sigma_reg_loss": sigma_reg, "cross_freq_heatmap_loss": mu.new_zeros(()),
    }
    return sanitize_loss_dict(losses, c)


# ── Terms that need extra decodes (training only) ────────────────────────────

def cycle_consistency_loss(model, z, K, pi) -> torch.Tensor:
    """``encode(decode(z)) ~= z`` through the full encoder -- the round trip the latent search uses.

    As in the search: placement from z alone, read out as the typed placement CAD builds (the K
    most likely slots, each with its most likely part) with a straight-through gradient, then map
    and spectrum conditioned on it.
    """
    occ_logits = model.decode(z, K, pi, occupancy=None)[1]
    occ_hard = hard_onehot_from_logits(occ_logits, K, ste=True)
    hm_recon, _, imp_recon = model.decode(z, K, pi, occupancy=occ_hard)
    z_cyc = model.encode(zero_offboard(hm_recon), occ_hard, imp_recon, K, pi)[1]
    return F.mse_loss(z_cyc, z.detach())


def cross_freq_loss(model, z, K, occ, pi_alt, hm_alt, *, c):
    """Decode the SAME z at the partner frequency and score it against that frequency's real map
    (lite heatmap loss). Returns ``(loss, decoded map, target map)`` for ``anticopy_loss``."""
    if hm_alt.dim() == 3:
        hm_alt = hm_alt.unsqueeze(1)
    rh_cf = model.decode(z, K, pi_alt, occupancy=occ)[0]
    cf = heatmap_loss(rh_cf, hm_alt, c, lite=True)
    return (cf.mean() if cf.ndim else cf), rh_cf, hm_alt


def anticopy_loss(rh_cf, hm_alt, hm_src) -> torch.Tensor:
    """``relu(corr(pred, source) - corr(truth, source))``: the cross-frequency prediction may not
    look more like the map it was encoded from than the true answer does. Zero when correct."""
    p_pred_src = pearson_fg(rh_cf, hm_src)
    p_true_src = pearson_fg(hm_alt, hm_src).detach()
    return (p_pred_src - p_true_src).clamp(min=0.0).mean()
