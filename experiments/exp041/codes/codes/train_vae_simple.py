"""Training — Multi-Input VAE (exp038_true_multi): multifreq PI heatmaps + peak losses."""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim

_ROOT = Path(__file__).resolve()
PROJECT_ROOT = next(
    (str(p) for p in _ROOT.parents if (p / "src_vae").is_dir() and (p / "experiments").is_dir()),
    str(_ROOT.parents[3]),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders
from experiments.exp038_true_multi.codes.impedance_spectrum_loss import (
    ImpedanceSpectrumWeights,
    impedance_spectrum_loss,
)
from experiments.exp038_true_multi.codes.physics_loss import PhysicsLoss
from experiments.exp038_true_multi.codes.vae_multi_input_simple import MultiInputVAE
from src_vae.others.vae_logger import VAETrainingLogger

# ── Config ────────────────────────────────────────────────────────────────────

_FRAC_FIELDS = (
    ("beta_end_epoch", "beta_end_frac"),
    ("beta_phase2_end_epoch", "beta_phase2_end_frac"),
    ("modality_dropout_anneal_epochs", "modality_dropout_anneal_frac"),
    ("physics_critic_warmup_epochs", "physics_critic_warmup_frac"),
    ("physics_slope_anneal_epochs", "physics_slope_anneal_frac"),
    ("penalty_warmup_epochs", "penalty_warmup_frac"),
    ("focal_gamma_warmup_epochs", "focal_gamma_warmup_frac"),
)

_CURRICULUM_FRAC_FIELDS = (
    ("cross_freq_start_epoch", "cross_freq_start_frac"),
    ("modality_dropout_protect_heatmap_epoch", "modality_dropout_protect_heatmap_frac"),
    ("heatmap_focus_start_epoch", "heatmap_focus_start_frac"),
    ("impedance_peak_start_epoch", "impedance_peak_start_frac"),
    ("impedance_peak_ramp_epochs", "impedance_peak_ramp_frac"),
    ("impedance_peak_focus_epoch", "impedance_peak_focus_frac"),
)


@dataclass
class Config:
    latent_dim: int = 42
    heatmap_private_dim: int = 8
    cond_dim: int = 8

    num_epochs: int = 850
    batch_size: int = 96   # reduce to 64 if CUDA OOM
    learning_rate: float = 2e-5
    lr_min: float = 3e-6
    lr_patience: int = 25
    lr_factor: float = 0.5
    train_split: float = 0.9
    num_workers: int = 8
    reset_lr_on_resume: bool = True

    balance_k: bool = True
    k_balance_power: float = 0.5
    k_balance_smoothing: float = 1e-3
    stratify_by_k: bool = True

    # Multifreq PI heatmap (1–600 MHz via PI_freq conditioning on heatmap only)
    split_by_design: bool = True
    balance_freq: bool = True
    freq_balance_power: float = 1.0
    train_samples_per_epoch: int = 50_000  # random train draws/epoch (full pool on disk)
    val_on_checkpoint_only: bool = True  # skip full val between checkpoint epochs (LR uses last val)

    heatmap_weight: float = 2.75
    cross_freq_weight: float = 1.0
    cross_freq_start_frac: float = 0.05
    cross_freq_start_epoch: int = 0
    modality_dropout_protect_heatmap_frac: float = 0.5
    modality_dropout_protect_heatmap_epoch: int = 0
    freq_fourier_features: int = 8
    use_heatmap_film: bool = True
    # Decode-time PI_freq jitter (native path): decode at perturbed MHz, target stays true map.
    freq_jitter_prob: float = 0.3
    freq_jitter_log10_sigma: float = 0.08
    # Physical-space FG percentile amplitude (denormed Ω).
    heatmap_phys_p99_weight: float = 0.5
    heatmap_phys_p99_percentile: float = 99.0
    # Heatmap + freq fine-tune phase (second half of training by default).
    heatmap_focus_start_frac: float = 0.5
    heatmap_focus_start_epoch: int = 0
    heatmap_focus_heatmap_weight: float = 3.5
    heatmap_focus_impedance_weight: float = 1.5
    heatmap_focus_cross_freq_weight: float = 1.0
    heatmap_focus_dynrange_weight: float = 3.0
    heatmap_focus_modality_dropout: float = 0.05
    # Match inference: train with occ+imp layout latent part of the time.
    layout_train_prob: float = 0.4
    # Cross-freq loss always uses layout z (matches sweep / layout_cross eval).
    cross_freq_layout_z_only: bool = False
    # Log cross-freq / layout metrics at these MHz during checkpoint val.
    eval_off_anchor_mhz: tuple[float, ...] = (80.0, 250.0)
    eval_off_anchor_max_batches: int = 30
    occupancy_weight: float = 8.0   # val occ ~0.35 still dominates recon
    impedance_weight: float = 3.0
    impedance_deriv_weight: float = 1.5
    impedance_topk_k: int = 20
    impedance_topk_weight: float = 7.0  # peaks: prefer topk_weight over raising impedance_weight
    impedance_under_penalty: float = 2.8
    impedance_concavity_weight: float = 2.5
    impedance_freq_weight_alpha: float = 2.0
    impedance_dual_topk_weight: float = 0.75
    impedance_peak_index_weight: float = 2.5
    impedance_peak_mag_weight: float = 2.0
    impedance_num_peaks: int = 8
    impedance_peak_start_frac: float = 0.5
    impedance_peak_ramp_frac: float = 0.11
    impedance_peak_focus_frac: float = 0.5
    impedance_peak_start_epoch: int = 0
    impedance_peak_ramp_epochs: int = 0
    impedance_peak_focus_epoch: int = 0
    impedance_decoder_lr_mult: float = 4.0
    heatmap_peak_weight: float = 3.0
    heatmap_grad_weight: float = 1.5
    heatmap_lap_weight: float = 2.0
    heatmap_contrast_weight: float = 1.5
    heatmap_contrast_margin: float = 0.5
    heatmap_bg_weight: float = 0.5
    heatmap_dynrange_weight: float = 2.0
    occupancy_focal_gamma: float = 1.5
    focal_gamma_warmup_frac: float = 0.0
    focal_gamma_warmup_epochs: int = 0
    occ_k_consistency_weight: float = 2.0
    use_k_weighting: bool = True
    occ_mid_k_center: float = 25.0
    occ_mid_k_sigma: float = 8.0
    occ_mid_k_boost: float = 0.75
    hm_low_k_threshold: int = 3
    hm_low_k_multiplier: float = 2.0

    free_bits: float = 0.05
    mu_hinge_threshold: float = 4.0
    mu_hinge_weight: float = 0.10
    mu_bias_weight: float = 0.05
    per_expert_kl_weight: float = 0.02
    sigma_reg_weight: float = 1.5
    sigma_reg_target: float = 0.45

    use_beta_annealing: bool = True
    beta_start_epoch: int = 0
    beta_end_frac: float = 0.40
    beta_initial: float = 0.0
    beta_final: float = 0.1
    beta_phase2_final: float = 0.15  # slightly lower KL late — frees recon (β=0.2 was not helping val)
    beta_phase2_end_frac: float = 0.80
    beta_end_epoch: int = 0
    beta_phase2_end_epoch: int = 0

    modality_dropout: float = 0.12
    modality_dropout_start: float = 0.08
    modality_dropout_anneal_frac: float = 0.08
    modality_dropout_anneal_epochs: int = 0

    cross_modal_weight: float = 0.85
    cross_modal_update_freq: int = 4

    physics_ri_weight: float = 1.0
    physics_critic_sup_weight: float = 2.0
    physics_ar_weight: float = 0.5
    physics_fg_clip_min: float = -1.04
    physics_critic_warmup_frac: float = 0.05
    physics_slope_anneal_frac: float = 0.10
    physics_critic_warmup_epochs: int = 0
    physics_slope_anneal_epochs: int = 0

    penalty_warmup_frac: float = 0.05
    penalty_warmup_epochs: int = 0

    # Allow starting new experiments without copying this code.
    # Use env vars to override where outputs are written/read.
    data_dir: str = field(
        default_factory=lambda: os.environ.get("VAE_DATA_DIR", "/home/ubuntu/gan/datasets/data_multifreq_norm"),
    )
    experiment_dir: str = field(
        default_factory=lambda: os.environ.get("VAE_EXPERIMENT_DIR", "/home/ubuntu/gan/experiments/exp038_true_multi"),
    )
    checkpoint_interval: int = 25   # CSV, plots, latent stats, and checkpoint_epoch_{N}.pt
    keep_last_n_checkpoints: int = 0  # 0 = keep every interval checkpoint (no pruning)
    resume_checkpoint: int | str | None = None  # None = fresh; "latest" / "last" / epoch int
    background_value: float = -3.6228

    amp: str = "bf16"   # Ampere+ (CC≥8); auto-downgraded to fp16 on Volta/Turing (e.g. GV100)
    tf32: bool = True   # Ampere+ only; disabled automatically on older GPUs
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    cache_in_ram: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    kan_spline_l1_weight: float = 1e-4
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        n = self.num_epochs
        for attr, frac in _FRAC_FIELDS + _CURRICULUM_FRAC_FIELDS:
            setattr(self, attr, round(n * getattr(self, frac)))
    @property
    def log_interval(self) -> int:
        """Alias for checkpoint_interval (external tools / config.yaml)."""
        return self.checkpoint_interval

    def is_cuda(self) -> bool:
        return "cuda" in self.device


# ── Schedules ─────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def compute_beta(epoch: int, c: Config) -> float:
    if not c.use_beta_annealing:
        return c.beta_final
    if epoch < c.beta_start_epoch:
        return c.beta_initial
    if epoch < c.beta_end_epoch:
        t = (epoch - c.beta_start_epoch) / max(c.beta_end_epoch - c.beta_start_epoch, 1)
        return _lerp(c.beta_initial, c.beta_final, t)
    if epoch < c.beta_phase2_end_epoch:
        t = (epoch - c.beta_end_epoch) / max(c.beta_phase2_end_epoch - c.beta_end_epoch, 1)
        return _lerp(c.beta_final, c.beta_phase2_final, t)
    return c.beta_phase2_final


def compute_modality_dropout(epoch: int, c: Config) -> float:
    if epoch >= c.heatmap_focus_start_epoch:
        return c.heatmap_focus_modality_dropout
    if epoch >= c.modality_dropout_anneal_epochs:
        return c.modality_dropout
    t = epoch / max(c.modality_dropout_anneal_epochs, 1)
    return _lerp(c.modality_dropout_start, c.modality_dropout, t)


def _phase_weights(epoch: int, c: Config) -> dict[str, float]:
    """Per-epoch loss weights; heatmap_focus phase boosts heatmap / cross-freq."""
    w = {
        "heatmap_weight": c.heatmap_weight,
        "impedance_weight": c.impedance_weight,
        "cross_freq_weight": c.cross_freq_weight,
        "heatmap_dynrange_weight": c.heatmap_dynrange_weight,
    }
    if epoch >= c.heatmap_focus_start_epoch:
        w["heatmap_weight"] = c.heatmap_focus_heatmap_weight
        w["impedance_weight"] = c.heatmap_focus_impedance_weight
        w["cross_freq_weight"] = c.heatmap_focus_cross_freq_weight
        w["heatmap_dynrange_weight"] = c.heatmap_focus_dynrange_weight
    return w


def _jitter_pi_freq_norm(pi: torch.Tensor, c: Config, *, train: bool) -> torch.Tensor:
    """Log10-jitter in norm space; decode-only (target heatmap unchanged)."""
    import math
    from src_vae.others.pi_freq_utils import _LOG10_MIN, _LOG10_RANGE

    if not train or c.freq_jitter_prob <= 0.0:
        return pi
    if torch.rand((), device=pi.device) >= c.freq_jitter_prob:
        return pi
    log10_max = math.log10(600e6)
    log10 = pi * _LOG10_RANGE + _LOG10_MIN
    log10_j = (log10 + torch.randn_like(pi) * c.freq_jitter_log10_sigma).clamp(
        _LOG10_MIN, log10_max,
    )
    return (log10_j - _LOG10_MIN) / _LOG10_RANGE


def compute_focal_gamma(epoch: int, c: Config) -> float:
    if c.focal_gamma_warmup_epochs <= 0:
        return c.occupancy_focal_gamma
    if epoch >= c.focal_gamma_warmup_epochs:
        return c.occupancy_focal_gamma
    return c.occupancy_focal_gamma * epoch / max(c.focal_gamma_warmup_epochs, 1)


def _physics_weights(epoch: int, c: Config) -> tuple[float, float, float]:
    if epoch < c.physics_critic_warmup_epochs:
        return 0.0, c.physics_critic_sup_weight, 0.0
    t = min(1.0, (epoch - c.physics_critic_warmup_epochs) / max(c.physics_slope_anneal_epochs, 1))
    return c.physics_ri_weight * t, c.physics_critic_sup_weight, c.physics_ar_weight * t


def _penalty_scale(epoch: int, c: Config) -> float:
    return 1.0 if c.penalty_warmup_epochs <= 0 else min(1.0, epoch / max(c.penalty_warmup_epochs, 1))


# ── Losses ────────────────────────────────────────────────────────────────────

LOSS_KEYS = (
    "total_loss", "recon_loss", "kl_loss", "kl_gaussian", "heatmap_loss", "occupancy_loss", "impedance_loss",
    "impedance_legacy_loss", "impedance_peak_loss", "cross_freq_heatmap_loss", "heatmap_phys_p99_loss",
    "occ_k_consistency_loss", "expert_kl_loss", "sigma_reg_loss",
    "physics_ri_loss", "physics_critic_sup_loss", "physics_ar_loss", "spline_l1_loss",
)

_LAP = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32)
_LAP_CACHE: dict[tuple, torch.Tensor] = {}


def _lap_k(device, dtype) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _LAP_CACHE:
        _LAP_CACHE[key] = _LAP.to(device=device, dtype=dtype)
    return _LAP_CACHE[key]


def _imp_ch0(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] if x.dim() == 3 else x


def focal_bce(pred: torch.Tensor, target: torch.Tensor, gamma: float,
              pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        pred, target.clamp(0, 1), pos_weight=pos_weight, reduction="none",
    )
    if gamma == 0.0:
        return bce.mean(1)
    return ((1.0 - torch.exp(-bce)).pow(gamma) * bce).mean(1)


def heatmap_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: Config,
    ps: float,
    *,
    dynrange_weight: float | None = None,
) -> torch.Tensor:
    bg = c.background_value + 0.5
    fg, bg_m = (target > bg).float(), (target <= bg).float()
    huber = F.huber_loss(recon, target, delta=1.0, reduction="none")
    n_fg = fg.sum((1, 2, 3)).clamp(min=1.0)
    base = (huber * fg).sum((1, 2, 3)) / n_fg
    intens = (target - bg).clamp(min=0)
    peak = (huber * fg * intens).sum((1, 2, 3)) / (intens * fg).sum((1, 2, 3)).clamp(min=1e-6)
    fg_dx, fg_dy = fg[:, :, :, 1:] * fg[:, :, :, :-1], fg[:, :, 1:, :] * fg[:, :, :-1, :]
    gx = F.huber_loss(recon[:, :, :, 1:] - recon[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1],
                      delta=0.5, reduction="none")
    gy = F.huber_loss(recon[:, :, 1:, :] - recon[:, :, :-1, :], target[:, :, 1:, :] - target[:, :, :-1, :],
                      delta=0.5, reduction="none")
    grad = (gx * fg_dx).sum((1, 2, 3)) / fg_dx.sum((1, 2, 3)).clamp(min=1.0)
    grad += (gy * fg_dy).sum((1, 2, 3)) / fg_dy.sum((1, 2, 3)).clamp(min=1.0)
    lap_k = _lap_k(recon.device, recon.dtype)
    lap = ((F.conv2d(recon, lap_k, padding=1) - F.conv2d(target, lap_k, padding=1)).pow(2) * fg).sum((1, 2, 3)) / n_fg
    n_bg = bg_m.sum((1, 2, 3)).clamp(min=1.0)
    contrast = F.relu((recon * bg_m).sum((1, 2, 3)) / n_bg + c.heatmap_contrast_margin - (recon * fg).sum((1, 2, 3)) / n_fg)
    bg_h = (huber * bg_m).sum((1, 2, 3)) / n_bg
    flat_r = (recon * fg + (1 - fg) * recon.min()).flatten(1)
    flat_t = (target * fg + (1 - fg) * target.min()).flatten(1)
    max_r, max_t = flat_r.max(1).values, flat_t.max(1).values
    dyn_max = F.relu(max_t - max_r).pow(2)
    p95_r = torch.quantile(flat_r, 0.95, dim=1)
    p95_t = torch.quantile(flat_t, 0.95, dim=1)
    dyn_p95 = F.relu(p95_t - p95_r).pow(2)
    dyn = dyn_max + 0.5 * dyn_p95
    dw = c.heatmap_dynrange_weight if dynrange_weight is None else dynrange_weight
    return (base + c.heatmap_peak_weight * peak + c.heatmap_grad_weight * grad
            + ps * (c.heatmap_lap_weight * lap + c.heatmap_contrast_weight * contrast
                    + c.heatmap_bg_weight * bg_h + dw * dyn))


def heatmap_phys_amplitude_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    hm_log_mean: float,
    hm_log_std: float,
    c: Config,
) -> torch.Tensor:
    """Foreground p99 in physical Ω — penalize under-prediction of peak level."""
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    recon_p = torch.exp(recon * hm_log_std + hm_log_mean).clamp(min=0.0) - 1e-6
    tgt_p = torch.exp(target * hm_log_std + hm_log_mean).clamp(min=0.0) - 1e-6
    fill_r = recon_p.amin(dim=(2, 3), keepdim=True)
    fill_t = tgt_p.amin(dim=(2, 3), keepdim=True)
    flat_r = (recon_p * fg + (1.0 - fg) * fill_r).flatten(1)
    flat_t = (tgt_p * fg + (1.0 - fg) * fill_t).flatten(1)
    q = float(c.heatmap_phys_p99_percentile) / 100.0
    p99_r = torch.quantile(flat_r, q, dim=1)
    p99_t = torch.quantile(flat_t, q, dim=1)
    return F.relu(p99_t - p99_r).pow(2).mean()


def _impedance_weights(c: Config) -> ImpedanceSpectrumWeights:
    return ImpedanceSpectrumWeights(
        deriv_weight=c.impedance_deriv_weight,
        concavity_weight=c.impedance_concavity_weight,
        topk_k=c.impedance_topk_k,
        topk_weight=c.impedance_topk_weight,
        under_penalty=c.impedance_under_penalty,
        freq_weight_alpha=c.impedance_freq_weight_alpha,
        dual_topk_weight=c.impedance_dual_topk_weight,
        peak_index_weight=c.impedance_peak_index_weight,
        peak_mag_weight=c.impedance_peak_mag_weight,
        num_peaks=c.impedance_num_peaks,
    )


def _peak_loss_scale(epoch: int, c: Config) -> float:
    """Ramp peak/dual-topk terms in after resume (0 before start_epoch)."""
    start = c.impedance_peak_start_epoch
    if start < 0 or epoch < start:
        return 0.0
    ramp = max(c.impedance_peak_ramp_epochs, 1)
    return min(1.0, (epoch - start + 1) / ramp)


def impedance_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: Config,
    imp_log_std: float,
    ps: float,
    *,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    peak_scale = _peak_loss_scale(epoch, c)
    return impedance_spectrum_loss(
        recon,
        target,
        imp_log_std=imp_log_std,
        penalty_scale=ps,
        peak_scale=peak_scale,
        w=_impedance_weights(c),
    )


def _k_weights(K: torch.Tensor, c: Config) -> tuple[torch.Tensor, torch.Tensor]:
    kf = K.float()
    bump = torch.exp(-0.5 * ((kf - c.occ_mid_k_center) / c.occ_mid_k_sigma) ** 2)
    w_occ = 1.0 + c.occ_mid_k_boost * bump
    w_hm = torch.where(K <= c.hm_low_k_threshold, torch.full_like(kf, c.hm_low_k_multiplier), torch.ones_like(kf))
    return w_hm, w_occ


def _wmean(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w.to(v.dtype)
    return (v * w).sum() / w.sum().clamp(min=1e-6)


def vae_loss(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp,
             mu, logvar, beta: float, c: Config, expert_stats, *, epoch: int,
             K: torch.Tensor | None, physics: PhysicsLoss | None,
             pw: tuple[float, float, float] | None, pi_freq: torch.Tensor | None,
             imp_log_std: float, ps: float, apply_k: bool,
             weight_overrides: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
    wo = weight_overrides or {}
    hw = wo.get("heatmap_weight", c.heatmap_weight)
    iw = wo.get("impedance_weight", c.impedance_weight)
    hdr = wo.get("heatmap_dynrange_weight", c.heatmap_dynrange_weight)
    hm_ps = heatmap_loss(recon_hm, tgt_hm, c, ps, dynrange_weight=hdr)
    pos_w = None
    if K is not None:
        pos_w = ((52.0 - K.float()) / K.float().clamp(min=1)).clamp(0.5, 10.0).unsqueeze(1).expand_as(recon_occ)
    occ_ps = focal_bce(recon_occ, tgt_occ, compute_focal_gamma(epoch, c), pos_w)
    imp_ps, imp_parts = impedance_loss(recon_imp, tgt_imp, c, imp_log_std, ps, epoch=epoch)
    if apply_k and c.use_k_weighting and K is not None:
        wh, wo = _k_weights(K, c)
        loss_hm, loss_occ = _wmean(hm_ps, wh), _wmean(occ_ps, wo)
    else:
        loss_hm, loss_occ = hm_ps.mean(), occ_ps.mean()
    loss_imp = imp_ps.mean()
    loss_imp_legacy = imp_parts["legacy"].mean()
    loss_imp_peak = imp_parts["peak_extra"].mean()

    occ_k = mu.new_zeros(())
    if c.occ_k_consistency_weight > 0 and K is not None:
        t = tgt_occ.clamp(0, 1)
        occ_k = (F.relu(0.5 - recon_occ) * t + F.relu(recon_occ + 0.5) * (1 - t)).mean()

    recon = (hw * loss_hm + c.occupancy_weight * loss_occ
             + iw * loss_imp + c.occ_k_consistency_weight * occ_k)
    kl_pd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if c.free_bits > 0:
        kl_pd = kl_pd.clamp(min=c.free_bits)
    kl = kl_pd.mean()
    total = recon + beta * kl + c.mu_hinge_weight * F.relu(mu.abs() - c.mu_hinge_threshold).pow(2).mean()
    total = total + c.mu_bias_weight * mu.mean(0).pow(2).mean()

    exp_kl = sigma_reg = mu.new_zeros(())
    if expert_stats and c.per_expert_kl_weight > 0:
        exp_kl = sum((-0.5 * (1.0 + lv - m.pow(2) - lv.exp())).mean()
                     for m, lv in expert_stats.values()) / len(expert_stats)
        total = total + c.per_expert_kl_weight * exp_kl
    if c.sigma_reg_weight > 0 and expert_stats:
        prec = torch.stack([1.0 / (lv.exp() + 1e-8) for _, lv in expert_stats.values()]).sum(0) + 1.0
        sig = (1.0 / (prec + 1e-8)).sqrt()
        d_lo = (c.sigma_reg_target - sig).clamp(min=0)
        d_hi = (sig - c.sigma_reg_target * 1.3).clamp(min=0)
        sigma_reg = (d_lo.pow(2) * (c.sigma_reg_target / sig.detach().clamp(min=0.05)) + d_hi.pow(2)).mean()
        total = total + c.sigma_reg_weight * sigma_reg

    ri = cs = ar = mu.new_zeros(())
    if physics is not None:
        wri, wcs, war = pw or (c.physics_ri_weight, c.physics_critic_sup_weight, c.physics_ar_weight)
        ri, cs, ar = physics(recon_hm, recon_occ, recon_imp,
                             target_occ=tgt_occ, target_hm=tgt_hm, pi_freq=pi_freq)
        total = total + wri * ri + wcs * cs + war * ar

    return {
        "total_loss": total, "recon_loss": recon, "kl_loss": kl, "kl_gaussian": kl,
        "heatmap_loss": loss_hm, "occupancy_loss": loss_occ, "impedance_loss": loss_imp,
        "impedance_legacy_loss": loss_imp_legacy, "impedance_peak_loss": loss_imp_peak,
        "cross_freq_heatmap_loss": mu.new_zeros(()),
        "heatmap_phys_p99_loss": mu.new_zeros(()),
        "occ_k_consistency_loss": occ_k, "expert_kl_loss": exp_kl, "sigma_reg_loss": sigma_reg,
        "physics_ri_loss": ri, "physics_critic_sup_loss": cs, "physics_ar_loss": ar,
        "spline_l1_loss": mu.new_zeros(()),
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _gpu_supports_native_bf16(device_index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device_index)[0] >= 8


def _adapt_amp_for_gpu(c: Config) -> None:
    """Match amp/tf32 to GPU so torch.compile + autocast avoid Volta bf16 warnings."""
    if not c.is_cuda():
        return
    try:
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception:
        return
    if major < 8:
        if c.amp in ("bf16", "bfloat16", "auto"):
            print(f"GPU {name} (compute {major}.{minor}): amp={c.amp!r} → fp16 (no native bf16)")
            c.amp = "fp16"
        if c.tf32:
            print(f"GPU {name}: tf32 disabled (requires compute capability ≥ 8.0)")
            c.tf32 = False
        if c.compile:
            print(f"GPU {name}: torch.compile disabled (Volta/Turing: graph breaks / recompiles)")
            c.compile = False


def _amp_dtype(c: Config) -> torch.dtype | None:
    if not c.is_cuda() or c.amp == "off":
        return None
    if c.amp in ("bf16", "bfloat16"):
        return torch.bfloat16 if _gpu_supports_native_bf16() else torch.float16
    if c.amp == "fp16":
        return torch.float16
    return torch.bfloat16 if _gpu_supports_native_bf16() else torch.float16


def _cross_freq_heatmap_loss(
    model, z: torch.Tensor, K: torch.Tensor, pi_alt: torch.Tensor,
    hm_alt: torch.Tensor, c: Config, ps: float,
    *, dynrange_weight: float | None = None,
) -> torch.Tensor:
    """Decode same z at alternate PI_freq; target is ground-truth heatmap at that MHz."""
    base = getattr(model, "_orig_mod", model)
    dev = z.device
    hm_alt = hm_alt.to(dev, non_blocking=c.is_cuda())
    if hm_alt.dim() == 3:
        hm_alt = hm_alt.unsqueeze(1)
    rh, _, _ = base.decode(z, K, pi_alt)
    return heatmap_loss(rh, hm_alt, c, ps, dynrange_weight=dynrange_weight).mean()


def _forward_train_batch(
    model,
    hm_enc: torch.Tensor,
    occ: torch.Tensor,
    imp: torch.Tensor,
    K: torch.Tensor,
    pi: torch.Tensor,
    c: Config,
    *,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, torch.Tensor]:
    """Encode for KL; decode from layout or posterior z with optional PI_freq jitter."""
    base = getattr(model, "_orig_mod", model)
    z_post, mu, logvar, expert_stats = base.encode(hm_enc, occ, imp, K, pi)
    use_layout = (
        train
        and c.layout_train_prob > 0.0
        and float(torch.rand(()).item()) < c.layout_train_prob
    )
    if use_layout:
        z = base.encode_layout_latent(occ, imp, K, pi)
    else:
        z = z_post
    pi_dec = _jitter_pi_freq_norm(pi, c, train=train)
    rh, ro, ri = base.decode(z, K, pi_dec)
    return rh, ro, ri, mu, logvar, expert_stats, z


def _prepare_batch(batch: dict, c: Config) -> tuple:
    nb = c.is_cuda()
    dev = c.device
    hm = batch["heatmap_norm"].to(dev, non_blocking=nb)
    occ = batch["occupancy"].to(dev, non_blocking=nb)
    imp = batch["impedance"].to(dev, non_blocking=nb)
    K = batch["K"].to(dev, non_blocking=nb)
    if "PI_freq" in batch:
        pi = batch["PI_freq"].to(dev, non_blocking=nb)
    else:
        from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm
        pi = torch.full((hm.shape[0],), pi_freq_mhz_to_norm(200.0), device=dev)
    if hm.dim() == 3:
        hm = hm.unsqueeze(1)
    if imp.dim() == 1:
        imp = imp.unsqueeze(0)
    if imp.dim() == 2 and imp.shape[-1] == 231:
        imp = imp.unsqueeze(1)
    elif imp.dim() == 3:
        imp = imp[:, :1]
    return hm, hm.masked_fill(hm < c.background_value + 0.5, 0.0), occ, imp, K, pi


def _cross_modal(model, hm_enc, occ, imp, hm, K, pi, c: Config, imp_std: float, ps: float) -> torch.Tensor:
    wh, wo = _k_weights(K, c) if c.use_k_weighting else (None, None)
    total = hm.new_zeros(())
    for src in ("heatmap", "impedance"):
        z = model.encode_cross_modal(src, K, pi, heatmap=hm_enc, impedance=imp)
        rh, ro, ri = model.decode(z, K, pi)
        lh = heatmap_loss(rh, hm, c, ps)
        lo = focal_bce(ro, occ, c.occupancy_focal_gamma)
        li, _ = impedance_loss(ri, imp, c, imp_std, ps, epoch=0)
        total = total + c.heatmap_weight * (_wmean(lh, wh) if wh is not None else lh.mean())
        total = total + c.occupancy_weight * (_wmean(lo, wo) if wo is not None else lo.mean())
        total = total + c.impedance_weight * li.mean()
    return total / 2.0


def _clean_previous_run(exp: Path) -> None:
    """Remove prior training artifacts when starting fresh (not resuming)."""
    removed: list[str] = []
    for name in ("checkpoints", "logs", "metrics"):
        p = exp / name
        if p.exists():
            shutil.rmtree(p)
            removed.append(name + "/")
    for name in ("config.yaml", "watchdog_report.json"):
        p = exp / name
        if p.is_file():
            p.unlink()
            removed.append(name)
    for p in sorted(exp.glob("visuals_*")):
        if p.is_dir():
            shutil.rmtree(p)
            removed.append(p.name + "/")
    if removed:
        print("Fresh run — removed previous outputs under", exp)
        for r in removed:
            print(f"  - {r}")
    else:
        print(f"Fresh run — no prior training artifacts in {exp}")


def _apply_yaml_config(c: Config) -> None:
    """Merge experiment config.yaml (extension / resume knobs)."""
    p = Path(c.experiment_dir) / "config.yaml"
    if not p.is_file():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    fields = {f.name for f in Config.__dataclass_fields__.values()}
    for key, val in data.items():
        if key not in fields:
            continue
        if key == "eval_off_anchor_mhz" and isinstance(val, list):
            setattr(c, key, tuple(float(x) for x in val))
        else:
            setattr(c, key, val)
    c.__post_init__()
    print(f"Loaded config overrides from {p.name}")
    print(
        f"  Curriculum (of {c.num_epochs} epochs): "
        f"cross_freq@{c.cross_freq_start_epoch}, "
        f"protect_hm@{c.modality_dropout_protect_heatmap_epoch}, "
        f"heatmap_focus@{c.heatmap_focus_start_epoch}, "
        f"imp_peak@{c.impedance_peak_start_epoch}→{c.impedance_peak_focus_epoch} "
        f"(ramp {c.impedance_peak_ramp_epochs}), "
        f"layout_train_prob={c.layout_train_prob}, "
        f"cross_freq_layout_z_only={c.cross_freq_layout_z_only}",
    )


def _impedance_module_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    prefixes = ("impedance_encoder", "impedance_mu", "impedance_logvar", "impedance_decoder")
    return [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(n.startswith(px) for px in prefixes)
    ]


def _build_optimizer(model: torch.nn.Module, physics: PhysicsLoss | None, c: Config, epoch_start: int) -> optim.AdamW:
    """AdamW with optional higher LR on impedance decoder/encoder during peak-focus phase."""
    base = c.learning_rate
    use_boost = (
        c.impedance_decoder_lr_mult > 1.0
        and epoch_start >= c.impedance_peak_focus_epoch
    )
    if not use_boost:
        params = list(model.parameters()) + (list(physics.parameters()) if physics else [])
        return optim.AdamW(params, lr=base, weight_decay=1e-4, foreach=True)

    imp_params = _impedance_module_params(model)
    imp_ids = {id(p) for p in imp_params}
    other_params = [p for _, p in model.named_parameters() if p.requires_grad and id(p) not in imp_ids]
    groups = [
        {"params": other_params, "lr": base},
        {"params": imp_params, "lr": base * c.impedance_decoder_lr_mult},
    ]
    if physics:
        groups.append({"params": list(physics.parameters()), "lr": base})
    print(
        f"  Peak-focus optimizer: base lr={base:.2e}, "
        f"impedance lr={base * c.impedance_decoder_lr_mult:.2e} "
        f"(epoch >= {c.impedance_peak_focus_epoch})",
    )
    return optim.AdamW(groups, weight_decay=1e-4, foreach=True)


def _append_impedance_split_csv(metrics_dir: Path, epoch: int, tr: dict, val: dict | None) -> None:
    path = metrics_dir / "impedance_split.csv"
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "epoch", "train_impedance_loss", "train_impedance_legacy", "train_impedance_peak",
                "val_impedance_loss", "val_impedance_legacy", "val_impedance_peak",
            ])
        w.writerow([
            epoch,
            tr.get("impedance_loss", ""),
            tr.get("impedance_legacy_loss", ""),
            tr.get("impedance_peak_loss", ""),
            val.get("impedance_loss", "") if val else "",
            val.get("impedance_legacy_loss", "") if val else "",
            val.get("impedance_peak_loss", "") if val else "",
        ])


def _resolve_resume(c: Config) -> None:
    rc = c.resume_checkpoint
    if rc is None:
        return
    d = Path(c.experiment_dir) / "checkpoints"
    if isinstance(rc, int):
        c.resume_checkpoint = str(d / f"checkpoint_epoch_{rc}.pt")
    elif isinstance(rc, str) and rc.lower() in ("last", "last_model"):
        c.resume_checkpoint = str(d / "last_model.pt")
    elif isinstance(rc, str) and rc.lower() == "latest":
        interval = sorted(
            d.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if interval:
            c.resume_checkpoint = str(interval[-1])
        elif (d / "last_model.pt").exists():
            c.resume_checkpoint = str(d / "last_model.pt")
        else:
            raise FileNotFoundError(f"No checkpoints in {d}")


def load_checkpoint(path: str, model, optimizer=None, device="cuda", physics=None) -> tuple[int, float]:
    print(f"Resume: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    msd = model.state_dict()
    compat = {k: v for k, v in ckpt["model_state_dict"].items() if k in msd and v.shape == msd[k].shape}
    if not compat and os.getenv("VAE_ALLOW_PARTIAL_CHECKPOINT") not in {"1", "true", "True"}:
        raise RuntimeError("Checkpoint incompatible — set VAE_ALLOW_PARTIAL_CHECKPOINT=1 to migrate")
    msd.update(compat)
    model.load_state_dict(msd)
    if optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError:
            print("  optimizer state skipped (param groups changed)")
    if physics and "physics_state_dict" in ckpt:
        physics.load_state_dict(ckpt["physics_state_dict"], strict=False)
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


def _ckpt_dict(epoch, model, opt, tr_loss, val, c, latent_stats, per_k, physics, sched) -> dict:
    d = {
        "epoch": epoch, "train_loss": tr_loss, "val_loss": val["total_loss"],
        "val_metrics": {k: float(val[k]) for k in LOSS_KEYS if k in val},
        "config": asdict(c), "latent_stats": latent_stats, "per_K_latent_stats": per_k,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
        "mu_mean": val["mu_mean"], "mu_std": val["mu_std"], "mu_min": val["mu_min"], "mu_max": val["mu_max"],
    }
    if physics:
        d["physics_state_dict"] = physics.state_dict()
    if sched:
        d["scheduler_state_dict"] = sched.state_dict()
    return d


# ── Train / validate ──────────────────────────────────────────────────────────

def _run_epoch(
    model, loader, c: Config, epoch: int, beta: float, md: float,
    physics: PhysicsLoss | None, pw: tuple[float, float, float] | None,
    imp_log_std: float, hm_log_mean: float, hm_log_std: float,
    *, train: bool, optimizer=None, scaler=None,
    collect_per_k: bool = False,
) -> dict[str, Any]:
    model.train(train)
    if physics:
        physics.train(train)
    model.modality_dropout = md if train else 0.0
    base = getattr(model, "_orig_mod", model)
    base.modality_dropout_protect_heatmap = (
        train and epoch >= c.modality_dropout_protect_heatmap_epoch
    )

    acc = {k: 0.0 for k in LOSS_KEYS}
    n_batches = 0
    mu_sum = mu_sq = sig_sum = None
    n_mu = 0
    mod_acc: dict[str, list[float]] = {}
    k_buckets: dict[int, dict] = {}

    amp = _amp_dtype(c)
    autocast = torch.autocast(device_type="cuda" if c.is_cuda() else "cpu", dtype=amp, enabled=amp is not None)
    cm_every = max(1, c.cross_modal_update_freq)
    ps = _penalty_scale(epoch, c)
    pwts = _phase_weights(epoch, c)
    params = list(model.parameters()) + (list(physics.parameters()) if physics else [])

    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for i, batch in enumerate(loader):
            hm, hm_enc, occ, imp, K, pi = _prepare_batch(batch, c)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with autocast:
                if train:
                    rh, ro, ri, mu, lv, ex, z = _forward_train_batch(
                        model, hm_enc, occ, imp, K, pi, c, train=True,
                    )
                else:
                    rh, ro, ri, mu, lv, ex = model(hm_enc, occ, imp, K, pi)
                    z = mu  # val cross-freq uses posterior mean
                losses = vae_loss(
                    rh, ro, ri, hm, occ, imp, mu, lv, beta, c, ex,
                    epoch=epoch, K=K, physics=physics, pw=pw, pi_freq=pi,
                    imp_log_std=imp_log_std, ps=ps, apply_k=train,
                    weight_overrides=pwts,
                )
                if c.heatmap_phys_p99_weight > 0:
                    phys = heatmap_phys_amplitude_loss(rh, hm, hm_log_mean, hm_log_std, c)
                    losses["heatmap_phys_p99_loss"] = phys
                    losses["total_loss"] = losses["total_loss"] + c.heatmap_phys_p99_weight * phys
                if train and c.cross_modal_weight > 0 and i % cm_every == 0:
                    losses["total_loss"] = losses["total_loss"] + c.cross_modal_weight * _cross_modal(
                        model, hm_enc, occ, imp, hm, K, pi, c, imp_log_std, ps,
                    )
                cf_w = pwts["cross_freq_weight"]
                if (
                    train
                    and cf_w > 0
                    and epoch >= c.cross_freq_start_epoch
                    and "heatmap_norm_alt" in batch
                ):
                    pi_alt = batch["PI_freq_alt"].to(c.device, non_blocking=c.is_cuda())
                    hm_alt = batch["heatmap_norm_alt"].to(c.device, non_blocking=c.is_cuda())
                    base_cf = getattr(model, "_orig_mod", model)
                    if c.cross_freq_layout_z_only:
                        z_cf = base_cf.encode_layout_latent(occ, imp, K, pi)
                    else:
                        z_cf = z
                    cf = _cross_freq_heatmap_loss(
                        model, z_cf, K, pi_alt, hm_alt, c, ps,
                        dynrange_weight=pwts["heatmap_dynrange_weight"],
                    )
                    losses["cross_freq_heatmap_loss"] = cf
                    losses["total_loss"] = losses["total_loss"] + cf_w * cf
                if train and c.kan_spline_l1_weight > 0 and hasattr(model, "spline_l1"):
                    spl = getattr(model, "_orig_mod", model).spline_l1()
                    losses["total_loss"] = losses["total_loss"] + c.kan_spline_l1_weight * spl
                    losses["spline_l1_loss"] = spl

            if train:
                if scaler and scaler.is_enabled():
                    scaler.scale(losses["total_loss"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total_loss"].backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
            else:
                dim = mu.shape[1]
                if mu_sum is None:
                    mu_sum = torch.zeros(dim)
                    mu_sq = torch.zeros(dim)
                    sig_sum = torch.zeros(dim)
                mc, sc = mu.cpu(), (0.5 * lv).exp().cpu()
                mu_sum += mc.sum(0)
                mu_sq += mc.pow(2).sum(0)
                sig_sum += sc.sum(0)
                n_mu += mc.shape[0]
                for name, (em, elv) in ex.items():
                    if name not in mod_acc:
                        mod_acc[name] = [0.0, 0.0, 0.0, 0.0]
                    es = (elv / 2).exp()
                    mod_acc[name][0] += em.mean().item()
                    mod_acc[name][1] += em.std().item()
                    mod_acc[name][2] += es.mean().item()
                    mod_acc[name][3] += es.std().item()
                if collect_per_k:
                    for kv in K.cpu().unique().tolist():
                        kv = int(kv)
                        mask = (K.cpu() == kv)
                        bk = k_buckets.setdefault(kv, {"mu": torch.zeros(dim), "sq": torch.zeros(dim),
                                                       "sig": torch.zeros(dim), "n": 0})
                        m = mc[mask]
                        s = sc[mask]
                        bk["mu"] += m.sum(0)
                        bk["sq"] += m.pow(2).sum(0)
                        bk["sig"] += s.sum(0)
                        bk["n"] += mask.sum().item()

            for k in LOSS_KEYS:
                if k in losses:
                    acc[k] += losses[k].item()
            n_batches += 1

    out: dict[str, Any] = {k: v / n_batches for k, v in acc.items()}
    out["beta"] = beta
    if not train and n_mu > 0:
        mu_pd = mu_sum / n_mu
        std_pd = ((mu_sq / n_mu - mu_pd.pow(2)).clamp(min=0) + (sig_sum / n_mu).pow(2)).sqrt()
        out.update(mu_mean=float(mu_pd.mean()), mu_std=float(mu_pd.std()),
                   mu_min=float(mu_pd.min()), mu_max=float(mu_pd.max()),
                   logvar_mean=0.0, logvar_std=0.0, logvar_min=0.0, logvar_max=0.0,
                   std_mean=float(std_pd.mean()))
        out["per_dim_stats"] = {"latent": {"mu_mean_per_dim": mu_pd.tolist(), "agg_std_per_dim": std_pd.tolist()}}
        out["modality_stats"] = {
            n: {"mu_mean": a[0] / n_batches, "mu_std": a[1] / n_batches,
                "std_mean": a[2] / n_batches, "std_std": a[3] / n_batches}
            for n, a in mod_acc.items()
        }
        per_k = {}
        for kv, bk in k_buckets.items():
            if bk["n"] < 2:
                continue
            nk = bk["n"]
            mk = bk["mu"] / nk
            sk = ((bk["sq"] / nk - mk.pow(2)).clamp(min=0) + (bk["sig"] / nk).pow(2)).sqrt()
            per_k[str(kv)] = {"latent": {"mu_mean_per_dim": mk.tolist(), "agg_std_per_dim": sk.tolist()}}
        out["per_K_latent_stats"] = per_k
    else:
        out.setdefault("modality_stats", {})
        out.setdefault("per_K_latent_stats", {})
    return out


def _should_run_validation(ep: int, num_epochs: int, checkpoint_interval: int, *, val_on_checkpoint_only: bool) -> bool:
    """Full val on epoch 1, every checkpoint_interval, and final epoch."""
    if not val_on_checkpoint_only:
        return True
    if ep == 1 or ep >= num_epochs:
        return True
    return ep % checkpoint_interval == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def train_vae() -> None:
    c = Config()
    _apply_yaml_config(c)
    _adapt_amp_for_gpu(c)
    _resolve_resume(c)

    exp = Path(c.experiment_dir)
    ckpt_dir, log_dir, metrics_dir = exp / "checkpoints", exp / "logs", exp / "metrics"
    plots_dir = metrics_dir / "plots"

    if not c.resume_checkpoint:
        _clean_previous_run(exp)
    for p in (exp, ckpt_dir, log_dir, metrics_dir, plots_dir):
        p.mkdir(parents=True, exist_ok=True)

    logger = VAETrainingLogger(
        str(log_dir), str(ckpt_dir), str(metrics_dir / "loss.csv"),
        checkpoint_interval=c.checkpoint_interval,
    )
    logger.log_start(exp.name, {
        "device": c.device, "batch": c.batch_size, "epochs": c.num_epochs,
        "checkpoint_interval": c.checkpoint_interval,
        "split_by_design": c.split_by_design, "balance_freq": c.balance_freq,
        "train_samples_per_epoch": c.train_samples_per_epoch,
        "val_on_checkpoint_only": c.val_on_checkpoint_only,
        "amp": c.amp, "tf32": c.tf32, "compile": c.compile, "ram_cache": c.cache_in_ram,
    })

    stats_path = Path(c.data_dir) / "normalization_stats.json"
    imp_log_std = 1.0
    hm_log_mean = 0.0
    hm_log_std = 1.0
    if stats_path.exists():
        raw = json.loads(stats_path.read_text())
        c.background_value = raw["background_value"]
        c.physics_fg_clip_min = raw["Heatmap"]["clip_min"]
        imp_log_std = float(raw["Impedance"]["log_std"]) or 1.0
        hm_log_mean = float(raw["Heatmap"]["log_mean"])
        hm_log_std = float(raw["Heatmap"]["log_std"]) or 1.0

    pin = c.is_cuda()
    train_ld, val_ld = create_multifreq_data_loaders(
        data_dir=c.data_dir, batch_size=c.batch_size, num_workers=c.num_workers,
        train_split=c.train_split, seed=42, pin_memory=pin,
        persistent_workers=c.persistent_workers and c.num_workers > 0,
        prefetch_factor=c.prefetch_factor if c.num_workers > 0 else None,
        split_by_design=c.split_by_design,
        balance_k=c.balance_k, balance_freq=c.balance_freq,
        k_balance_power=c.k_balance_power,
        freq_balance_power=c.freq_balance_power,
        k_balance_smoothing=c.k_balance_smoothing,
        stratify_by_k=c.stratify_by_k,
        cache_in_ram=c.cache_in_ram,
        train_samples_per_epoch=c.train_samples_per_epoch,
        cross_freq_pairs=c.cross_freq_weight > 0,
    )

    if c.is_cuda():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = c.tf32
        torch.backends.cudnn.allow_tf32 = c.tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if c.tf32 else "highest")

    model = MultiInputVAE(
        latent_dim=c.latent_dim,
        cond_dim=c.cond_dim,
        heatmap_private_dim=c.heatmap_private_dim,
        modality_dropout=c.modality_dropout,
        freq_fourier_features=c.freq_fourier_features,
        use_heatmap_film=c.use_heatmap_film,
    ).to(c.device)
    if c.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=c.compile_mode)
            print(f"torch.compile enabled ({c.compile_mode})")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    physics = None
    if c.physics_ri_weight > 0 or c.physics_critic_sup_weight > 0:
        physics = PhysicsLoss(c.background_value, c.physics_fg_clip_min).to(c.device)

    start = 0
    resume = c.resume_checkpoint
    resume_path = resume if isinstance(resume, str) and Path(resume).exists() else None
    if resume_path:
        start, _ = load_checkpoint(resume_path, model, None, c.device, physics)

    opt = _build_optimizer(model, physics, c, epoch_start=start)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=c.lr_factor, patience=c.lr_patience, min_lr=c.lr_min,
    )

    if resume_path:
        tmp = torch.load(resume_path, map_location="cpu", weights_only=False)
        if c.reset_lr_on_resume:
            for i, pg in enumerate(opt.param_groups):
                if len(opt.param_groups) >= 2 and i == 1 and start >= c.impedance_peak_focus_epoch:
                    pg["lr"] = c.learning_rate * c.impedance_decoder_lr_mult
                else:
                    pg["lr"] = c.learning_rate
            print(f"  Reset LR on resume (epoch start={start})")
        elif "optimizer_state_dict" in tmp:
            try:
                opt.load_state_dict(tmp["optimizer_state_dict"])
            except ValueError:
                print("  optimizer state skipped (param groups changed)")
        if not c.reset_lr_on_resume and "scheduler_state_dict" in tmp:
            try:
                sched.load_state_dict(tmp["scheduler_state_dict"])
            except ValueError:
                print("  scheduler state skipped (optimizer groups changed)")

    timing_path = metrics_dir / "timing.json"
    prev_t = json.loads(timing_path.read_text()).get("training_time_seconds", 0) if timing_path.exists() else 0
    amp = _amp_dtype(c)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16))
    t0 = time.time()

    print(f"epochs {start + 1}–{c.num_epochs}  amp={c.amp}  tf32={c.tf32}  "
          f"compile={c.compile}  ram_cache={c.cache_in_ram}  batch={c.batch_size}")
    print(f"checkpoint_interval={c.checkpoint_interval}  keep_last_n={c.keep_last_n_checkpoints}")
    if c.val_on_checkpoint_only:
        print("  Val: epoch 1, every checkpoint interval, and final epoch only")
    print()
    on_disk = {int(p.stem.split("_")[-1]) for p in ckpt_dir.glob("checkpoint_epoch_*.pt")}
    expect = [e for e in range(c.checkpoint_interval, start + 1, c.checkpoint_interval)]
    missing = [e for e in expect if e not in on_disk]
    if missing:
        print(f"  Note: no checkpoint file for epoch(s) {missing} "
              f"(older runs may have pruned them). Logs still resume from epoch {start}.")
    val: dict[str, Any] = {}
    last_val_loss = float("inf")
    epoch_timing_path = metrics_dir / "epoch_timing.csv"
    if start == 0:
        with epoch_timing_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "train_sec", "val_sec", "total_sec", "val_ran", "train_loss", "val_loss"],
            )
    train_times: list[float] = []
    val_times: list[float] = []

    for epoch in range(start, c.num_epochs):
        t_ep0 = time.perf_counter()
        beta = compute_beta(epoch, c)
        md = compute_modality_dropout(epoch, c)
        pw = _physics_weights(epoch, c) if physics else None
        t_tr0 = time.perf_counter()
        if epoch + 1 == c.heatmap_focus_start_epoch + 1:
            print(
                f"  >> Heatmap focus phase (epoch {epoch + 1}+): "
                f"hm_w={c.heatmap_focus_heatmap_weight} imp_w={c.heatmap_focus_impedance_weight} "
                f"cf_w={c.heatmap_focus_cross_freq_weight}",
            )
        tr = _run_epoch(
            model, train_ld, c, epoch, beta, md, physics, pw, imp_log_std,
            hm_log_mean, hm_log_std,
            train=True, optimizer=opt, scaler=scaler,
        )
        t_train = time.perf_counter() - t_tr0
        ep = epoch + 1
        log_metrics = logger.at_checkpoint_epoch(ep) or ep == 1
        save_ckpt = logger.at_checkpoint_epoch(ep)
        run_val = _should_run_validation(
            ep, c.num_epochs, c.checkpoint_interval, val_on_checkpoint_only=c.val_on_checkpoint_only,
        )
        t_val = 0.0
        if run_val:
            t_v0 = time.perf_counter()
            val = _run_epoch(
                model, val_ld, c, epoch, beta, 0.0, physics, pw, imp_log_std,
                hm_log_mean, hm_log_std,
                train=False, collect_per_k=save_ckpt,
            )
            t_val = time.perf_counter() - t_v0
            last_val_loss = float(val["total_loss"])
            val_times.append(t_val)
        sched.step(last_val_loss)
        lr = opt.param_groups[0]["lr"]
        t_total = time.perf_counter() - t_ep0
        train_times.append(t_train)
        val_loss_str = f"{last_val_loss:.4f}" if run_val else ""
        with epoch_timing_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [ep, f"{t_train:.2f}", f"{t_val:.2f}", f"{t_total:.2f}", int(run_val),
                 f"{tr['total_loss']:.6f}", val_loss_str],
            )

        if log_metrics:
            tr_ld = {k: tr[k] for k in ("total_loss", "recon_loss", "kl_loss", "kl_gaussian",
                                        "heatmap_loss", "occupancy_loss", "impedance_loss",
                                        "impedance_legacy_loss", "impedance_peak_loss")}
            logger.log_dict(ep, loss_dict=tr_ld, val_loss_dict={k: val[k] for k in LOSS_KEYS} if run_val else None)
            _append_impedance_split_csv(metrics_dir, ep, tr, val if run_val else None)
            logger.log_latent_stats(log_dir / "latent_stats.csv", ep, beta, val)
            print(
                f"Ep {ep:>3}/{c.num_epochs}  "
                f"train={t_train:.1f}s  val={t_val:.1f}s  total={t_total:.1f}s  "
                f"loss train={tr['total_loss']:.4f}  val={val['total_loss']:.4f}  "
                f"β={beta:.3f}  lr={lr:.2e}",
            )
            if physics and ep >= c.physics_critic_warmup_epochs:
                print(f"  phys ri={val['physics_ri_loss']:.4f} cs={val['physics_critic_sup_loss']:.4f}"
                      f" ar={val['physics_ar_loss']:.4f}")
            if ep > c.impedance_peak_start_epoch:
                print(
                    f"  imp legacy={val.get('impedance_legacy_loss', 0):.4f}  "
                    f"peak={val.get('impedance_peak_loss', 0):.4f}  "
                    f"(scale={_peak_loss_scale(epoch, c):.2f})",
                )
        else:
            skip = "  (val skipped)" if not run_val else ""
            print(
                f"Ep {ep:>3}/{c.num_epochs}  train={t_train:.1f}s  total={t_total:.1f}s{skip}  "
                f"loss={tr['total_loss']:.4f}  lr={lr:.2e}",
            )

        if save_ckpt:
            lat = VAETrainingLogger.build_latent_stats(val)
            per_k = val.get("per_K_latent_stats", {})
            blob = _ckpt_dict(ep, model, opt, tr["total_loss"], val, c, lat, per_k, physics, sched)
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{ep}.pt"
            torch.save(blob, ckpt_path)
            timing_path.write_text(json.dumps({"training_time_seconds": prev_t + int(time.time() - t0)}))
            if c.keep_last_n_checkpoints > 0:
                olds = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                              key=lambda p: int(p.stem.split("_")[-1]))
                for p in olds[:-c.keep_last_n_checkpoints]:
                    p.unlink()
            else:
                print(f"  checkpoint → {ckpt_path.name}")
            if c.eval_off_anchor_mhz:
                from experiments.exp038_true_multi.codes.eval_cross_freq import run_off_anchor_eval

                run_off_anchor_eval(
                    model,
                    val_ld,
                    bg=c.background_value + 0.5,
                    off_anchor_mhz=c.eval_off_anchor_mhz,
                    max_batches=c.eval_off_anchor_max_batches,
                    device=c.device,
                    out_csv=metrics_dir / f"off_anchor_eval_epoch_{ep}.csv",
                )

    total_s = prev_t + int(time.time() - t0)
    n_ep = max(len(train_times), 1)
    avg_train = sum(train_times) / n_ep
    avg_val = sum(val_times) / len(val_times) if val_times else 0.0
    val_frac = len(val_times) / n_ep if val_times else 0.0
    avg_epoch = avg_train + avg_val * val_frac
    timing_summary = {
        "training_time_seconds": total_s,
        "epochs_in_run": n_ep,
        "avg_train_sec": round(avg_train, 2),
        "avg_val_sec": round(avg_val, 2),
        "avg_epoch_sec_est": round(avg_epoch, 2),
        "val_on_checkpoint_only": c.val_on_checkpoint_only,
        "val_runs": len(val_times),
    }
    timing_path.write_text(json.dumps(timing_summary, indent=2))
    print(
        f"\nTiming: {total_s // 60}m total  |  avg train {avg_train:.1f}s/ep  "
        f"|  avg val {avg_val:.1f}s (when run)  |  est ~{avg_epoch:.1f}s/ep  "
        f"|  → ~{avg_epoch * c.num_epochs / 60:.0f}m for {c.num_epochs} epochs",
    )
    print(f"  Per-epoch log → {epoch_timing_path}")

    # If num_epochs is not a multiple of checkpoint_interval, still log final metrics.
    if val and c.num_epochs % c.checkpoint_interval != 0:
        logger.log_dict(
            c.num_epochs,
            loss_dict={k: tr[k] for k in ("total_loss", "recon_loss", "kl_loss", "kl_gaussian",
                                          "heatmap_loss", "occupancy_loss", "impedance_loss")},
            val_loss_dict={k: val[k] for k in LOSS_KEYS},
        )
        logger.log_latent_stats(log_dir / "latent_stats.csv", c.num_epochs, compute_beta(c.num_epochs - 1, c), val)

    lat = VAETrainingLogger.build_latent_stats(val)
    torch.save(_ckpt_dict(c.num_epochs, model, opt, tr["total_loss"], val, c, lat,
                          val.get("per_K_latent_stats", {}), physics, sched),
               ckpt_dir / "last_model.pt")
    logger.log_complete(val.get("total_loss", float("nan")), ckpt_dir, f"{total_s // 60}m")
    logger.plot(save_path=str(plots_dir / "convergence_final.png"))
    logger.plot_loss_components(save_path=str(plots_dir / "loss_components_final.png"))
    logger.plot_overfitting(save_path=str(plots_dir / "overfitting_final.png"))
    if physics:
        logger.plot_physics(save_path=str(plots_dir / "physics_losses_final.png"))
    logger.print_statistics()
    (exp / "config.yaml").write_text(json.dumps(asdict(c), indent=2), encoding="utf-8")


if __name__ == "__main__":
    train_vae()
