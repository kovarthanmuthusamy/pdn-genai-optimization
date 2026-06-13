"""Training script — Multi-Input VAE (exp037_lat_change)."""
# ── Stdlib ────────────────────────────────────────────────────────────────────
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Project root detection ────────────────────────────────────────────────────
_this_file = Path(__file__).resolve()
PROJECT_ROOT: str = next(
    (str(p) for p in _this_file.parents
     if (p / 'src_vae').exists() and (p / 'experiments').exists()),
    str(_this_file.parents[3]),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.exp037_lat_change.codes.vae_multi_input_simple import MultiInputVAE
from src_vae.others.dataloader import create_data_loaders
from src_vae.others.vae_logger import VAETrainingLogger


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Architecture ──────────────────────────────────────────────────────────
    latent_dim:          int   = 32
    heatmap_private_dim: int   = 8
    cond_dim:            int   = 8
    # Impedance: (1,231) log-z ch0 only. K conditions occ/imp; no PI_freq in this experiment.

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs:   int   = 400
    batch_size:   int   = 64
    learning_rate: float = 1e-4
    lr_patience:  int   = 20    # ReduceLROnPlateau patience (epochs)
    lr_factor:    float = 0.5   # LR multiplier on plateau
    lr_min:       float = 5e-6  # LR floor
    train_split:  float = 0.9
    num_workers:  int   = 4

    # ── K-balancing ───────────────────────────────────────────────────────────
    balance_k:           bool  = True
    k_balance_power:     float = 0.5
    k_balance_smoothing: float = 1e-3
    stratify_by_k:       bool  = True

    # ── Reconstruction weights ────────────────────────────────────────────────
    heatmap_weight:         float = 3.0
    occupancy_weight:       float = 4.0   # raised from 2→4 to match exp035  # lowered from 3.0: occ train=0.08 vs val=0.24 was overfitting
    impedance_weight:       float = 3.0
    impedance_deriv_weight: float = 1.0
    impedance_peak_weight:  float = 2.0
    impedance_peak_max_weight: float = 3.0  # direct MSE on max(ch0) per sample (peak height)
    # ── Peak-specific losses ──────────────────────────────────────────────────
    # In PDN design a missed/underestimated resonance peak = design failure.
    # Three complementary terms ensure all peaks are faithfully reproduced:
    #   topk_k            : supervise the N highest-amplitude peaks (not just max)
    #   topk_weight       : MSE weight for those top-K frequencies
    #   under_penalty     : extra cost when recon < target at ANY freq (asymmetric)
    #                       1.0 = symmetric, 2.0 = underestimate costs 2× more
    #   concavity_weight  : upweight Huber at all peak regions (d²<0 ⟹ local max)
    impedance_topk_k:          int   = 15   # supervise top-15 amplitude frequencies
    impedance_topk_weight:     float = 2.0  # MSE at those frequencies (lowered: was dominating loss x12 combined with impedance_weight)
    impedance_under_penalty:   float = 2.0  # asymmetric multiplier when recon < target
    impedance_concavity_weight: float = 2.0 # Huber weight at concave regions (-d²>0)
    # (impedance_peak_weight and impedance_peak_max_weight removed — redundant
    #  with concavity_loss and topk_loss respectively)
    heatmap_peak_weight:    float = 3.0
    heatmap_grad_weight:    float = 1.5
    # Anti-averaging sharpness losses — counteract VAE's tendency to blur heatmaps:
    #   lap_weight     : MSE on Laplacian residual (high-freq edge content)
    #   contrast_weight: margin loss pushing fg pixels above bg pixels by a margin
    heatmap_lap_weight:      float = 2.0
    heatmap_contrast_weight: float = 2.5
    heatmap_contrast_margin: float = 1.5  # minimum gap between fg and bg means (in z-score units)
    heatmap_bg_weight:       float = 0.5  # Huber on background pixels (was 0 = no bg gradient — root cause of mid-range output)
    heatmap_dynrange_weight: float = 2.0  # one-sided MSE: penalise when max(fg_recon) < max(fg_target)
    occupancy_focal_gamma:  float = 2.0
    focal_gamma_warmup_epochs: int = 120  # linear warmup 0→gamma_final over N epochs
    occ_k_consistency_weight: float = 1.0  # MSE(sigmoid(logits).sum(), K) — pushes exactly K slots active
    # ── K-aware loss weighting ────────────────────────────────────────────────
    use_k_weighting:    bool  = True
    occ_mid_k_center:   float = 25.0
    occ_mid_k_sigma:    float = 8.0
    occ_mid_k_boost:    float = 0.75
    hm_low_k_threshold: int   = 3
    hm_low_k_multiplier: float = 2.0

    # ── Latent regularisation ─────────────────────────────────────────────────
    free_bits:            float = 0.05   # lowered to match exp034: lets KL relax more
    mu_hinge_threshold:   float = 4.0
    mu_hinge_weight:      float = 0.10
    mu_bias_weight:       float = 0.05
    per_expert_kl_weight: float = 0.02
    sigma_reg_weight:     float = 2.0   # 2.0: gentler push — was 4.0 which dominated recon gradients
    sigma_reg_target:     float = 0.45  # target posterior sigma; lowered to match exp034
                                        # inside the N(0,1) prior cloud for good inference sampling

    # ── KL annealing (two-phase linear) ───────────────────────────────────────
    use_beta_annealing:    bool  = True
    beta_start_epoch:      int   = 0
    beta_end_epoch:        int   = 200
    beta_initial:          float = 0.01   # small non-zero start avoids pure-AE phase
    beta_final:            float = 0.1
    beta_phase2_final:     float = 0.2
    beta_phase2_end_epoch: int   = 500

    # ── Modality dropout ──────────────────────────────────────────────────────
    modality_dropout:               float = 0.2
    modality_dropout_start:         float = 0.0
    modality_dropout_anneal_epochs: int   = 150

    # ── Cross-modal reconstruction ────────────────────────────────────────────
    cross_modal_weight:      float = 0.5
    cross_modal_update_freq: int   = 4   # run every N batches

    # ── Physics losses ────────────────────────────────────────────────────────
    physics_ri_weight:            float = 1.0
    physics_critic_sup_weight:    float = 2.0
    physics_ar_weight:            float = 0.5
    physics_fg_clip_min:          float = -1.04
    physics_critic_warmup_epochs: int   = 50
    physics_slope_anneal_epochs:  int   = 100

    # ── Penalty curriculum ────────────────────────────────────────────────
    # topk / concavity / Laplacian / contrast / bg / dynrange are noisy in
    # early training. Ramp in linearly so base recon stabilises first.
    penalty_warmup_epochs: int = 50   # epochs to ramp penalty terms 0→1 (≈10% of 500)

    # ── Paths & checkpointing ─────────────────────────────────────────────────
    data_dir:            str = "/home/ubuntu/gan/datasets/data_norm"
    experiment_dir:      str = "/home/ubuntu/gan/experiments/exp037_lat_change"
    checkpoint_interval: int = 50
    keep_last_n_checkpoints: int = 0   # keep N most recent epoch checkpoints (0 = keep all)
    resume_checkpoint:   int | None = None  # epoch number, or None to start fresh

    # ── Runtime overrides (set from normalization_stats.json) ─────────────────
    background_value: float = -3.6228

    # ── Performance ───────────────────────────────────────────────────────────
    amp:                str  = "bf16"           # off | auto | bf16 | fp16
    tf32:               bool = False
    compile:            bool = False
    compile_mode:       str  = "reduce-overhead"
    persistent_workers: bool = True
    prefetch_factor:    int  = 2

    # ── KAN impedance decoder anti-overfit ────────────────────────────────────
    # Spline L1 penalty penalises large spline weights; combined with a lower
    # LR and extra weight-decay on spline params this prevents the KAN from
    # memorising the sparse 30k dataset while keeping MLP paths unaffected.
    kan_spline_l1_weight: float = 1e-4   # added to total loss each step
    kan_spline_wd:        float = 1e-4   # AdamW weight-decay on spline params only
    kan_spline_lr_factor: float = 0.5    # spline LR = base_lr * this factor

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS LOSS  (imported from physics_loss.py so all networks are trained and
#               their weights are saved in the checkpoint's physics_state_dict)
# ══════════════════════════════════════════════════════════════════════════════

from experiments.exp037_lat_change.codes.physics_loss import PhysicsLoss  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_beta(epoch: int, cfg: Config) -> float:
    if not cfg.use_beta_annealing:
        return cfg.beta_final
    if epoch < cfg.beta_start_epoch:
        return cfg.beta_initial
    if epoch < cfg.beta_end_epoch:
        t = (epoch - cfg.beta_start_epoch) / (cfg.beta_end_epoch - cfg.beta_start_epoch)
        return cfg.beta_initial + t * (cfg.beta_final - cfg.beta_initial)
    if epoch < cfg.beta_phase2_end_epoch:
        t = (epoch - cfg.beta_end_epoch) / (cfg.beta_phase2_end_epoch - cfg.beta_end_epoch)
        return cfg.beta_final + t * (cfg.beta_phase2_final - cfg.beta_final)
    return cfg.beta_phase2_final


def compute_modality_dropout(epoch: int, cfg: Config) -> float:
    if epoch >= cfg.modality_dropout_anneal_epochs:
        return cfg.modality_dropout
    t = epoch / max(cfg.modality_dropout_anneal_epochs, 1)
    return cfg.modality_dropout_start + t * (cfg.modality_dropout - cfg.modality_dropout_start)


def compute_focal_gamma(epoch: int, cfg: Config) -> float:
    """Linear warmup of focal gamma: 0→cfg.occupancy_focal_gamma."""
    if cfg.focal_gamma_warmup_epochs <= 0 or cfg.occupancy_focal_gamma <= 0:
        return cfg.occupancy_focal_gamma
    if epoch >= cfg.focal_gamma_warmup_epochs:
        return cfg.occupancy_focal_gamma
    t = epoch / max(cfg.focal_gamma_warmup_epochs, 1)
    return t * cfg.occupancy_focal_gamma


def _physics_stage_weights(epoch: int, cfg: Config) -> tuple[float, ...]:
    if epoch < cfg.physics_critic_warmup_epochs:
        # During critic warmup: only critic supervision losses active; analytical/consistency off
        return (0.0, cfg.physics_critic_sup_weight, 0.0)
    # Analytical (AR/RI) and consistency losses anneal in after warmup
    slope_t = min(1.0, (epoch - cfg.physics_critic_warmup_epochs) /
                  max(cfg.physics_slope_anneal_epochs, 1))
    return (cfg.physics_ri_weight * slope_t,
            cfg.physics_critic_sup_weight,
            cfg.physics_ar_weight * slope_t)


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

LOSS_KEYS: tuple[str, ...] = (
    'total_loss', 'recon_loss', 'kl_loss', 'kl_gaussian',
    'heatmap_loss', 'occupancy_loss', 'impedance_loss',
    'sigma_floor_loss', 'expert_kl_loss', 'sigma_reg_loss',
    'physics_ri_loss', 'physics_critic_sup_loss', 'physics_ar_loss',
)


def focal_bce_per_sample(pred: torch.Tensor, target: torch.Tensor, gamma: float,
                         pos_weight: 'torch.Tensor | None' = None) -> torch.Tensor:
    """Focal BCE per sample (B,). pred = raw logits.

    pos_weight: (52,) or (B, 52) — passed to BCEWithLogitsLoss to up-weight
    the positive class.  When None, all classes are weighted equally.
    """
    bce = F.binary_cross_entropy_with_logits(
        pred, target.clamp(0.0, 1.0),
        pos_weight=pos_weight,
        reduction='none',
    )
    bce_ps = bce if bce.dim() == 1 else bce.mean(dim=1)
    if gamma == 0.0 or bce.dim() == 1:
        return bce_ps
    return ((1.0 - torch.exp(-bce)).pow(gamma) * bce).mean(dim=1)


# Fixed Laplacian kernel for sharpness loss (registered once at module level).
_LAP_KERNEL = torch.tensor(
    [[[[0.,  1., 0.],
       [1., -4., 1.],
       [0.,  1., 0.]]]], dtype=torch.float32
)  # (1,1,3,3)
_LAP_KERNEL_CACHE: dict = {}


def _lap_kernel(device, dtype=torch.float32) -> torch.Tensor:
    """Return _LAP_KERNEL on the requested device, caching after first use."""
    key = (str(device), dtype)
    if key not in _LAP_KERNEL_CACHE:
        _LAP_KERNEL_CACHE[key] = _LAP_KERNEL.to(device=device, dtype=dtype)
    return _LAP_KERNEL_CACHE[key]


def _penalty_scale(epoch: int, cfg) -> float:
    """Linear ramp 0→1 for noisy penalty terms (topk, concavity, Laplacian,
    contrast, bg, dynrange). Base reconstruction + peak + grad stay always-on."""
    if cfg.penalty_warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / max(cfg.penalty_warmup_epochs, 1))


def heatmap_loss_per_sample(recon: torch.Tensor, target: torch.Tensor, cfg: Config,
                            penalty_scale: float = 1.0) -> torch.Tensor:
    """Foreground Huber + gradient + Laplacian sharpness + fg/bg contrast. Returns (B,).

    Laplacian term: MSE(lap(recon), lap(target)) on fg region.
    Penalises smooth blobs that have the right mean but wrong high-frequency
    content — the primary cause of the averaging effect in VAE heatmaps.

    Contrast term: hinge loss pushing mean(fg_recon) > mean(bg_recon) + margin.
    Prevents the model from “cheating” by raising the background to meet the
    foreground instead of sharpening the foreground itself.
    """
    bg    = cfg.background_value + 0.5
    fg    = (target > bg).float()          # (B,1,H,W) binary foreground mask
    bg_m  = 1.0 - fg
    huber = F.huber_loss(recon, target, delta=1.0, reduction='none')

    n_fg   = fg.sum(dim=(1,2,3)).clamp(min=1.0)
    base   = (huber * fg).sum(dim=(1,2,3)) / n_fg
    intens = (target - bg).clamp(min=0.0)
    peak   = (huber * fg * intens).sum(dim=(1,2,3)) / (intens * fg).sum(dim=(1,2,3)).clamp(min=1e-6)

    fg_dx = fg[:,:,:,1:] * fg[:,:,:,:-1]
    fg_dy = fg[:,:,1:,:] * fg[:,:,:-1,:]
    gx = F.huber_loss(recon[:,:,:,1:]-recon[:,:,:,:-1], target[:,:,:,1:]-target[:,:,:,:-1], delta=0.5, reduction='none')
    gy = F.huber_loss(recon[:,:,1:,:]-recon[:,:,:-1,:], target[:,:,1:,:]-target[:,:,:-1,:], delta=0.5, reduction='none')
    grad_x = (gx * fg_dx).sum(dim=(1,2,3)) / fg_dx.sum(dim=(1,2,3)).clamp(min=1.0)
    grad_y = (gy * fg_dy).sum(dim=(1,2,3)) / fg_dy.sum(dim=(1,2,3)).clamp(min=1.0)

    # ── Laplacian sharpness ──────────────────────────────────────────────────
    # Apply Laplacian filter to both recon and target, then MSE on the fg
    # high-frequency residual.  Blurry recons have near-zero Laplacian response
    # while sharp targets have strong edges — this directly penalises blurring.
    lap_k = _lap_kernel(recon.device, recon.dtype)
    lap_r = F.conv2d(recon,  lap_k, padding=1)   # (B,1,H,W)
    lap_t = F.conv2d(target, lap_k, padding=1)
    lap_err    = (lap_r - lap_t).pow(2)
    lap_loss   = (lap_err * fg).sum(dim=(1,2,3)) / n_fg              # (B,)

    # ── Foreground / background contrast margin ────────────────────────────
    # Hinge: 0 if mean(fg_recon) ≥ mean(bg_recon) + margin, else penalise.
    # Stops the blurring strategy of “raise background to meet foreground”.
    n_bg     = bg_m.sum(dim=(1,2,3)).clamp(min=1.0)
    mean_fg  = (recon * fg).sum(dim=(1,2,3))  / n_fg     # (B,)
    mean_bg  = (recon * bg_m).sum(dim=(1,2,3)) / n_bg
    contrast_loss = F.relu(mean_bg + cfg.heatmap_contrast_margin - mean_fg)   # (B,)

    # ── Background reconstruction ────────────────────────────────────────────
    # Background pixels had zero gradient — model defaulted to z-score mean (~0)
    # instead of true background (-3.6), compressing the full dynamic range.
    bg_huber  = (huber * bg_m).sum(dim=(1,2,3)) / n_bg   # (B,)

    # ── Dynamic range preservation ────────────────────────────────────────────
    # One-sided: penalise when the reconstructed peak is below the target peak.
    # Prevents "safe middle" predictions that underestimate hot-spot amplitude.
    max_fg_recon = (recon  * fg + (1 - fg) * recon.min()).flatten(1).max(dim=1).values  # (B,)
    max_fg_targ  = (target * fg + (1 - fg) * target.min()).flatten(1).max(dim=1).values  # (B,)
    dynrange_loss = F.relu(max_fg_targ - max_fg_recon).pow(2)                   # (B,)

    return (base
            + cfg.heatmap_peak_weight * peak
            + cfg.heatmap_grad_weight * (grad_x + grad_y)
            + penalty_scale * (
                cfg.heatmap_lap_weight      * lap_loss
                + cfg.heatmap_contrast_weight * contrast_loss
                + cfg.heatmap_bg_weight       * bg_huber
                + cfg.heatmap_dynrange_weight * dynrange_loss))


def _imp_ch0(x: torch.Tensor) -> torch.Tensor:
    """Log-z impedance channel: (B, 1, 231) or (B, 231) → (B, 231)."""
    return x[:, 0] if x.dim() == 3 else x


def impedance_loss_per_sample(
    recon: torch.Tensor, target: torch.Tensor, cfg: Config,
    *, delta_raw: float = 1.0, delta_deriv: float = 1.0,
    imp_log_std: float = 1.0, penalty_scale: float = 1.0,
) -> torch.Tensor:
    """Multi-peak-aware impedance loss.  Returns (B,).

    Terms
    -----
    base_huber       : Huber on ch0 in log-Ohm space (broad shape).
    concavity_huber  : Huber reweighted by -d² (peak bodies) — upweights ALL
                       local-max regions, not just the global max.
    d1_loss          : Huber on the 1st derivative (slope / peak sharpness).
    d1_peak_emphasis : d1-magnitude-weighted Huber (existing, kept for compat).
    peak_max_mse     : Direct MSE on global max(ch0).
    topk_asym_mse    : MSE at the top-K amplitude frequencies with an asymmetric
                       penalty: when recon < target (under-estimate of resonance)
                       the error is multiplied by cfg.impedance_under_penalty.
                       This directly prevents the model from safely ignoring
                       peaks by regression-to-mean.
    """
    s     = imp_log_std
    ch0_r = _imp_ch0(recon) * s         # (B, 231)
    ch0_t = _imp_ch0(target) * s

    # ── Base Huber ─────────────────────────────────────────────────────────
    raw_err = F.huber_loss(ch0_r, ch0_t, delta=delta_raw, reduction='none')    # (B, 231)
    raw     = raw_err.mean(dim=1)                                               # (B,)

    # ── Concavity peak mask: upweight ALL peak bodies ───────────────────────
    # d2 = second discrete derivative; d2 < 0 at local maxima (concave down).
    # (-d2).clamp(min=0) is zero on flat/rising regions, positive at peaks.
    # Normalised per sample so the total weight sums to 231.
    d2          = torch.diff(ch0_t, n=2, dim=-1)                               # (B, 229)
    concav_w    = (-d2).clamp(min=0.0)                                         # (B, 229)
    concav_norm = concav_w / (concav_w.mean(dim=1, keepdim=True) + 1e-6)       # (B, 229)
    concavity_loss = (raw_err[:, 1:-1] * concav_norm).mean(dim=1)              # (B,)

    # ── Derivative supervision ─────────────────────────────────────────────
    d1_r  = torch.diff(ch0_r, dim=-1)                                          # (B, 230)
    d1_t  = torch.diff(ch0_t, dim=-1)
    d1_loss = F.huber_loss(d1_r, d1_t, delta=delta_deriv, reduction='none').mean(dim=1)

    # d1-weighted peak emphasis (kept for backward compat)
    peak_w   = d1_t.abs() / (d1_t.abs().amax(dim=1, keepdim=True) + 1e-6)
    peak_raw = (raw_err[:, :-1] * peak_w).sum(dim=1) / peak_w.sum(dim=1).clamp(min=1e-6)

    # ── Global peak height MSE ─────────────────────────────────────────────
    peak_max_err = (ch0_r.max(dim=-1).values - ch0_t.max(dim=-1).values).pow(2)

    # ── Top-K multi-peak supervision with asymmetric underestimate penalty ──
    # Select the K frequencies with highest target amplitude (non-differentiable
    # selection, but values at those fixed indices ARE differentiable w.r.t. recon).
    K_top   = min(cfg.impedance_topk_k, ch0_t.shape[-1])
    _, topk_idx = ch0_t.topk(K_top, dim=-1)                                    # (B, K_top)
    r_at_peaks  = ch0_r.gather(1, topk_idx)                                    # (B, K_top)
    t_at_peaks  = ch0_t.gather(1, topk_idx)                                    # (B, K_top)
    err_at_peaks = (r_at_peaks - t_at_peaks).pow(2)                            # (B, K_top)
    # Asymmetric: when recon < target, multiply by under_penalty
    under_mask   = (r_at_peaks < t_at_peaks).float()
    asym_weight  = 1.0 + (cfg.impedance_under_penalty - 1.0) * under_mask      # (B, K_top)
    topk_loss    = (err_at_peaks * asym_weight).mean(dim=1)                    # (B,)

    return (raw
            + cfg.impedance_deriv_weight * d1_loss
            + penalty_scale * (
                cfg.impedance_concavity_weight * concavity_loss
                + cfg.impedance_topk_weight    * topk_loss))


def _weighted_mean(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w.to(dtype=v.dtype)
    return (v * w).sum() / w.sum().clamp(min=1e-6)


def _k_loss_weights(K: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    kf    = K.float()
    bump  = torch.exp(-0.5 * ((kf - cfg.occ_mid_k_center) / cfg.occ_mid_k_sigma) ** 2)
    w_occ = 1.0 + cfg.occ_mid_k_boost * bump
    w_hm  = torch.where(K <= cfg.hm_low_k_threshold,
                        torch.full_like(kf, cfg.hm_low_k_multiplier), torch.ones_like(kf))
    return w_hm, w_occ


def _expert_kl_loss(expert_stats: dict) -> torch.Tensor:
    total = sum((-0.5 * (1.0 + lv - mu.pow(2) - lv.exp())).mean()
                for mu, lv in expert_stats.values())
    return total / max(len(expert_stats), 1)    # type: ignore[return-value]


def vae_loss(
    recon_hm, recon_occ, recon_imp,
    target_hm, target_occ, target_imp,
    mu, logvar, beta: float, cfg: Config,
    expert_stats=None, *, epoch: int = 0,
    K: torch.Tensor | None = None,
    apply_k_weights: bool = True,
    physics: 'PhysicsLoss | None' = None,
    physics_weights: tuple[float, ...] | None = None,
    imp_log_std: float = 1.0,
    penalty_scale: float = 1.0,
) -> dict:
    # Reconstruction
    hm_ps  = heatmap_loss_per_sample(recon_hm, target_hm, cfg, penalty_scale)
    # Dynamic pos_weight: for each sample, up-weight positives by (52-K)/K so that
    # rare active slots (low K) get proportionally stronger gradient signal.
    # Clamped to [0.5, 10] to avoid extreme weights at K=1 or K=51.
    occ_pos_w: torch.Tensor | None = None
    if K is not None:
        pw_per_sample = ((52.0 - K.float()) / K.float().clamp(min=1.0)).clamp(0.5, 10.0)  # (B,)
        occ_pos_w = pw_per_sample.unsqueeze(1).expand_as(recon_occ)  # (B, 52)
    focal_gamma = compute_focal_gamma(epoch, cfg)
    occ_ps = focal_bce_per_sample(recon_occ, target_occ, gamma=focal_gamma,
                                   pos_weight=occ_pos_w)
    imp_ps = impedance_loss_per_sample(recon_imp, target_imp, cfg, imp_log_std=imp_log_std,
                                       penalty_scale=penalty_scale)

    if apply_k_weights and cfg.use_k_weighting and K is not None:
        w_hm, w_occ = _k_loss_weights(K, cfg)
        loss_hm, loss_occ = _weighted_mean(hm_ps, w_hm), _weighted_mean(occ_ps, w_occ)
    else:
        loss_hm, loss_occ = hm_ps.mean(), occ_ps.mean()
    loss_imp = imp_ps.mean()

    # K-count consistency: push sum(sigmoid(logits)) toward K.
    # This forces the decoder to activate exactly K slots, making probabilities
    # sharp (K slots near 1, rest near 0) rather than uniformly near 0.5.
    occ_k_loss = mu.new_zeros(())
    if cfg.occ_k_consistency_weight > 0 and K is not None:
        prob_sum   = torch.sigmoid(recon_occ).sum(dim=1)  # (B,)
        occ_k_loss = F.mse_loss(prob_sum, K.float())

    recon = (cfg.heatmap_weight * loss_hm
             + cfg.occupancy_weight * loss_occ
             + cfg.impedance_weight * loss_imp
             + cfg.occ_k_consistency_weight * occ_k_loss)

    # KL with free bits
    kl_pd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if cfg.free_bits > 0:
        kl_pd = kl_pd.clamp(min=cfg.free_bits)
    kl = kl_pd.mean()

    # Latent regularisers
    mu_hinge = F.relu(mu.abs() - cfg.mu_hinge_threshold).pow(2).mean()
    mu_bias  = mu.mean(dim=0).pow(2).mean()

    exp_kl = mu.new_zeros(())
    if expert_stats and cfg.per_expert_kl_weight > 0:
        exp_kl = _expert_kl_loss(expert_stats)

    sigma_reg = mu.new_zeros(())
    if cfg.sigma_reg_weight > 0 and expert_stats:
        eps       = 1e-8
        prec      = torch.stack([1.0/(lv.exp()+eps) for _, lv in expert_stats.values()]).sum(0) + 1.0
        sig_f     = (1.0/(prec+eps)).sqrt()
        d_low     = (cfg.sigma_reg_target - sig_f).clamp(min=0.0)
        d_high    = (sig_f - cfg.sigma_reg_target*1.3).clamp(min=0.0)
        w_low     = cfg.sigma_reg_target / sig_f.detach().clamp(min=0.05)
        sigma_reg = (d_low.pow(2)*w_low + d_high.pow(2)).mean()

    total = (recon + beta*kl
             + cfg.mu_hinge_weight * mu_hinge
             + cfg.mu_bias_weight  * mu_bias
             + cfg.per_expert_kl_weight * exp_kl
             + cfg.sigma_reg_weight     * sigma_reg)

    # Physics — forward() returns (ri, cs, ar).
    z = mu.new_zeros
    phys_ri = phys_cs = phys_ar = z(())
    if physics is not None:
        w_ri, w_cs, w_ar = physics_weights or (
            cfg.physics_ri_weight, cfg.physics_critic_sup_weight,
            cfg.physics_ar_weight,
        )
        phys_ri, phys_cs, phys_ar = physics(
            recon_hm, recon_occ, recon_imp,
            target_occ=target_occ, target_hm=target_hm,
        )
        total = total + w_ri * phys_ri + w_cs * phys_cs + w_ar * phys_ar

    return dict(
        total_loss=total, recon_loss=recon, kl_loss=kl, kl_gaussian=kl,
        heatmap_loss=loss_hm, occupancy_loss=loss_occ, impedance_loss=loss_imp,
        sigma_floor_loss=z(()), expert_kl_loss=exp_kl, sigma_reg_loss=sigma_reg,
        physics_ri_loss=phys_ri, physics_critic_sup_loss=phys_cs,
        physics_ar_loss=phys_ar,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAT TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def _init_trackers(modalities=('heatmap', 'occupancy', 'impedance')):
    scalar  = {k: {'sum':0.0,'std_sum':0.0,'min':float('inf'),'max':float('-inf')}
               for k in ('mu','logvar','std')}
    per_mod = {m: {'mu_mean':0.0,'mu_std':0.0,'std_mean':0.0,'std_std':0.0}
               for m in modalities}
    return scalar, per_mod


def _update_scalar(s: dict, t: torch.Tensor):
    s['sum']     += t.mean().item()
    s['std_sum'] += t.std().item()
    s['min']      = min(s['min'], t.min().item())
    s['max']      = max(s['max'], t.max().item())


def _finalize_scalar(s: dict, n: int) -> dict:
    return {'mean':s['sum']/n,'std':s['std_sum']/n,'min':s['min'],'max':s['max']}


def _update_modality_stats(per_mod: dict, expert_stats: dict):
    for name, (mu_m, lv_m) in expert_stats.items():
        if name not in per_mod: continue
        std_m = (lv_m/2).exp()
        per_mod[name]['mu_mean']  += mu_m.mean().item()
        per_mod[name]['mu_std']   += mu_m.std().item()
        per_mod[name]['std_mean'] += std_m.mean().item()
        per_mod[name]['std_std']  += std_m.std().item()


def _finalize_modality_stats(per_mod: dict, n: int) -> dict:
    return {name: {k: v/n for k, v in vals.items()} for name, vals in per_mod.items()}


# ══════════════════════════════════════════════════════════════════════════════
# BATCH / CHECKPOINT / CROSS-MODAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _device_type(device: str) -> str:
    return 'cuda' if 'cuda' in device else 'cpu'


def _autocast_dtype(cfg: Config) -> torch.dtype | None:
    if cfg.amp == 'off' or _device_type(cfg.device) != 'cuda': return None
    if cfg.amp == 'bf16': return torch.bfloat16
    if cfg.amp == 'fp16': return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _prepare_batch(batch: dict, cfg: Config):
    nb  = _device_type(cfg.device) == 'cuda'
    hm  = batch['heatmap_norm'].to(cfg.device, non_blocking=nb)
    occ = batch['occupancy'].to(cfg.device, non_blocking=nb)
    imp = batch['impedance'].to(cfg.device, non_blocking=nb)
    K   = batch['K'].to(cfg.device, non_blocking=nb)
    if hm.dim()  == 3: hm  = hm.unsqueeze(1)
    if imp.dim() == 1:
        imp = imp.unsqueeze(0)
    if imp.dim() == 2 and imp.shape[1] == 231:
        imp = imp.unsqueeze(1)
    elif imp.dim() == 3:
        imp = imp[:, :1, :]   # ch0 log-z only (legacy multi-ch files)
    hm_enc = hm.masked_fill(hm < cfg.background_value + 0.5, 0.0)
    return hm, hm_enc, occ, imp, K


def load_checkpoint(path: str, model, optimizer=None, device='cuda', physics=None):
    print(f"\n{'='*60}\nRESUMING FROM: {path}\n{'='*60}")
    ckpt = torch.load(path, map_location=device)
    state = ckpt['model_state_dict']
    # Remap occupancy_decoder key indices if the checkpoint was saved before the
    # second Dropout was added (old index 7 → new index 8 for the final Linear).
    remap = {
        'occupancy_decoder.7.weight': 'occupancy_decoder.8.weight',
        'occupancy_decoder.7.bias':   'occupancy_decoder.8.bias',
    }
    for old_k, new_k in remap.items():
        if old_k in state and new_k not in state:
            state[new_k] = state.pop(old_k)
            print(f"  Remapped checkpoint key: {old_k} → {new_k}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  Checkpoint mismatch — loaded with strict=False.")
        if missing:     print(f"    Missing  (new params, randomly init): {missing}")
        if unexpected:  print(f"    Ignored  (removed from arch):         {unexpected}")
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if physics:
        if 'physics_state_dict' in ckpt:
            physics.load_state_dict(ckpt['physics_state_dict'], strict=False)
            print("  Loaded physics networks (critic).")
        else:
            print("  Warning: no physics_state_dict — all critic networks start from random.")
    epoch    = ckpt.get('epoch', 0)
    best_val = ckpt.get('val_loss', float('inf'))
    print(f"  Resumed epoch={epoch}  best_val={best_val:.4f}\n")
    return epoch, best_val


def _build_checkpoint(epoch, model, optimizer, train_loss, val, cfg,
                       latent_stats, per_K, physics=None, scheduler=None) -> dict:
    d = {
        'epoch': epoch, 'train_loss': train_loss, 'val_loss': val['total_loss'],
        'config': asdict(cfg), 'latent_stats': latent_stats, 'per_K_latent_stats': per_K,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mu_mean': val['mu_mean'], 'mu_std': val['mu_std'],
        'mu_min':  val['mu_min'],  'mu_max': val['mu_max'],
    }
    if physics   is not None: d['physics_state_dict']   = physics.state_dict()
    if scheduler is not None: d['scheduler_state_dict'] = scheduler.state_dict()
    return d


def _cross_modal_loss(model, hm_enc, occ, imp, hm, K, cfg: Config,
                      imp_log_std: float = 1.0, penalty_scale: float = 1.0) -> torch.Tensor:
    """Cross-modal reconstruction loss (heatmap + impedance sources), averaged."""
    w_hm_cm, w_occ_cm = _k_loss_weights(K, cfg) if cfg.use_k_weighting else (None, None)
    total = hm.new_zeros(())
    for source in ('heatmap', 'impedance'):
        z_cm            = model.encode_cross_modal(source, K, heatmap=hm_enc, impedance=imp)
        r_hm, r_occ, r_imp = model.decode(z_cm, K)
        hm_ps   = heatmap_loss_per_sample(r_hm, hm, cfg, penalty_scale)
        occ_ps  = focal_bce_per_sample(r_occ, occ, gamma=cfg.occupancy_focal_gamma)
        imp_ps  = impedance_loss_per_sample(r_imp, imp, cfg, delta_raw=1.0, delta_deriv=2.0,
                                            imp_log_std=imp_log_std, penalty_scale=penalty_scale)
        loss_hm  = _weighted_mean(hm_ps,  w_hm_cm)  if w_hm_cm  is not None else hm_ps.mean()
        loss_occ = _weighted_mean(occ_ps, w_occ_cm) if w_occ_cm is not None else occ_ps.mean()
        total = total + cfg.heatmap_weight*loss_hm + cfg.occupancy_weight*loss_occ + cfg.impedance_weight*imp_ps.mean()
    return total / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / VALIDATE EPOCHS
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, cfg: Config, epoch: int,
                beta: float, modality_dropout: float,
                physics=None, physics_weights=None,
                imp_log_std=1.0, scaler=None) -> dict:
    model.train()
    if physics is not None: physics.train()
    model.modality_dropout = modality_dropout

    loss_acc        = {k: 0.0 for k in LOSS_KEYS}
    scalar, per_mod = _init_trackers()
    amp_dtype       = _autocast_dtype(cfg)
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) # pyright: ignore[reportPrivateImportUsage]
    autocast  = torch.autocast(device_type=_device_type(cfg.device),
                                dtype=amp_dtype, enabled=(amp_dtype is not None))
    cm_freq   = max(1, cfg.cross_modal_update_freq)

    for batch_idx, batch in enumerate(loader):
        hm, hm_enc, occ, imp, K = _prepare_batch(batch, cfg)

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            recon_hm, recon_occ, recon_imp, mu, logvar, expert_stats = model(hm_enc, occ, imp, K)
            _ps = _penalty_scale(epoch, cfg)
            losses = vae_loss(recon_hm, recon_occ, recon_imp, hm, occ, imp,
                              mu, logvar, beta, cfg, expert_stats,
                              epoch=epoch, K=K, apply_k_weights=True,
                              physics=physics, physics_weights=physics_weights,
                              imp_log_std=imp_log_std, penalty_scale=_ps)
            if cfg.cross_modal_weight > 0 and batch_idx % cm_freq == 0:
                losses['total_loss'] = losses['total_loss'] + cfg.cross_modal_weight * _cross_modal_loss(
                    model, hm_enc, occ, imp, hm, K, cfg, imp_log_std=imp_log_std,
                    penalty_scale=_ps
                )

        with torch.no_grad():
            _update_scalar(scalar['mu'],     mu)
            _update_scalar(scalar['logvar'], logvar)
            _update_modality_stats(per_mod,  expert_stats)

        params = list(model.parameters()) + (list(physics.parameters()) if physics else [])
        if scaler.is_enabled():
            scaler.scale(losses['total_loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        for k in loss_acc:
            loss_acc[k] += losses[k].item()

    n      = len(loader)
    result: dict[str, Any] = {k: v/n for k, v in loss_acc.items()}
    result['beta'] = beta
    print(f"Ep {epoch+1} | loss={result['total_loss']:.4f}  recon={result['recon_loss']:.4f}"
          f"  kl={result['kl_loss']:.4f}  imp={result['impedance_loss']:.4f}"
          f"  occ={result['occupancy_loss']:.4f}  ri={result['physics_ri_loss']:.4f}",
          flush=True)
    mu_s = _finalize_scalar(scalar['mu'],     n)
    lv_s = _finalize_scalar(scalar['logvar'], n)
    result.update(mu_mean=mu_s['mean'], mu_std=mu_s['std'], mu_min=mu_s['min'], mu_max=mu_s['max'],
                  logvar_mean=lv_s['mean'], logvar_std=lv_s['std'],
                  logvar_min=lv_s['min'],   logvar_max=lv_s['max'])
    result['modality_stats'] = _finalize_modality_stats(per_mod, n)
    return result


def validate(model, loader, cfg: Config, beta: float, epoch: int,
             physics=None, physics_weights=None,
             imp_log_std=1.0) -> dict:
    model.eval()
    if physics is not None: physics.eval()

    loss_acc        = {k: 0.0 for k in LOSS_KEYS}
    scalar, per_mod = _init_trackers()
    gauss_dim       = model.latent_dim
    mu_sum    = torch.zeros(gauss_dim)
    mu_sq_sum = torch.zeros(gauss_dim)
    sigma_sum = torch.zeros(gauss_dim)
    n_val     = 0
    k_buckets: dict = {}

    amp_dtype = _autocast_dtype(cfg)
    autocast  = torch.autocast(device_type=_device_type(cfg.device),
                                dtype=amp_dtype, enabled=(amp_dtype is not None))

    with torch.inference_mode():
        for batch in loader:
            hm, hm_enc, occ, imp, K = _prepare_batch(batch, cfg)
            with autocast:
                recon_hm, recon_occ, recon_imp, mu, logvar, expert_stats = model(hm_enc, occ, imp, K)
                losses = vae_loss(recon_hm, recon_occ, recon_imp, hm, occ, imp,
                                  mu, logvar, beta, cfg, expert_stats,
                                  epoch=epoch, K=K, apply_k_weights=False,
                                  physics=physics, physics_weights=physics_weights,
                                  imp_log_std=imp_log_std)

            _update_scalar(scalar['mu'],     mu)
            _update_scalar(scalar['logvar'], logvar)
            _update_scalar(scalar['std'],    (logvar/2).exp())
            _update_modality_stats(per_mod,  expert_stats)

            mu_cpu  = mu.cpu()
            sig_cpu = (0.5 * logvar.cpu()).exp()
            mu_sum    += mu_cpu.sum(0); mu_sq_sum += mu_cpu.pow(2).sum(0)
            sigma_sum += sig_cpu.sum(0); n_val += mu_cpu.shape[0]

            # Vectorized per-K accumulation: iterate over unique K values in batch
            # rather than over every sample, reducing Python iterations significantly.
            K_cpu = K.cpu()
            for k_val in K_cpu.unique().tolist():
                k_val = int(k_val)
                mask = K_cpu == k_val
                mu_k_rows  = mu_cpu[mask]
                sig_k_rows = sig_cpu[mask]
                if k_val not in k_buckets:
                    k_buckets[k_val] = dict(mu_s=torch.zeros(gauss_dim),
                                            mu_sq=torch.zeros(gauss_dim),
                                            sig_s=torch.zeros(gauss_dim), n=0)
                bk = k_buckets[k_val]
                bk['mu_s']  += mu_k_rows.sum(0)
                bk['mu_sq'] += mu_k_rows.pow(2).sum(0)
                bk['sig_s'] += sig_k_rows.sum(0)
                bk['n']     += mask.sum().item()

            for k in loss_acc:
                loss_acc[k] += losses[k].item()

    n_b    = len(loader)
    result: dict[str, Any] = {k: v/n_b for k, v in loss_acc.items()}
    result['beta'] = beta

    for key in ('mu', 'logvar', 'std'):
        fs = _finalize_scalar(scalar[key], n_b)
        result.update({f"{key}_mean":fs['mean'],f"{key}_std":fs['std'],
                       f"{key}_min":fs['min'],  f"{key}_max":fs['max']})
    result['modality_stats'] = _finalize_modality_stats(per_mod, n_b)

    if n_val > 0:
        mu_pd  = mu_sum / n_val
        std_pd = ((mu_sq_sum/n_val - mu_pd.pow(2)).clamp(min=0.0) + (sigma_sum/n_val).pow(2)).sqrt()
        result['per_dim_stats'] = {'latent': {'mu_mean_per_dim': mu_pd.tolist(),
                                               'agg_std_per_dim': std_pd.tolist()}}

    per_K = {}
    for k_val, bk in k_buckets.items():
        if bk['n'] < 2: continue
        n_k  = bk['n']; mu_k = bk['mu_s']/n_k
        std_k = ((bk['mu_sq']/n_k - mu_k.pow(2)).clamp(min=0.0) + (bk['sig_s']/n_k).pow(2)).sqrt()
        per_K[str(k_val)] = {'latent': {'mu_mean_per_dim': mu_k.tolist(),
                                         'agg_std_per_dim': std_k.tolist()}}
    result['per_K_latent_stats'] = per_K
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_vae():
    cfg = Config()

    exp_path     = Path(cfg.experiment_dir)
    ckpt_path    = exp_path / "checkpoints"
    log_path     = exp_path / "logs"
    metrics_path = exp_path / "metrics"
    plots_path   = metrics_path / "plots"

    # Resolve int epoch → checkpoint file path
    resume_path: str | None = None
    if cfg.resume_checkpoint is not None:
        ep = int(cfg.resume_checkpoint)
        candidate = ckpt_path / f"checkpoint_epoch_{ep}.pt"
        if candidate.exists():
            resume_path = str(candidate)
        else:
            raise FileNotFoundError(
                f"Checkpoint for epoch {ep} not found: {candidate}"
            )

    if not resume_path:
        import shutil
        for p in (ckpt_path, log_path, metrics_path):
            if p.exists(): shutil.rmtree(p)
    for p in (exp_path, ckpt_path, log_path, metrics_path, plots_path):
        p.mkdir(parents=True, exist_ok=True)

    logger = VAETrainingLogger(log_dir=str(log_path), checkpoint_dir=str(ckpt_path),
                               csv_path=str(metrics_path / "loss.csv"))

    logger.log_start(exp_path.name, {
        'Device':  cfg.device,
        'Latent':  f"{cfg.latent_dim} (shared={cfg.latent_dim-cfg.heatmap_private_dim} priv={cfg.heatmap_private_dim})",
        'LR':      cfg.learning_rate,
        'Batch':   cfg.batch_size,
        'Epochs':  cfg.num_epochs,
    })

    # Normalization stats
    data_path  = Path(cfg.data_dir)
    stats_path = data_path / "normalization_stats.json"
    imp_log_std = 0.0
    if stats_path.exists():
        raw = json.loads(stats_path.read_text())
        cfg.background_value    = raw['background_value']
        cfg.physics_fg_clip_min = raw['Heatmap']['clip_min']
        imp_log_std  = float(raw['Impedance']['log_std']) or 1.0

    # Data loaders
    pin = _device_type(cfg.device) == 'cuda'
    train_loader, val_loader = create_data_loaders(
        data_dir=str(data_path), batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, normalize=False,
        train_split=cfg.train_split, seed=42, pin_memory=pin,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        stratify_by_k=cfg.stratify_by_k, balance_k=cfg.balance_k,
        k_balance_power=cfg.k_balance_power, k_balance_smoothing=cfg.k_balance_smoothing,
    )

    # Model
    if _device_type(cfg.device) == 'cuda':
        torch.backends.cudnn.benchmark    = True
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32   = cfg.tf32
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high' if cfg.tf32 else 'highest')

    model = MultiInputVAE(
        latent_dim=cfg.latent_dim, modality_dropout=cfg.modality_dropout,
        cond_dim=cfg.cond_dim, heatmap_private_dim=cfg.heatmap_private_dim,
    ).to(cfg.device)

    if cfg.compile and hasattr(torch, 'compile'):
        try:    model = torch.compile(model, mode=cfg.compile_mode)
        except Exception as e: print(f"  torch.compile failed ({e}) — skipping.")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=1e-4,
        foreach=True,
    )

    physics: PhysicsLoss | None = None
    if cfg.physics_ri_weight > 0 or cfg.physics_critic_sup_weight > 0:
        physics = PhysicsLoss(
            background_value=cfg.background_value, fg_clip_min=cfg.physics_fg_clip_min,
        ).to(cfg.device)
        optimizer.add_param_group({'params': list(physics.parameters()), 'lr': cfg.learning_rate})
        print(f"Physics: {sum(p.numel() for p in physics.parameters()):,} params"
              f"  warmup={cfg.physics_critic_warmup_epochs}ep"
              f"  [ri={cfg.physics_ri_weight} cs={cfg.physics_critic_sup_weight}"
              f" ar={cfg.physics_ar_weight}]")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_factor,
        patience=cfg.lr_patience, min_lr=cfg.lr_min,
    )

    # Resume
    start_epoch, best_val = 0, float('inf')
    if resume_path:
        start_epoch, best_val = load_checkpoint(resume_path, model, optimizer,
                                                 cfg.device, physics=physics)
        tmp = torch.load(resume_path, map_location='cpu')
        if 'scheduler_state_dict' in tmp:
            scheduler.load_state_dict(tmp['scheduler_state_dict'])
            print("  Loaded scheduler state.")

    # Timing
    timing_path  = metrics_path / "timing.json"
    prev_seconds = 0
    if timing_path.exists():
        try: prev_seconds = json.loads(timing_path.read_text()).get('training_time_seconds', 0)
        except Exception: pass

    amp_dtype   = _autocast_dtype(cfg)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) # type: ignore
    t0          = time.time()

    print(f"\nTraining epochs {start_epoch+1}–{cfg.num_epochs}...\n")

    for epoch in range(start_epoch, cfg.num_epochs):
        beta = compute_beta(epoch, cfg)
        md   = compute_modality_dropout(epoch, cfg)
        pw   = _physics_stage_weights(epoch, cfg) if physics is not None else None

        tr  = train_epoch(model, train_loader, optimizer, cfg, epoch, beta, md,
                          physics=physics, physics_weights=pw,
                          imp_log_std=imp_log_std,
                          scaler=grad_scaler)
        val = validate(model, val_loader, cfg, beta, epoch,
                       physics=physics, physics_weights=pw,
                       imp_log_std=imp_log_std)

        scheduler.step(val['total_loss'])
        cur_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            logger.log_dict(
                epoch=epoch+1,
                loss_dict={k: tr[k] for k in ('total_loss','recon_loss','kl_loss','kl_gaussian',
                                               'heatmap_loss','occupancy_loss','impedance_loss')},
                val_loss_dict={k: val[k] for k in LOSS_KEYS},
            )
            ms = val['modality_stats']
            s  = 2 if physics is not None and epoch+1 >= cfg.physics_critic_warmup_epochs else 1
            best_marker = " ★" if val['total_loss'] <= best_val else ""
            print(f"\n  Ep {epoch+1:>3}/{cfg.num_epochs}  "
                  f"train={tr['total_loss']:.4f}  val={val['total_loss']:.4f}{best_marker}  "
                  f"recon={val['recon_loss']:.4f}  kl={val['kl_loss']:.4f}  "
                  f"sig_reg={val['sigma_reg_loss']:.4f}  β={beta:.4f}  lr={cur_lr:.2e}")
            if physics is not None:
                print(f"         Physics[s{s}]  ri={val['physics_ri_loss']:.4f}"
                      f"  cs={val['physics_critic_sup_loss']:.4f}"
                      f"  ar={val['physics_ar_loss']:.4f}")
            print(f"         Experts   hm:{ms['heatmap']['mu_mean']:+.3f}/{ms['heatmap']['std_mean']:.3f}"
                  f"  occ:{ms['occupancy']['mu_mean']:+.3f}/{ms['occupancy']['std_mean']:.3f}"
                  f"  imp:{ms['impedance']['mu_mean']:+.3f}/{ms['impedance']['std_mean']:.3f}")
            print(f"         Fused     μ={val['mu_mean']:+.3f}±{val['mu_std']:.3f}"
                  f"  σ={val['std_mean']:.3f}  [{val['mu_min']:.2f},{val['mu_max']:.2f}]")

        if (epoch + 1) % 50 == 0:
            logger.log_latent_stats(log_path / "latent_stats.csv", epoch+1, beta, val)

        do_ckpt = (epoch + 1) % cfg.checkpoint_interval == 0
        do_best = val['total_loss'] < best_val
        if do_ckpt or do_best:
            latent_stats = VAETrainingLogger.build_latent_stats(val)
            per_K        = val.get('per_K_latent_stats', {})
            ckpt_dict    = _build_checkpoint(epoch+1, model, optimizer, tr['total_loss'],
                                              val, cfg, latent_stats, per_K,
                                              physics=physics, scheduler=scheduler)
            if do_ckpt:
                sp = ckpt_path / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save(ckpt_dict, sp); print(f"  Saved: {sp.name}")
                timing_path.write_text(json.dumps(
                    {'training_time_seconds': prev_seconds + int(time.time() - t0)}
                ))
                # Purge old epoch checkpoints beyond keep_last_n
                if cfg.keep_last_n_checkpoints > 0:
                    old = sorted(ckpt_path.glob("checkpoint_epoch_*.pt"),
                                 key=lambda p: int(p.stem.split('_')[-1]))
                    for old_ckpt in old[:-cfg.keep_last_n_checkpoints]:
                        old_ckpt.unlink()
                        print(f"  Removed old: {old_ckpt.name}")
            if do_best:
                best_val = val['total_loss']
                torch.save(ckpt_dict, ckpt_path / "best_model.pt")
                print(f"  New best → val={best_val:.4f}")

    # Final — always save the last epoch checkpoint
    elapsed  = int(time.time() - t0)
    total_s  = prev_seconds + elapsed
    h, rem   = divmod(total_s, 3600); m, s = divmod(rem, 60)
    time_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    latent_stats = VAETrainingLogger.build_latent_stats(val)
    per_K        = val.get('per_K_latent_stats', {})
    last_ckpt    = _build_checkpoint(cfg.num_epochs, model, optimizer, tr['total_loss'],
                                      val, cfg, latent_stats, per_K,
                                      physics=physics, scheduler=scheduler)
    torch.save(last_ckpt, ckpt_path / "last_model.pt")
    print(f"  Saved final: last_model.pt  (epoch {cfg.num_epochs}, val={val['total_loss']:.4f})")
    timing_path.write_text(json.dumps({'training_time_seconds': total_s}))

    logger.log_complete(best_val, ckpt_path, time_str)

    logger.plot(save_path=str(plots_path / "convergence_final.png"))
    logger.plot_loss_components(save_path=str(plots_path / "loss_components_final.png"))
    logger.plot_overfitting(save_path=str(plots_path / "overfitting_final.png")) # type: ignore
    logger.plot_physics(save_path=str(plots_path / "physics_losses_final.png")) # type: ignore
    logger.print_statistics()

    cfg_dict = asdict(cfg)
    config_data: dict[str, Any] = {
        'architecture':  {k: cfg_dict[k] for k in ('latent_dim','heatmap_private_dim','cond_dim')},
        'training':      {k: cfg_dict[k] for k in ('num_epochs','batch_size','learning_rate',
                                                     'lr_patience','lr_factor','lr_min',
                                                     'train_split','num_workers')},
        'recon_weights': {k: cfg_dict[k] for k in ('heatmap_weight','occupancy_weight',
                                                     'impedance_weight','impedance_deriv_weight',
                                                     'impedance_peak_weight','heatmap_peak_weight',
                                                     'heatmap_grad_weight','occupancy_focal_gamma')},
        'latent_reg':    {k: cfg_dict[k] for k in ('free_bits','mu_hinge_threshold','mu_hinge_weight',
                                                     'mu_bias_weight','per_expert_kl_weight',
                                                     'sigma_reg_weight','sigma_reg_target')},
        'kl_annealing':  {k: cfg_dict[k] for k in ('use_beta_annealing','beta_start_epoch',
                                                     'beta_end_epoch','beta_initial','beta_final',
                                                     'beta_phase2_final','beta_phase2_end_epoch')},
        'modality':      {k: cfg_dict[k] for k in ('modality_dropout','modality_dropout_start',
                                                     'modality_dropout_anneal_epochs',
                                                     'cross_modal_weight','cross_modal_update_freq')},
        'physics':       {k: cfg_dict[k] for k in ('physics_ri_weight','physics_critic_sup_weight',
                                                     'physics_ar_weight','physics_critic_warmup_epochs')},
        'performance':   {k: cfg_dict[k] for k in ('amp','tf32','compile','compile_mode',
                                                     'persistent_workers','prefetch_factor')},
        'model_params':  sum(p.numel() for p in model.parameters()),
    }
    logger.save_config(exp_path / "config.yaml", config_data)


if __name__ == "__main__":
    train_vae()
