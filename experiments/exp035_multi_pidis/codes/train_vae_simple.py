"""Training script — Multi-Input VAE (exp035_multi_pidis)."""
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

from experiments.exp035_multi_pidis.codes.vae_multi_input_simple import MultiInputVAE
from src_vae.others.dataloader import create_data_loaders
from src_vae.others.vae_logger import VAETrainingLogger


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Architecture ──────────────────────────────────────────────────────────
    latent_dim:          int   = 42
    heatmap_private_dim: int   = 8
    cond_dim:            int   = 8

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs:   int   = 700
    batch_size:   int   = 64
    learning_rate: float = 3e-4
    lr_patience:  int   = 15    # ReduceLROnPlateau patience (epochs)
    lr_factor:    float = 0.5   # LR multiplier on plateau
    lr_min:       float = 1e-5  # LR floor
    train_split:  float = 0.9
    num_workers:  int   = 4

    # ── K-balancing ───────────────────────────────────────────────────────────
    balance_k:           bool  = True
    k_balance_power:     float = 0.5
    k_balance_smoothing: float = 1e-3
    stratify_by_k:       bool  = True

    # ── Reconstruction weights ────────────────────────────────────────────────
    heatmap_weight:         float = 2.0
    occupancy_weight:       float = 4.0   # raised from 3→4: occ underfitting (train≈0.46, val≈0.51 at ep310)
    impedance_weight:       float = 3.0
    impedance_deriv_weight: float = 1.0
    # ── Peak-specific losses ──────────────────────────────────────────────────
    # In PDN design a missed/underestimated resonance peak = design failure.
    # Three complementary terms ensure all peaks are faithfully reproduced:
    #   topk_k            : supervise the N highest-amplitude peaks (not just max)
    #   topk_weight       : MSE weight for those top-K frequencies
    #   under_penalty     : extra cost when recon < target at ANY freq (asymmetric)
    #                       1.0 = symmetric, 2.0 = underestimate costs 2× more
    #   concavity_weight  : upweight Huber at all peak regions (d²<0 ⟹ local max)
    impedance_topk_k:           int   = 15   # supervise top-15 amplitude frequencies
    impedance_topk_weight:      float = 4.0  # MSE at those frequencies
    impedance_under_penalty:    float = 2.0  # asymmetric multiplier when recon < target
    impedance_concavity_weight: float = 2.0  # Huber weight at concave regions (-d²>0)
    heatmap_peak_weight:    float = 3.0
    heatmap_grad_weight:    float = 1.5
    # Anti-averaging sharpness losses — counteract VAE's tendency to blur heatmaps:
    #   lap_weight     : MSE on Laplacian residual (high-freq edge content)
    #   contrast_weight: margin loss pushing fg pixels above bg pixels by a margin
    heatmap_lap_weight:      float = 2.0
    heatmap_contrast_weight: float = 1.5
    heatmap_contrast_margin: float = 0.5  # minimum gap between fg and bg means (in z-score units)
    occupancy_focal_gamma:  float = 0.0   # disabled: pos_weight handles imbalance; gamma=2 kills gradient when predictions are uncertain (early training deadlock)
    occ_k_consistency_weight: float = 1.0  # target-aware margin: bounded by 2*margin per element, always agrees with BCE direction
    # ── K-aware loss weighting ────────────────────────────────────────────────
    use_k_weighting:    bool  = True
    occ_mid_k_center:   float = 25.0
    occ_mid_k_sigma:    float = 8.0
    occ_mid_k_boost:    float = 0.75
    hm_low_k_threshold: int   = 3
    hm_low_k_multiplier: float = 2.0

    # ── Latent regularisation ─────────────────────────────────────────────────
    free_bits:            float = 0.10   # raised: prevents KL near-zero that pulls sigma below target
    mu_hinge_threshold:   float = 4.0
    mu_hinge_weight:      float = 0.10
    mu_bias_weight:       float = 0.05
    per_expert_kl_weight: float = 0.02
    sigma_reg_weight:     float = 4.0   # raised: push posterior sigma toward target
    sigma_reg_target:     float = 0.65  # target posterior sigma (0.65 keeps posteriors inside N(0,1) prior cloud)

    # ── KL annealing (two-phase linear) ───────────────────────────────────────
    use_beta_annealing:    bool  = True
    beta_start_epoch:      int   = 0
    beta_end_frac:         float = 0.40   # fraction of num_epochs
    beta_initial:          float = 0.0
    beta_final:            float = 0.1
    beta_phase2_final:     float = 0.2
    beta_phase2_end_frac:  float = 0.80   # fraction of num_epochs
    # computed in __post_init__
    beta_end_epoch:        int   = 0
    beta_phase2_end_epoch: int   = 0

    # ── Modality dropout ──────────────────────────────────────────────────────
    modality_dropout:               float = 0.2
    modality_dropout_start:         float = 0.1
    modality_dropout_anneal_frac:   float = 0.20   # fraction of num_epochs
    modality_dropout_anneal_epochs: int   = 0       # computed in __post_init__

    # ── Cross-modal reconstruction ────────────────────────────────────────────
    cross_modal_weight:      float = 0.7
    cross_modal_update_freq: int   = 4   # run every N batches

    # ── Physics losses ────────────────────────────────────────────────────────
    physics_ri_weight:            float = 1.0
    physics_critic_sup_weight:    float = 2.0
    physics_ar_weight:            float = 0.5
    physics_fg_clip_min:          float = -1.04
    physics_critic_warmup_frac:   float = 0.10   # fraction of num_epochs
    physics_slope_anneal_frac:    float = 0.20   # fraction of num_epochs
    physics_critic_warmup_epochs: int   = 0       # computed in __post_init__
    physics_slope_anneal_epochs:  int   = 0       # computed in __post_init__

    # ── Z_eff 4th impedance channel ───────────────────────────────────────────
    decap_profiles_path: str = "/home/ubuntu/gan/configs/decap_profiles.npy"

    # ── Paths & checkpointing ─────────────────────────────────────────────────
    data_dir:            str = "/home/ubuntu/gan/datasets/data_multifreq_norm"
    experiment_dir:      str = "/home/ubuntu/gan/experiments/exp035_multi_pidis"
    checkpoint_interval: int = 50
    keep_last_n_checkpoints: int = 0   # keep N most recent epoch checkpoints (0 = keep all)
    resume_checkpoint:   int | str | None = 450
    # Set to an epoch number (e.g. 50) to resume from checkpoint_epoch_50.pt,
    # "best" to resume from best_model.pt, or None to start fresh.

    # ── Runtime overrides (set from normalization_stats.json) ─────────────────
    background_value: float = -3.6228

    # ── Performance ───────────────────────────────────────────────────────────
    amp:                str  = "off"            # off | auto | bf16 | fp16
    tf32:               bool = False
    compile:            bool = False
    compile_mode:       str  = "reduce-overhead"
    persistent_workers: bool = True
    prefetch_factor:    int  = 2

    # ── KAN impedance decoder anti-overfit ────────────────────────────────────
    # Spline L1 penalty penalises large spline weights; combined with a lower
    # LR and extra weight-decay on spline params this prevents the KAN from
    # memorising the dataset while keeping MLP paths unaffected.
    kan_spline_l1_weight: float = 1e-4   # added to total loss each step
    kan_spline_wd:        float = 1e-4   # AdamW weight-decay on spline params only
    kan_spline_lr_factor: float = 0.5    # spline LR = base_lr * this factor

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """Compute annealing epoch counts proportionally from num_epochs."""
        n = self.num_epochs
        self.beta_end_epoch             = round(n * self.beta_end_frac)
        self.beta_phase2_end_epoch      = round(n * self.beta_phase2_end_frac)
        self.modality_dropout_anneal_epochs = round(n * self.modality_dropout_anneal_frac)
        self.physics_critic_warmup_epochs   = round(n * self.physics_critic_warmup_frac)
        self.physics_slope_anneal_epochs    = round(n * self.physics_slope_anneal_frac)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS LOSS  (imported from physics_loss.py so all networks are trained and
#               their weights are saved in the checkpoint's physics_state_dict)
# ══════════════════════════════════════════════════════════════════════════════

from experiments.exp035_multi_pidis.codes.physics_loss import PhysicsLoss  # noqa: E402


def _z_eff_log(occ: torch.Tensor, decap_profiles: torch.Tensor) -> torch.Tensor:
    """Compute log(Z_eff) in log-Ohm space for a batch of occupancy vectors.

    Z_eff is the effective parallel impedance of all active decaps:
        Y_eff(f) = sum_i [ occ_i / |Z_i(f)| ]   (real-valued admittance sum)
        Z_eff(f) = 1 / Y_eff(f)

    Returning log(Z_eff) puts the result on the same perceptual scale as the
    log-impedance used throughout the model, so MSE is meaningful.

    Args:
        occ:            (B, 52) occupancy probabilities (or binary ground truth)
        decap_profiles: (52, 231) per-slot |Z_decap(f)| in Ohms
    Returns:
        (B, 231) log(Z_eff) in log-Ohm space
    """
    admittance = 1.0 / decap_profiles.clamp(min=1e-6)   # (52, 231)
    y_eff = occ @ admittance                              # (B, 231)
    z_eff = 1.0 / y_eff.clamp(min=1e-6)                 # (B, 231)
    return z_eff.clamp(1e-4, 100.0).log()               # (B, 231) log-Ohm


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
    'spline_l1_loss',
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


def heatmap_loss_per_sample(recon: torch.Tensor, target: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Foreground Huber + gradient + Laplacian sharpness + fg/bg contrast. Returns (B,).

    Laplacian term: MSE(lap(recon), lap(target)) on fg region.
    Penalises smooth blobs that have the right mean but wrong high-frequency
    content — the primary cause of the averaging effect in VAE heatmaps.

    Contrast term: hinge loss pushing mean(fg_recon) > mean(bg_recon) + margin.
    Prevents the model from "cheating" by raising the background to meet the
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
    lap_k = _LAP_KERNEL.to(recon.device)
    lap_r = F.conv2d(recon,  lap_k, padding=1)   # (B,1,H,W)
    lap_t = F.conv2d(target, lap_k, padding=1)
    lap_err  = (lap_r - lap_t).pow(2)
    lap_loss = (lap_err * fg).sum(dim=(1,2,3)) / n_fg              # (B,)

    # ── Foreground / background contrast margin ────────────────────────────
    n_bg         = bg_m.sum(dim=(1,2,3)).clamp(min=1.0)
    mean_fg      = (recon * fg).sum(dim=(1,2,3))  / n_fg           # (B,)
    mean_bg      = (recon * bg_m).sum(dim=(1,2,3)) / n_bg
    contrast_loss = F.relu(mean_bg + cfg.heatmap_contrast_margin - mean_fg)   # (B,)

    return (base
            + cfg.heatmap_peak_weight     * peak
            + cfg.heatmap_grad_weight     * (grad_x + grad_y)
            + cfg.heatmap_lap_weight      * lap_loss
            + cfg.heatmap_contrast_weight * contrast_loss)


def impedance_loss_per_sample(
    recon: torch.Tensor, target: torch.Tensor, cfg: Config,
    *, delta_raw: float = 1.0, delta_deriv: float = 1.0,
    imp_log_std: float = 1.0,
) -> torch.Tensor:
    """Multi-peak-aware impedance loss.  Returns (B,).

    Terms
    -----
    base_huber      : Huber on ch0 in log-Ohm space (broad shape).
    concavity_huber : Huber reweighted by -d² (peak bodies) — upweights ALL
                      local-max regions, not just the global max.
    d1_loss         : Huber on the 1st derivative (slope / peak sharpness).
    topk_asym_mse   : MSE at the top-K amplitude frequencies with an asymmetric
                      penalty: when recon < target the error is multiplied by
                      cfg.impedance_under_penalty, preventing regression-to-mean.
    """
    s     = imp_log_std
    ch0_r = recon[:, 0, :] * s          # (B, 231)
    ch0_t = target[:, 0, :] * s

    # ── Base Huber ─────────────────────────────────────────────────────────
    raw_err = F.huber_loss(ch0_r, ch0_t, delta=delta_raw, reduction='none')    # (B, 231)
    raw     = raw_err.mean(dim=1)                                               # (B,)

    # ── Concavity peak mask: upweight ALL peak bodies ───────────────────────
    d2          = torch.diff(ch0_t, n=2, dim=-1)                               # (B, 229)
    concav_w    = (-d2).clamp(min=0.0)                                         # (B, 229)
    concav_norm = concav_w / (concav_w.mean(dim=1, keepdim=True) + 1e-6)       # (B, 229)
    concavity_loss = (raw_err[:, 1:-1] * concav_norm).mean(dim=1)              # (B,)

    # ── Derivative supervision ─────────────────────────────────────────────
    d1_r  = torch.diff(ch0_r, dim=-1)                                          # (B, 230)
    d1_t  = torch.diff(ch0_t, dim=-1)
    d1_loss = F.huber_loss(d1_r, d1_t, delta=delta_deriv, reduction='none').mean(dim=1)

    # ── Top-K multi-peak supervision with asymmetric underestimate penalty ──
    K_top   = min(cfg.impedance_topk_k, ch0_t.shape[-1])
    _, topk_idx = ch0_t.topk(K_top, dim=-1)                                    # (B, K_top)
    r_at_peaks  = ch0_r.gather(1, topk_idx)                                    # (B, K_top)
    t_at_peaks  = ch0_t.gather(1, topk_idx)                                    # (B, K_top)
    err_at_peaks = (r_at_peaks - t_at_peaks).pow(2)                            # (B, K_top)
    under_mask   = (r_at_peaks < t_at_peaks).float()
    asym_weight  = 1.0 + (cfg.impedance_under_penalty - 1.0) * under_mask      # (B, K_top)
    topk_loss    = (err_at_peaks * asym_weight).mean(dim=1)                    # (B,)

    return (raw
            + cfg.impedance_concavity_weight * concavity_loss
            + cfg.impedance_deriv_weight     * d1_loss
            + cfg.impedance_topk_weight      * topk_loss)


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
    expert_stats=None, *,
    K: torch.Tensor | None = None,
    apply_k_weights: bool = True,
    physics: 'PhysicsLoss | None' = None,
    physics_weights: tuple[float, ...] | None = None,
    imp_log_std: float = 1.0,
    decap_profiles: 'torch.Tensor | None' = None,
) -> dict:
    # Reconstruction
    hm_ps  = heatmap_loss_per_sample(recon_hm, target_hm, cfg)
    # No dynamic pos_weight: per-sample pw=(52-K)/K applied uniformly across all 52 slots
    # sets BCE equilibrium at sigmoid=K/52≈0.48 for every slot, causing gradient cancellation
    # and trapping the model at near-uniform logit output. K-consistency (weight=0.1) + plain
    # BCE is sufficient to learn which slots to activate.
    occ_ps = focal_bce_per_sample(recon_occ, target_occ, gamma=cfg.occupancy_focal_gamma)
    imp_ps = impedance_loss_per_sample(recon_imp, target_imp, cfg, imp_log_std=imp_log_std)

    if apply_k_weights and cfg.use_k_weighting and K is not None:
        w_hm, w_occ = _k_loss_weights(K, cfg)
        loss_hm, loss_occ = _weighted_mean(hm_ps, w_hm), _weighted_mean(occ_ps, w_occ)
    else:
        loss_hm, loss_occ = hm_ps.mean(), occ_ps.mean()
    loss_imp = imp_ps.mean()

    # Target-aware margin loss: active slots pushed above +margin, inactive below -margin.
    # Unlike sorted-threshold hinge, this uses ground-truth labels so gradients always
    # agree with BCE (both push active↑ and inactive↓).  Bounded by 2*margin per element.
    occ_k_loss = mu.new_zeros(())
    if cfg.occ_k_consistency_weight > 0 and K is not None:
        margin = 0.5
        t = target_occ.clamp(0.0, 1.0)
        active_margin   = F.relu(margin - recon_occ) * t          # penalise active slots with logit < +0.5
        inactive_margin = F.relu(recon_occ + margin) * (1.0 - t)  # penalise inactive slots with logit > -0.5
        occ_k_loss = (active_margin + inactive_margin).mean()

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
        spline_l1_loss=z(()),   # placeholder; real value wired in train_epoch
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


def _compute_z_eff(occ: torch.Tensor, decap_profiles: torch.Tensor,
                   imp_log_mean: float, imp_log_std: float) -> torch.Tensor:
    """(B,52) + (52,231) profiles → (B,1,231) normalized log Z_eff."""
    z_eff = 1.0 / (occ @ (1.0 / decap_profiles.clamp(min=1e-6))).clamp(min=1e-6)
    return ((z_eff.clamp(1e-4, 100.0).log() - imp_log_mean) / imp_log_std).unsqueeze(1)


def _prepare_batch(batch: dict, cfg: Config,
                   decap_profiles=None, imp_log_mean=0.0, imp_log_std=1.0):
    nb  = _device_type(cfg.device) == 'cuda'
    hm  = batch['heatmap_norm'].to(cfg.device, non_blocking=nb)
    occ = batch['occupancy'].to(cfg.device, non_blocking=nb)
    imp = batch['impedance'].to(cfg.device, non_blocking=nb)
    K   = batch['K'].to(cfg.device, non_blocking=nb)
    # PI_freq: log10-normalised frequency scalar in [0,1].
    # Falls back to 0.828 (~200 MHz) if the dataset has no PI_freq directory.
    if 'PI_freq' in batch:
        PI_freq = batch['PI_freq'].to(cfg.device, non_blocking=nb)
    else:
        PI_freq = torch.full((hm.shape[0],), 0.828, dtype=torch.float32, device=cfg.device)
    if hm.dim()  == 3: hm  = hm.unsqueeze(1)
    if imp.dim() == 1: imp = imp.unsqueeze(0)
    imp = imp[:, :2, :]  # keep ch0 + d1 only (drop d2)
    hm_enc = hm.masked_fill(hm < cfg.background_value + 0.5, 0.0)
    if decap_profiles is not None:
        imp = torch.cat([imp, _compute_z_eff(occ, decap_profiles, imp_log_mean, imp_log_std)], dim=1)
    return hm, hm_enc, occ, imp, K, PI_freq


def load_checkpoint(path: str, model, optimizer=None, device='cuda', physics=None):
    print(f"\n{'='*60}\nRESUMING FROM: {path}\n{'='*60}")
    ckpt = torch.load(path, map_location=device)
    ckpt_sd  = ckpt['model_state_dict']
    model_sd = model.state_dict()
    # Keep only keys that exist in both and have matching shapes
    compatible = {k: v for k, v in ckpt_sd.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
    skipped = [k for k in ckpt_sd if k not in compatible]
    new_keys = [k for k in model_sd if k not in ckpt_sd]
    model_sd.update(compatible)
    model.load_state_dict(model_sd)
    print(f"  Loaded {len(compatible)} matching tensors from checkpoint.")
    if skipped:
        print(f"  Skipped {len(skipped)} mismatched/removed keys — e.g. {skipped[:3]}")
    if new_keys:
        print(f"  {len(new_keys)} new keys randomly initialised — e.g. {new_keys[:3]}")
    if optimizer and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError as e:
            print(f"  Warning: optimizer state not loaded ({e}). "
                  "Optimizer will start fresh (param groups changed).")
    if physics:
        if 'physics_state_dict' in ckpt:
            physics.load_state_dict(ckpt['physics_state_dict'], strict=False)
            print("  Loaded physics networks (critic, imp_critic, k_imp_mlp).")
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


def _cross_modal_loss(model, hm_enc, occ, imp, hm, K, PI_freq, cfg: Config, imp_log_std: float = 1.0) -> torch.Tensor:
    """Cross-modal reconstruction loss (heatmap + impedance sources), averaged."""
    w_hm_cm, w_occ_cm = _k_loss_weights(K, cfg) if cfg.use_k_weighting else (None, None)
    total = hm.new_zeros(())
    for source in ('heatmap', 'impedance'):
        z_cm            = model.encode_cross_modal(source, K, PI_freq, heatmap=hm_enc, impedance=imp)
        r_hm, r_occ, r_imp = model.decode(z_cm, K, PI_freq)
        hm_ps   = heatmap_loss_per_sample(r_hm, hm, cfg)
        occ_ps  = focal_bce_per_sample(r_occ, occ, gamma=cfg.occupancy_focal_gamma)
        imp_ps  = impedance_loss_per_sample(r_imp, imp, cfg, delta_raw=1.0, delta_deriv=2.0, imp_log_std=imp_log_std)
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
                decap_profiles=None, imp_log_mean=0.0, imp_log_std=1.0,
                scaler=None) -> dict:
    model.train()
    if physics is not None: physics.train()
    model.modality_dropout = modality_dropout

    loss_acc        = {k: 0.0 for k in LOSS_KEYS}
    scalar, per_mod = _init_trackers()
    amp_dtype       = _autocast_dtype(cfg)
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    autocast  = torch.autocast(device_type=_device_type(cfg.device),
                                dtype=amp_dtype, enabled=(amp_dtype is not None))
    cm_freq   = max(1, cfg.cross_modal_update_freq)

    for batch_idx, batch in enumerate(loader):
        hm, hm_enc, occ, imp, K, PI_freq = _prepare_batch(batch, cfg, decap_profiles, imp_log_mean, imp_log_std)

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            recon_hm, recon_occ, recon_imp, mu, logvar, expert_stats = model(hm_enc, occ, imp, K, PI_freq)
            losses = vae_loss(recon_hm, recon_occ, recon_imp, hm, occ, imp,
                              mu, logvar, beta, cfg, expert_stats,
                              K=K, apply_k_weights=True,
                              physics=physics, physics_weights=physics_weights,
                              imp_log_std=imp_log_std)
            if cfg.cross_modal_weight > 0 and batch_idx % cm_freq == 0:
                losses['total_loss'] = losses['total_loss'] + cfg.cross_modal_weight * _cross_modal_loss(
                    model, hm_enc, occ, imp, hm, K, PI_freq, cfg, imp_log_std=imp_log_std
                )

            # KAN spline L1 regularisation (only impedance decoder has KAN)
            spline_reg = torch.tensor(0.0, device=cfg.device)
            if cfg.kan_spline_l1_weight > 0 and hasattr(model, 'spline_l1'):
                raw_model = getattr(model, '_orig_mod', model)  # unwrap torch.compile
                spline_reg = raw_model.spline_l1()
                losses['total_loss'] = losses['total_loss'] + cfg.kan_spline_l1_weight * spline_reg
            losses['spline_l1_loss'] = spline_reg

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
          f"  kl={result['kl_loss']:.4f}  hm={result['heatmap_loss']:.4f}"
          f"  imp={result['impedance_loss']:.4f}  occ={result['occupancy_loss']:.4f}"
          f"  ri={result['physics_ri_loss']:.4f}",
          flush=True)
    mu_s = _finalize_scalar(scalar['mu'],     n)
    lv_s = _finalize_scalar(scalar['logvar'], n)
    result.update(mu_mean=mu_s['mean'], mu_std=mu_s['std'], mu_min=mu_s['min'], mu_max=mu_s['max'],
                  logvar_mean=lv_s['mean'], logvar_std=lv_s['std'],
                  logvar_min=lv_s['min'],   logvar_max=lv_s['max'])
    result['modality_stats'] = _finalize_modality_stats(per_mod, n)
    return result


def validate(model, loader, cfg: Config, beta: float,
             physics=None, physics_weights=None,
             decap_profiles=None, imp_log_mean=0.0, imp_log_std=1.0) -> dict:
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
            hm, hm_enc, occ, imp, K, PI_freq = _prepare_batch(batch, cfg, decap_profiles, imp_log_mean, imp_log_std)
            with autocast:
                recon_hm, recon_occ, recon_imp, mu, logvar, expert_stats = model(hm_enc, occ, imp, K, PI_freq)
                losses = vae_loss(recon_hm, recon_occ, recon_imp, hm, occ, imp,
                                  mu, logvar, beta, cfg, expert_stats,
                                  K=K, apply_k_weights=False,
                                  physics=physics, physics_weights=physics_weights,
                                  imp_log_std=imp_log_std,
                                  decap_profiles=decap_profiles)

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

def _resolve_checkpoint(cfg: Config) -> None:
    """Normalise cfg.resume_checkpoint to an absolute path string (or None).

    Accepts:
      None          → fresh training run
      int           → checkpoint_epoch_{N}.pt  (e.g. 50)
      "best"        → best_model.pt
      str (path)    → used as-is (backwards compat)
    """
    rc = cfg.resume_checkpoint
    if rc is None:
        return
    ckpt_dir = Path(cfg.experiment_dir) / "checkpoints"
    if isinstance(rc, int):
        cfg.resume_checkpoint = str(ckpt_dir / f"checkpoint_epoch_{rc}.pt")
    elif isinstance(rc, str) and rc.lower() == "best":
        cfg.resume_checkpoint = str(ckpt_dir / "best_model.pt")
    # else: already a path string — leave unchanged


def train_vae():
    cfg = Config()
    _resolve_checkpoint(cfg)   # int / "best" → full path string

    exp_path     = Path(cfg.experiment_dir)
    ckpt_path    = exp_path / "checkpoints"
    log_path     = exp_path / "logs"
    metrics_path = exp_path / "metrics"
    plots_path   = metrics_path / "plots"

    if not cfg.resume_checkpoint:
        import shutil
        for p in (log_path, metrics_path):   # keep checkpoints as backup; new run overwrites as it goes
            if p.exists(): shutil.rmtree(p)
    for p in (exp_path, ckpt_path, log_path, metrics_path, plots_path):
        p.mkdir(parents=True, exist_ok=True)

    logger     = VAETrainingLogger(log_dir=str(log_path), checkpoint_dir=str(ckpt_path),
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
    imp_log_mean = imp_log_std = 0.0
    if stats_path.exists():
        raw = json.loads(stats_path.read_text())
        cfg.background_value    = raw['background_value']
        cfg.physics_fg_clip_min = raw['Heatmap']['clip_min']
        imp_log_mean = float(raw['Impedance']['log_mean'])
        imp_log_std  = float(raw['Impedance']['log_std']) or 1.0

    # Decap profiles
    decap_profiles: torch.Tensor | None = None
    dp = Path(cfg.decap_profiles_path)
    if cfg.decap_profiles_path and dp.exists():
        decap_profiles = torch.from_numpy(np.load(str(dp)).astype(np.float32)).to(cfg.device)
        print(f"Decap profiles: {dp.name}  {tuple(decap_profiles.shape)}  → 4-ch impedance")
    else:
        print("Decap profiles not found — using 3-channel impedance.")

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

    # Split AdamW optimizer: spline params get lower LR + weight decay to prevent overfit;
    # all other params get standard LR and no weight decay (KAN base weights + everything else).
    from experiments.exp035_multi_pidis.codes.vae_multi_input_simple import FastKANLayer
    spline_ids = {id(p) for m in model.modules() if isinstance(m, FastKANLayer)
                  for p in [m.spline_weight]}
    spline_params = [p for p in model.parameters() if id(p) in spline_ids]
    other_params  = [p for p in model.parameters() if id(p) not in spline_ids]
    optimizer = optim.AdamW([
        {'params': other_params,  'lr': cfg.learning_rate, 'weight_decay': 0.0},
        {'params': spline_params, 'lr': cfg.learning_rate * cfg.kan_spline_lr_factor,
         'weight_decay': cfg.kan_spline_wd},
    ])

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

    # Resume  (_resolve_checkpoint guarantees resume_checkpoint is str | None by here)
    start_epoch, best_val = 0, float('inf')
    _ckpt = cfg.resume_checkpoint
    if isinstance(_ckpt, str) and Path(_ckpt).exists():
        start_epoch, best_val = load_checkpoint(_ckpt, model, optimizer,
                                                 cfg.device, physics=physics)
        tmp = torch.load(_ckpt, map_location='cpu')
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
    grad_scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    t0          = time.time()

    print(f"\nTraining epochs {start_epoch+1}–{cfg.num_epochs}...\n")

    for epoch in range(start_epoch, cfg.num_epochs):
        beta = compute_beta(epoch, cfg)
        md   = compute_modality_dropout(epoch, cfg)
        pw   = _physics_stage_weights(epoch, cfg) if physics is not None else None

        tr  = train_epoch(model, train_loader, optimizer, cfg, epoch, beta, md,
                          physics=physics, physics_weights=pw,
                          decap_profiles=decap_profiles,
                          imp_log_mean=imp_log_mean, imp_log_std=imp_log_std,
                          scaler=grad_scaler)
        val = validate(model, val_loader, cfg, beta,
                       physics=physics, physics_weights=pw,
                       decap_profiles=decap_profiles,
                       imp_log_mean=imp_log_mean, imp_log_std=imp_log_std)

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
    logger.plot_overfitting(save_path=str(plots_path / "overfitting_final.png"))
    logger.plot_physics(save_path=str(plots_path / "physics_losses_final.png"))
    logger.print_statistics()

    cfg_dict = asdict(cfg)
    config_data: dict[str, Any] = {
        'architecture':  {k: cfg_dict[k] for k in ('latent_dim','heatmap_private_dim','cond_dim')},
        'training':      {k: cfg_dict[k] for k in ('num_epochs','batch_size','learning_rate',
                                                     'lr_patience','lr_factor','lr_min',
                                                     'train_split','num_workers')},
        'recon_weights': {k: cfg_dict[k] for k in ('heatmap_weight','occupancy_weight',
                                                     'impedance_weight','impedance_deriv_weight',
                                                     'impedance_topk_k','impedance_topk_weight',
                                                     'impedance_under_penalty','impedance_concavity_weight',
                                                     'heatmap_peak_weight','heatmap_grad_weight',
                                                     'heatmap_lap_weight','heatmap_contrast_weight',
                                                     'occupancy_focal_gamma')},
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
