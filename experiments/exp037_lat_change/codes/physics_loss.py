"""Physics-informed loss modules for exp031 Multi-Input VAE.

Exported symbols:
    PhysicsCritic      — occupancy grid → spatial effectiveness map (CNN)
    PhysicsLoss        — combined module: all critics + all loss methods
    _physics_stage_weights — staged training weight schedule

These are imported by train_vae_simple.py and by Latent_opm/latent_optimization_impedance.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Critic networks
# =============================================================================

class PhysicsCritic(nn.Module):
    """
    Small CNN that learns the spatial effectiveness field of decap occupancy.

    Replaces the manual Gaussian distance-decay kernel: the network learns the
    actual spread of decap influence from data, making the constraint adaptive.

    Input:  (B, 1, 7, 8)  — soft occupancy probability grid
    Output: (B, 1, 64, 64) — learned influence/effectiveness map in [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16,  1, kernel_size=1)

    def forward(self, occ_spatial: torch.Tensor) -> torch.Tensor:
        """occ_spatial: (B, 1, 7, 8) → effectiveness map (B, 1, 64, 64) in [0, 1]."""
        x = F.interpolate(occ_spatial, size=(64, 64), mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
        return torch.sigmoid(self.conv4(x))


# =============================================================================
# Combined PhysicsLoss
# =============================================================================

class PhysicsLoss(nn.Module):
    """
    Physics-informed constraint losses for the Multi-Modal VAE.

    Contains one learned network:
        critic — PhysicsCritic: occ grid → spatial effectiveness map

    Loss catalogue:
        critic_supervision_loss — trains critic on real (occ, hm) pairs
        radius_influence_loss   — penalizes VAE heatmap w/ no nearby decaps
        anti_resonance_loss     — every peak must be preceded by a dip
    """

    OCC_GRID_H: int = 7
    OCC_GRID_W: int = 8

    def __init__(
        self,
        background_value: float = -3.6228,
        fg_clip_min: float = -1.04,
    ):
        super().__init__()
        self.bg_threshold  = background_value + 0.5
        self.low_hm_thresh = fg_clip_min + 0.4

        self.critic = PhysicsCritic()

        self._register_occ_scatter()

    # ------------------------------------------------------------------
    # Buffer registration
    # ------------------------------------------------------------------

    def _register_occ_scatter(self) -> None:
        H, W = self.OCC_GRID_H, self.OCC_GRID_W
        INVALID = {(0, 3), (0, 4), (6, 3), (6, 4)}
        physical_layout: dict[tuple[int, int], str] = {
            (0, 0): "C4", (0, 1): "C5", (0, 2): "C6",
            (0, 5): "C1", (0, 6): "C2", (0, 7): "C3",
        }
        label_idx = 7
        for row in range(1, H):
            for col in range(W):
                if (row, col) not in INVALID:
                    physical_layout[(row, col)] = f"C{label_idx}"
                    label_idx += 1
        label_to_coord = {v: k for k, v in physical_layout.items()}
        scatter = torch.zeros(52, H * W)
        for i in range(52):
            label = f"C{i + 1}"
            if label in label_to_coord:
                r, c = label_to_coord[label]
                scatter[i, r * W + c] = 1.0
        self.register_buffer('occ_scatter', scatter)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _occ_to_spatial(self, occ_prob: torch.Tensor) -> torch.Tensor:
        """(B, 52) → (B, 1, 7, 8)."""
        scatter: torch.Tensor = self.occ_scatter  # type: ignore[assignment]
        return (occ_prob @ scatter).view(-1, 1, self.OCC_GRID_H, self.OCC_GRID_W)

    def _heatmap_to_critic_target(self, hm: torch.Tensor) -> torch.Tensor:
        """Invert + min-max normalise heatmap foreground → [0, 1] critic target.

        Low impedance (well-decoupled) → target ≈ 1.0.
        Background pixels → target = 0.
        """
        fg     = (hm > self.bg_threshold).float()
        hm_fg  = hm * fg
        _neg   = torch.finfo(hm.dtype).min
        _pos   = torch.finfo(hm.dtype).max
        hm_min = hm.masked_fill(fg == 0, _pos).flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
        hm_max = hm.masked_fill(fg == 0, _neg).flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
        hm_norm = (hm_fg - hm_min) / (hm_max - hm_min + 1e-8)
        return (1.0 - hm_norm) * fg

    # ------------------------------------------------------------------
    # Occ-critic losses
    # ------------------------------------------------------------------

    def critic_supervision_loss(
        self,
        target_occ: torch.Tensor,
        target_hm: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise PhysicsCritic on real (occ, hm) pairs."""
        occ_spatial   = self._occ_to_spatial(target_occ.float())
        pred_eff      = self.critic(occ_spatial)
        critic_target = self._heatmap_to_critic_target(target_hm)
        fg       = (target_hm > self.bg_threshold).float()
        fg_count = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return (((pred_eff - critic_target).pow(2) * fg).sum(dim=(1, 2, 3)) / fg_count).mean()

    def radius_influence_loss(
        self,
        recon_hm: torch.Tensor,
        occ_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize heatmap pixels with low impedance but low decap coverage."""
        occ_spatial   = self._occ_to_spatial(occ_prob)
        effectiveness = self.critic(occ_spatial).detach()
        fg       = (recon_hm > self.bg_threshold).float()
        low_hm   = F.relu(self.low_hm_thresh - recon_hm) * fg
        low_eff  = 1.0 - effectiveness
        fg_count = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
        return ((low_hm * low_eff).sum(dim=(1, 2, 3)) / fg_count).mean()

    # ------------------------------------------------------------------
    # Analytical spectrum losses
    # ------------------------------------------------------------------

    def anti_resonance_loss(self, recon_imp: torch.Tensor) -> torch.Tensor:
        """Every impedance peak must be preceded by a resonance dip (causality).

        Applied to channel 0 (raw log-Z) only.  The HF region (idx 190+) is
        monotonically increasing by construction (no peaks possible), so the loss
        only fires meaningfully in the LF region.
        """
        ch0    = recon_imp[:, 0:1, :]   # (B, 1, 231)
        left   = ch0[:, :, :-2]
        center = ch0[:, :, 1:-1]
        right  = ch0[:, :, 2:]
        peak_score = torch.minimum(F.relu(center - left), F.relu(center - right))
        dip_score  = torch.minimum(F.relu(left - center), F.relu(right - center))
        cum_dip    = torch.cummax(dip_score, dim=-1).values
        prior_dip  = torch.cat([torch.zeros_like(cum_dip[:, :, :1]), cum_dip[:, :, :-1]], dim=-1)
        return (peak_score * torch.exp(-10.0 * prior_dip)).mean()

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        recon_hm: torch.Tensor,
        recon_occ_logits: torch.Tensor,
        recon_imp: torch.Tensor,
        target_occ: torch.Tensor | None = None,
        target_hm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Returns 3 scalar losses: (ri, cs, ar)."""
        occ_prob = torch.sigmoid(recon_occ_logits)

        ri_loss = self.radius_influence_loss(recon_hm, occ_prob)
        ar_loss = self.anti_resonance_loss(recon_imp)

        cs_loss = recon_hm.new_zeros(())
        if target_occ is not None and target_hm is not None:
            cs_loss = self.critic_supervision_loss(target_occ, target_hm)

        return ri_loss, cs_loss, ar_loss


# =============================================================================
# Staged weight schedule
# =============================================================================

def _physics_stage_weights(epoch: int, cfg: object) -> dict[str, float]:
    """Return per-loss weight dict for the current epoch.

    Stages:
        1  (epoch < physics_critic_warmup_epochs):
            All three critics supervised; VAE constraints OFF.
        2  (physics_critic_warmup_epochs ≤ epoch < physics_stage3_start_epoch):
            All constraints active at base weights.
        3  (epoch ≥ physics_stage3_start_epoch):
            VAE-facing constraint weights ramp linearly to stage3_multiplier× by
            physics_stage3_end_epoch.  Supervision weights stay at base.
    """
    if epoch < cfg.physics_critic_warmup_epochs:  # type: ignore[union-attr]
        return dict(
            ri=0.0, cs=cfg.physics_critic_sup_weight, ar=0.0,  # type: ignore[union-attr]
        )

    scale = 1.0
    if epoch >= cfg.physics_stage3_start_epoch:  # type: ignore[union-attr]
        t = min(1.0, (epoch - cfg.physics_stage3_start_epoch) / # type: ignore
                max(1, cfg.physics_stage3_end_epoch - cfg.physics_stage3_start_epoch))  # type: ignore[union-attr]
        scale = 1.0 + t * (cfg.physics_stage3_multiplier - 1.0)  # type: ignore[union-attr]

    return dict(
        ri=cfg.physics_ri_weight * scale,    # type: ignore[union-attr]
        cs=cfg.physics_critic_sup_weight,    # type: ignore[union-attr]
        ar=cfg.physics_ar_weight * scale,    # type: ignore[union-attr]
    )
