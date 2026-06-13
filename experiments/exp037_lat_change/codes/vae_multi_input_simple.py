import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

Dr_value    = 0.1   # Global dropout rate
IMP_DROPOUT = 0.1   # Dropout in impedance decoder


class DeepMLP(nn.Module):
    """Deep MLP with LayerNorm, SiLU, dropout, and residual skip connections.

    A residual skip is added whenever two consecutive layers share the same
    width, improving gradient flow in deep stacks.
    """

    def __init__(self, dims: list, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.skips  = nn.ModuleList()
        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)
            block_layers: list = [nn.Linear(dims[i], dims[i + 1])]
            if not is_last:
                block_layers += [nn.LayerNorm(dims[i + 1]), nn.SiLU()]
                if dropout > 0.0:
                    block_layers.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*block_layers))
            # Identity skip when dims match; learned 1×1 projection otherwise
            if dims[i] == dims[i + 1] and not is_last:
                self.skips.append(nn.Identity())
            else:
                self.skips.append(None)  # type: ignore[arg-type]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, skip in zip(self.blocks, self.skips):
            out = block(x)
            if skip is not None:
                out = out + skip(x)
            x = out
        return x


class SelfAttn2d(nn.Module):
    """Lightweight spatial self-attention for 2D feature maps (B, C, H, W)."""
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        seq = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        out, _ = self.attn(seq, seq, seq)
        return x + out.permute(0, 2, 1).view(B, C, H, W)


class MultiInputVAE(nn.Module):
    """
    Multi-modal VAE with a shared latent space + heatmap-private tail dims.

    - Heatmap expert predicts the full latent vector (shared + private).
    - Occupancy/impedance experts predict only the shared part; the private
      dims are forced to the unit-Gaussian prior (mu=0, logvar=0).
    - Decoder wiring:
        * Heatmap decoder sees full z (shared + private).
        * Occupancy/impedance decoders see only shared z.

    This implements "heatmap private dimensions" while keeping the overall
    latent size constant (e.g. total=32, heatmap_private=8, shared=24).
    """
    def __init__(
        self,
        latent_dim: int = 32,
        modality_dropout: float = 0.5,
        cond_dim: int = 8,
        heatmap_private_dim: int = 8,
    ):
        super().__init__()
        if heatmap_private_dim < 0:
            raise ValueError(f"heatmap_private_dim must be >= 0, got {heatmap_private_dim}")
        if heatmap_private_dim >= latent_dim:
            raise ValueError(
                f"heatmap_private_dim must be < latent_dim (total). Got heatmap_private_dim={heatmap_private_dim}, latent_dim={latent_dim}"
            )

        self.latent_dim          = latent_dim
        self.heatmap_private_dim = heatmap_private_dim
        self.shared_latent_dim   = latent_dim - heatmap_private_dim
        self.modality_dropout = modality_dropout
        self.cond_dim         = cond_dim

        # K conditioning: K = number of 1s in occupancy vector, K ∈ {0, ..., 52}
        self.k_embedding = nn.Embedding(53, cond_dim)

        # ── Encoders ─────────────────────────────────────────────────────────
        self.heatmap_enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
        )
        self.heatmap_enc_attn = SelfAttn2d(32, num_heads=1)
        self.heatmap_enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(512, 64),
        )

        self.occupancy_encoder = nn.Sequential(
            nn.Linear(52, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
        )

        # Impedance input: (B, 1, 231) → flatten to 231 (log-z-score PI spectrum)
        self.impedance_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(231, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(Dr_value),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
        )

        # ── Latent projections (PoE experts) ────────────────────────────────
        # Heatmap predicts full z; occupancy/impedance predict shared only.
        enc_in = 64 + cond_dim
        self.heatmap_mu       = nn.Linear(enc_in, latent_dim)
        self.heatmap_logvar   = nn.Linear(enc_in, latent_dim)
        self.occupancy_mu     = nn.Linear(enc_in, self.shared_latent_dim)
        self.occupancy_logvar = nn.Linear(enc_in, self.shared_latent_dim)
        self.impedance_mu     = nn.Linear(enc_in, self.shared_latent_dim)
        self.impedance_logvar = nn.Linear(enc_in, self.shared_latent_dim)

        # ── Decoders ────────────────────────────────────────────────────────
        # Heatmap decoder consumes full z; occupancy/impedance consume shared z.
        dec_in_heatmap = latent_dim + cond_dim
        dec_in_shared  = self.shared_latent_dim + cond_dim

        self.heatmap_fc = nn.Sequential(
            nn.Linear(dec_in_heatmap, 512),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(512, 128 * 4 * 4),
            nn.LayerNorm(128 * 4 * 4),
            nn.LeakyReLU(),
        )
        self.heatmap_dec_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
        )
        self.heatmap_dec_attn = SelfAttn2d(32, num_heads=1)
        self.heatmap_dec_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

        self.occupancy_decoder = nn.Sequential(
            nn.Linear(dec_in_shared, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),   # second dropout — prevents memorising sparse training placements
            nn.Linear(256, 52),
            # No Sigmoid here — BCEWithLogitsLoss is used in training (numerically stable)
            # and sigmoid is applied in inference() before returning probabilities
        )

        self.impedance_decoder = DeepMLP(
            [dec_in_shared, 256, 512, 1024, 1024, 512, 256, 231], dropout=IMP_DROPOUT
        )

    def _pad_shared_to_full(self, mu_shared: torch.Tensor, logvar_shared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad shared (B, shared_latent_dim) expert stats to full (B, latent_dim).

        Private dims are set to unit Gaussian prior: mu=0, logvar=0.
        """
        if self.heatmap_private_dim == 0:
            return mu_shared, logvar_shared
        B = mu_shared.shape[0]
        zeros_mu = torch.zeros(B, self.heatmap_private_dim, device=mu_shared.device, dtype=mu_shared.dtype)
        zeros_lv = torch.zeros(B, self.heatmap_private_dim, device=logvar_shared.device, dtype=logvar_shared.dtype)
        mu_full = torch.cat([mu_shared, zeros_mu], dim=1)
        lv_full = torch.cat([logvar_shared, zeros_lv], dim=1)
        return mu_full, lv_full

    # ── PoE fusion ────────────────────────────────────────────────────────────

    def product_of_experts(self, mu_list, logvar_list, eps=1e-8):
        """
        Combine Gaussian experts via Product of Experts.
        A unit-Gaussian prior (mu=0, logvar=0) is always appended as an anchor.
        """
        prior_mu     = torch.zeros_like(mu_list[0])
        prior_logvar = torch.zeros_like(logvar_list[0])
        mu_list      = mu_list      + [prior_mu]
        logvar_list  = logvar_list  + [prior_logvar]

        var_list           = [torch.exp(lv) + eps for lv in logvar_list]
        precision_list     = [1.0 / v for v in var_list]
        combined_precision = torch.stack(precision_list).sum(dim=0)
        combined_var       = 1.0 / (combined_precision + eps)
        combined_mu        = combined_var * torch.stack(
            [mu * p for mu, p in zip(mu_list, precision_list)]
        ).sum(dim=0)
        return combined_mu, torch.log(combined_var + eps)

    # ── Core encode / decode / reparameterize ─────────────────────────────────

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, min=-4.0, max=2.0))
        return mu + torch.randn_like(std) * std

    def encode(self, heatmap, occupancy, impedance, K):
        """
        Encode all three modalities, fuse via PoE, return z + stats.

        Returns:
            z          (B, latent_dim)
            mu         (B, latent_dim)   — fused posterior mean
            logvar     (B, latent_dim)   — fused posterior log-variance
            expert_stats  dict — pre-PoE (mu, logvar) per modality (for monitoring)
        """
        heatmap_feat   = self.heatmap_enc2(self.heatmap_enc_attn(self.heatmap_enc1(heatmap)))
        occupancy_feat = self.occupancy_encoder(occupancy)
        impedance_feat = self.impedance_encoder(impedance)

        k_emb  = self.k_embedding(K)
        hm_c   = torch.cat([heatmap_feat,   k_emb], dim=1)
        occ_c  = torch.cat([occupancy_feat, k_emb], dim=1)
        imp_c  = torch.cat([impedance_feat, k_emb], dim=1)

        # Per-modality expert predictions (pre-PoE, pre-dropout)
        # Clamp expert logvars before PoE to prevent variance explosion.
        # Without this, an encoder can learn exp(logvar)→∞ → precision≈0 →
        # near-zero PoE gradient → unbounded runaway.
        _EXP_LV_MIN, _EXP_LV_MAX = -6.0, 1.0   # cap expert σ ≤ 1.65 to prevent experts opting out of PoE fusion
        hm_mu = self.heatmap_mu(hm_c)
        hm_lv = self.heatmap_logvar(hm_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)

        occ_mu_s = self.occupancy_mu(occ_c)
        occ_lv_s = self.occupancy_logvar(occ_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)
        occ_mu, occ_lv = self._pad_shared_to_full(occ_mu_s, occ_lv_s)

        imp_mu_s = self.impedance_mu(imp_c)
        imp_lv_s = self.impedance_logvar(imp_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)
        imp_mu, imp_lv = self._pad_shared_to_full(imp_mu_s, imp_lv_s)

        # Expert stats before dropout (for monitoring only — not used in loss)
        expert_stats = {
            'heatmap':   (hm_mu,  hm_lv),
            'occupancy': (occ_mu, occ_lv),
            'impedance': (imp_mu, imp_lv),
        }

        mu_list     = [hm_mu,  occ_mu,  imp_mu]
        logvar_list = [hm_lv,  occ_lv,  imp_lv]

        # Modality dropout: randomly replace some experts with prior (keep at least one)
        if self.training and self.modality_dropout > 0:
            # Vectorized: single GPU call instead of 3 CPU-GPU syncs
            keep = torch.bernoulli(
                torch.full((3,), 1.0 - self.modality_dropout)
            ).bool()
            if not keep.any():
                keep[torch.randint(3, (1,))] = True
            prior_mu = torch.zeros_like(mu_list[0])
            prior_lv = torch.zeros_like(logvar_list[0])
            for i in range(3):
                if not keep[i].item():
                    mu_list[i]     = prior_mu
                    logvar_list[i] = prior_lv

        mu, logvar = self.product_of_experts(mu_list, logvar_list)
        logvar     = torch.clamp(logvar, min=-4.0, max=2.0)
        z          = self._reparameterize(mu, logvar)

        return z, mu, logvar, expert_stats

    def decode(self, z, K):
        k_emb = self.k_embedding(K)
        z_shared = z[:, : self.shared_latent_dim]
        dec_hm   = torch.cat([z,        k_emb], dim=1)
        dec_sh   = torch.cat([z_shared, k_emb], dim=1)

        hm_feat  = self.heatmap_fc(dec_hm)
        hm_feat  = self.heatmap_dec_deconv1(hm_feat.view(-1, 128, 4, 4))
        hm_recon = self.heatmap_dec_deconv2(self.heatmap_dec_attn(hm_feat))

        occ_recon = self.occupancy_decoder(dec_sh)          # (B, 52) logits

        imp_recon = self.impedance_decoder(dec_sh).unsqueeze(1)  # (B, 1, 231)

        return hm_recon, occ_recon, imp_recon

    def decode_impedance(self, z: torch.Tensor, K) -> torch.Tensor:
        """Decode only impedance from latent z and K.

        Useful for impedance-guided latent optimization (faster than full decode).

        Args:
            z: (B, latent_dim) or (latent_dim,)
            K: int or tensor broadcastable to (B,)

        Returns:
            imp_recon: (B, 1, 231) (or (1,231) if input z was 1D)
        """
        squeeze = False
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        if z.dim() != 2:
            raise ValueError(f"Expected z shape (B, latent_dim) or (latent_dim,), got {tuple(z.shape)}")

        B = z.shape[0]
        if isinstance(K, int):
            K_tensor = torch.full((B,), K, dtype=torch.long, device=z.device)
        elif isinstance(K, torch.Tensor):
            K_tensor = K.to(device=z.device, dtype=torch.long)
            if K_tensor.dim() == 0:
                K_tensor = K_tensor.expand(B)
            else:
                K_tensor = K_tensor.view(-1)
            if K_tensor.shape[0] != B:
                raise ValueError(f"K has shape {tuple(K_tensor.shape)} but z batch size is {B}")
        else:
            raise TypeError(f"K must be int or torch.Tensor, got {type(K).__name__}")

        k_emb = self.k_embedding(K_tensor)
        z_shared = z[:, : self.shared_latent_dim]
        dec      = torch.cat([z_shared, k_emb], dim=1)

        imp_recon = self.impedance_decoder(dec).unsqueeze(1)  # (B, 1, 231)
        return imp_recon[0] if squeeze else imp_recon


    def spline_l1(self) -> torch.Tensor:
        """Returns zero (no spline weights; kept for train-script API compatibility)."""
        return torch.tensor(0.0)

    def encode_cross_modal(self, source: str, K, heatmap=None, impedance=None):
        """
        Encode from a single modality (PoE with that one expert + prior).
        Used for cross-modal reconstruction loss during training.
        """
        k_emb = self.k_embedding(K)

        if source == 'heatmap':
            feat   = self.heatmap_enc2(self.heatmap_enc_attn(self.heatmap_enc1(heatmap)))
            feat_c = torch.cat([feat, k_emb], dim=1)
            mu_s   = self.heatmap_mu(feat_c)
            lv_s   = self.heatmap_logvar(feat_c).clamp(-6.0, 1.0)
        elif source == 'impedance':
            if impedance is None:
                raise ValueError("impedance must be provided when source='impedance'")
            imp_in = impedance
            if imp_in.dim() == 1:
                imp_in = imp_in.unsqueeze(0).unsqueeze(1)
            elif imp_in.dim() == 2:
                if imp_in.shape[1] == 231:
                    imp_in = imp_in.unsqueeze(1)
                elif imp_in.shape[1] % 231 == 0:
                    imp_in = imp_in.reshape(-1, 1, 231)
                else:
                    raise ValueError(
                        f"Expected impedance (B,231) or (B,1,231); got {tuple(imp_in.shape)}"
                    )
            elif imp_in.dim() == 3:
                if imp_in.shape[2] != 231:
                    raise ValueError(f"Expected last dim 231; got {tuple(imp_in.shape)}")
                imp_in = imp_in[:, :1, :]
            else:
                raise ValueError(f"Expected impedance (B,1,231); got {tuple(imp_in.shape)}")

            feat   = self.impedance_encoder(imp_in)
            feat_c = torch.cat([feat, k_emb], dim=1)
            mu_s   = self.impedance_mu(feat_c)
            lv_s   = self.impedance_logvar(feat_c).clamp(-6.0, 1.0)
            mu_s, lv_s = self._pad_shared_to_full(mu_s, lv_s)
        else:
            raise ValueError(f"Unsupported source: {source!r}. Use 'heatmap' or 'impedance'.")

        mu, logvar = self.product_of_experts([mu_s], [lv_s])
        z = self._reparameterize(mu, torch.clamp(logvar, -4.0, 2.0))
        return z

    def forward(self, heatmap, occupancy, impedance, K):
        z, mu, logvar, expert_stats = self.encode(heatmap, occupancy, impedance, K)
        hm_recon, occ_recon, imp_recon = self.decode(z, K)
        return hm_recon, occ_recon, imp_recon, mu, logvar, expert_stats

    def inference(self, num_samples, device, K, latent_stats=None,
                  per_K_latent_stats=None, shared_temp: float = 1.0):
        """
        Sample from the aggregate posterior marginal conditioned on K.

        Args:
            num_samples:         number of samples to generate
            device:              torch device
            K:                   int or 1-D LongTensor of length num_samples
            latent_stats:        dict from checkpoint (key 'latent' with per-dim stats)
            per_K_latent_stats:  K-conditioned per-dim stats dict
            shared_temp:         scale factor on agg_std (>1 = more diversity)
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if isinstance(K, int):
                K_tensor = torch.full((num_samples,), K, dtype=torch.long, device=device)
            else:
                K_tensor = K.long().to(device)
                if K_tensor.dim() == 0:
                    K_tensor = K_tensor.expand(num_samples)

            stats = latent_stats if latent_stats is not None else {}
            if per_K_latent_stats is not None and isinstance(K, int):
                k_key = str(K)
                if k_key in per_K_latent_stats:
                    stats = per_K_latent_stats[k_key]
                else:
                    available = [int(k) for k in per_K_latent_stats]
                    if available:
                        nearest = str(min(available, key=lambda x: abs(x - K)))
                        stats = per_K_latent_stats[nearest]

            s = stats.get('latent', {})
            if 'mu_mean_per_dim' in s and 'agg_std_per_dim' in s:
                mu      = torch.tensor(s['mu_mean_per_dim'], dtype=torch.float32, device=device)
                agg_std = torch.tensor(s['agg_std_per_dim'],  dtype=torch.float32, device=device)
                z = torch.randn(num_samples, self.latent_dim, device=device) * (agg_std * shared_temp) + mu
            else:
                mu_val  = s.get('mu_mean', 0.0)
                agg_std = (s.get('mu_std', 1.0) ** 2 + s.get('sigma_mean', 1.0) ** 2) ** 0.5
                z = torch.randn(num_samples, self.latent_dim, device=device) * (agg_std * shared_temp) + mu_val

            heatmap, occupancy, impedance = self.decode(z, K_tensor)
            occupancy = torch.sigmoid(occupancy)  # logits → probabilities for inference

        if was_training:
            self.train()
        return heatmap, occupancy, impedance
