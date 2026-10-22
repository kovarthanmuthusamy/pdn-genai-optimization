"""Building blocks of the VAE: frequency conditioning, conv blocks and the occupancy tower.

Usage (import only):
FreqConditioner(cond_dim, num_fourier)   normalized PI frequency (B,) in [0, 1] -> (B, cond_dim)
FiLM2d(channels, cond_dim)               x * (1 + gamma(cond)) + beta(cond), identity at init
SelfAttn2d(channels)                     single-head spatial self-attention with a residual
ResidualConv2d(channels)                 two 3x3 convs with a residual
UpsampleConv2d(in_ch, out_ch)            bilinear x2 upsample + 3x3 conv (no checkerboard)
OccSpatialTower(occ_ch)                  (B,1,7,8) slot-presence grid -> features at 16/32/64 px
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqConditioner(nn.Module):
    """Map the normalized PI frequency in [0, 1] to a ``cond_dim`` vector (Fourier features + MLP)."""

    def __init__(self, cond_dim: int, num_fourier: int = 8):
        super().__init__()
        self.num_fourier = num_fourier
        self.net = nn.Sequential(
            nn.Linear(1 + 2 * num_fourier, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _fourier(self, f: torch.Tensor) -> torch.Tensor:
        """(B, 1) in [0, 1] -> (B, 2 * num_fourier) sin/cos features."""
        bands = torch.arange(1, self.num_fourier + 1, device=f.device, dtype=f.dtype)
        angles = (2.0 * math.pi) * f * bands.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, pi_freq_norm: torch.Tensor) -> torch.Tensor:
        f = pi_freq_norm.float().reshape(-1, 1).clamp(0.0, 1.0)
        return self.net(torch.cat([f, self._fourier(f)], dim=-1))


class FiLM2d(nn.Module):
    """Per-channel scale and shift predicted from a conditioning vector: ``x * (1 + gamma) + beta``.
    The projection starts at zero weight, so at initialization it ignores the condition."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:channels].fill_(1.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=1)
        return x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)


class SelfAttn2d(nn.Module):
    """Single-head spatial self-attention over a (B, C, H, W) map, with a residual."""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = self.norm(x).view(b, c, h * w).permute(0, 2, 1)
        out, _ = self.attn(seq, seq, seq)
        return x + out.permute(0, 2, 1).view(b, c, h, w)


class ResidualConv2d(nn.Module):
    """Two 3x3 convs with a residual connection, same resolution in and out."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class UpsampleConv2d(nn.Module):
    """Bilinear upsample then conv (avoids ConvTranspose2d checkerboard artifacts)."""

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        ]
        if out_ch >= 8:
            layers.append(nn.GroupNorm(min(8, out_ch), out_ch))
        layers.append(nn.LeakyReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OccSpatialTower(nn.Module):
    """7x8 decap-slot presence grid -> layout features at 16, 32 and 64 px for the heatmap decoder."""

    def __init__(self, occ_ch: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
        )
        self.proj16 = nn.Conv2d(32, occ_ch, kernel_size=1)
        self.proj32 = nn.Conv2d(32, occ_ch, kernel_size=1)
        self.proj64 = nn.Conv2d(32, occ_ch, kernel_size=1)
        self.occ_ch = occ_ch

    def forward(self, grid: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(grid)
        up = lambda proj, s: F.interpolate(proj(feat), size=(s, s), mode="bilinear", align_corners=False)  # noqa: E731
        return {"16": up(self.proj16, 16), "32": up(self.proj32, 32), "64": up(self.proj64, 64)}
