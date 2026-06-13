"""PI frequency conditioning: Fourier features + MLP; FiLM for heatmap conv blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FreqConditioner(nn.Module):
    """Map normalised PI_freq in [0, 1] to cond_dim (replaces Linear(1, cond_dim))."""

    def __init__(self, cond_dim: int, num_fourier: int = 8):
        super().__init__()
        self.num_fourier = num_fourier
        in_dim = 1 + 2 * num_fourier
        self.net = nn.Sequential(
            nn.Linear(in_dim, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
        )
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _fourier(self, f: torch.Tensor) -> torch.Tensor:
        """f: (B, 1) in [0, 1] → (B, 2*num_fourier)."""
        bands = torch.arange(1, self.num_fourier + 1, device=f.device, dtype=f.dtype)
        angles = (2.0 * math.pi) * f * bands.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, pi_freq_norm: torch.Tensor) -> torch.Tensor:
        f = pi_freq_norm.float().reshape(-1, 1).clamp(0.0, 1.0)
        x = torch.cat([f, self._fourier(f)], dim=-1)
        return self.net(x)


class FiLM2d(nn.Module):
    """Feature-wise scale/shift from a conditioning vector."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:channels].fill_(1.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.proj(cond)
        gamma, beta = gb.chunk(2, dim=1)
        return x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
