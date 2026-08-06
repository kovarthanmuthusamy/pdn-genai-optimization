"""1D spectrum GNN for PI impedance (231 bins) — exp058."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.exp058_asymmetric_kl.codes.graph_occ import GraphConvLayer

NUM_BINS = 231


def build_spectrum_adjacency(*, self_loops: bool = True) -> torch.Tensor:
    """Chain graph along log-frequency bins (231 nodes)."""
    adj = torch.zeros(NUM_BINS, NUM_BINS, dtype=torch.float32)
    for i in range(NUM_BINS):
        if self_loops:
            adj[i, i] = 1.0
        if i + 1 < NUM_BINS:
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
    deg = adj.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)


def _bin_positions_norm() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, NUM_BINS).view(NUM_BINS, 1)


class ImpSpectrumGraphEncoder(nn.Module):
    """(B, 1, 231) or (B, 231) -> pooled feature (B, out_dim)."""

    def __init__(
        self,
        out_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer("adj", build_spectrum_adjacency(), persistent=False)
        self.register_buffer("bin_pos", _bin_positions_norm(), persistent=False)
        self.bin_embed = nn.Embedding(NUM_BINS, hidden_dim // 2)
        in_dim = 1 + hidden_dim // 2 + 1
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(GraphConvLayer(d, hidden_dim, dropout=dropout))
            d = hidden_dim
        self.gnn = nn.ModuleList(layers)
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1),
        )

    def _as_bins(self, imp: torch.Tensor) -> torch.Tensor:
        if imp.dim() == 1:
            imp = imp.unsqueeze(0)
        if imp.dim() == 2:
            if imp.shape[1] != NUM_BINS:
                raise ValueError(f"expected {NUM_BINS} bins, got {imp.shape}")
            return imp
        if imp.dim() == 3:
            return imp[:, 0, :]
        raise ValueError(f"bad impedance shape {tuple(imp.shape)}")

    def forward(self, impedance: torch.Tensor) -> torch.Tensor:
        bins = self._as_bins(impedance)
        b, device = bins.shape[0], bins.device
        adj = self.adj.to(device)
        pos = self.bin_pos.to(device).unsqueeze(0).expand(b, -1, -1)
        ids = torch.arange(NUM_BINS, device=device)
        emb = self.bin_embed(ids).unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([bins.unsqueeze(-1), emb, pos], dim=-1)
        for layer in self.gnn:
            x = layer(x, adj)
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return self.pool_proj(pooled)


class ImpSpectrumGraphDecoder(nn.Module):
    """Global cond + optional occ context -> 231-bin spectrum."""

    def __init__(
        self,
        dec_in: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("adj", build_spectrum_adjacency(), persistent=False)
        self.register_buffer("bin_pos", _bin_positions_norm(), persistent=False)
        self.global_proj = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.bin_embed = nn.Embedding(NUM_BINS, hidden_dim // 2)
        in_dim = hidden_dim + hidden_dim // 2 + 1
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(GraphConvLayer(d, hidden_dim, dropout=dropout))
            d = hidden_dim
        self.gnn = nn.ModuleList(layers)
        self.node_out = nn.Linear(hidden_dim, 1)

    def forward(self, dec_imp: torch.Tensor) -> torch.Tensor:
        b = dec_imp.shape[0]
        device = dec_imp.device
        adj = self.adj.to(device)
        pos = self.bin_pos.to(device).unsqueeze(0).expand(b, -1, -1)
        g = self.global_proj(dec_imp).unsqueeze(1).expand(-1, NUM_BINS, -1)
        ids = torch.arange(NUM_BINS, device=device)
        emb = self.bin_embed(ids).unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([g, emb, pos], dim=-1)
        for layer in self.gnn:
            x = layer(x, adj)
        return self.node_out(x).squeeze(-1)
