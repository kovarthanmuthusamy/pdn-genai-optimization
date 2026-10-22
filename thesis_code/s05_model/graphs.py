"""Graph networks over the 52 decap slots and the 231 spectrum bins (pure PyTorch).

Occupancy: the slots are nodes on the 7x8 board grid, edges join 4-neighbours. Impedance: the
bins are nodes on a chain along log-frequency. Both use the same ``GraphConvLayer`` (self + mean
neighbour message, LayerNorm, LeakyReLU) and a symmetric-normalized adjacency.

Usage (import only):
OccGraphEncoder(out_dim, ...)          layout (B,52,T) one-hot or (B,52) presence -> (B, out_dim)
OccGraphDecoder(dec_in, ...)           (B, dec_in) -> per-slot part logits (B, 52, T)
ImpSpectrumGraphEncoder(out_dim, ...)  spectrum (B,1,231) / (B,231) -> (B, out_dim)
ImpSpectrumGraphDecoder(dec_in, ...)   (B, dec_in) -> spectrum (B, 231)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis_code.common.placement import LABEL_TO_COORD, LABELS_ORDERED

NUM_SLOTS = len(LABELS_ORDERED)
NUM_BINS = 231


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("nm,bmf->bnf", adj, x)
        out = self.lin_self(x) + self.lin_neigh(neigh)
        out = self.norm(out)
        out = F.leaky_relu(out, 0.1)
        if self.dropout > 0.0 and self.training:
            out = F.dropout(out, p=self.dropout, training=True)
        return out


def _init_linear_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------------------
# Occupancy: 7x8 grid graph over the 52 decap slots
# --------------------------------------------------------------------------------------


def _grid_neighbors(h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dh, dw in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nh, nw = h + dh, w + dw
        if 0 <= nh < 7 and 0 <= nw < 8:
            out.append((nh, nw))
    return out


def build_slot_adjacency(*, self_loops: bool = True) -> torch.Tensor:
    """Symmetric normalized adjacency (N, N) for 52 decap slots."""
    label_to_idx = {label: i for i, label in enumerate(LABELS_ORDERED)}
    coord_to_label = {coord: label for label, coord in LABEL_TO_COORD.items()}

    adj = torch.zeros(NUM_SLOTS, NUM_SLOTS, dtype=torch.float32)
    for i, label in enumerate(LABELS_ORDERED):
        h, w = LABEL_TO_COORD[label]
        for nh, nw in _grid_neighbors(h, w):
            nlabel = coord_to_label.get((nh, nw))
            if nlabel is None:
                continue
            j = label_to_idx[nlabel]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        if self_loops:
            adj[i, i] = 1.0

    deg = adj.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = deg.pow(-0.5)
    adj = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
    return adj


def _slot_positions_norm() -> torch.Tensor:
    """(N, 2) row/col normalized to [0, 1]."""
    pos = torch.zeros(NUM_SLOTS, 2, dtype=torch.float32)
    for i, label in enumerate(LABELS_ORDERED):
        h, w = LABEL_TO_COORD[label]
        pos[i, 0] = h / 6.0
        pos[i, 1] = w / 7.0
    return pos


class OccGraphEncoder(nn.Module):
    """occupancy (B, 52, T) one-hot -> graph pool -> (B, out_dim).

    T catalog types; empty = all-zero row. Legacy (B, 52) binary -> type-1 channel.
    """

    def __init__(
        self,
        out_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        n_occ_classes: int = 2,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.n_occ_classes = int(n_occ_classes)
        self.register_buffer("adj", build_slot_adjacency(), persistent=False)
        self.register_buffer("slot_pos", _slot_positions_norm(), persistent=False)

        self.slot_embed = nn.Embedding(NUM_SLOTS, hidden_dim // 2)
        in_dim = self.n_occ_classes + hidden_dim // 2 + 2
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
        _init_linear_weights(self)

    def forward(self, occupancy: torch.Tensor) -> torch.Tensor:
        if occupancy.dim() == 2:
            # legacy binary (B, 52) -> (B, 52, T); type-1 on channel 0, empty = zeros
            occupancy = occupancy.unsqueeze(-1)
            if self.n_occ_classes > 1:
                pad = occupancy.new_zeros(*occupancy.shape[:-1], self.n_occ_classes - 1)
                occupancy = torch.cat([occupancy, pad], dim=-1)
        elif occupancy.dim() != 3 or occupancy.shape[-1] != self.n_occ_classes:
            raise ValueError(
                f"OccGraphEncoder expected (B, {NUM_SLOTS}, {self.n_occ_classes}), "
                f"got {tuple(occupancy.shape)}"
            )
        b = occupancy.shape[0]
        device = occupancy.device
        adj = self.adj.to(device)
        pos = self.slot_pos.to(device).unsqueeze(0).expand(b, -1, -1)
        slot_ids = torch.arange(NUM_SLOTS, device=device)
        slot_feat = self.slot_embed(slot_ids).unsqueeze(0).expand(b, -1, -1)
        occ_n = occupancy
        x = torch.cat([occ_n, slot_feat, pos], dim=-1)
        for layer in self.gnn:
            x = layer(x, adj)
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return self.pool_proj(pooled)


class OccGraphDecoder(nn.Module):
    """Global latent cond (B, dec_in) -> per-node class logits (B, 52, T)."""

    def __init__(
        self,
        dec_in: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        n_occ_classes: int = 2,
    ):
        super().__init__()
        self.n_occ_classes = int(n_occ_classes)
        self.register_buffer("adj", build_slot_adjacency(), persistent=False)
        self.register_buffer("slot_pos", _slot_positions_norm(), persistent=False)

        self.global_proj = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        self.slot_embed = nn.Embedding(NUM_SLOTS, hidden_dim // 2)
        in_dim = hidden_dim + hidden_dim // 2 + 2
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(GraphConvLayer(d, hidden_dim, dropout=dropout))
            d = hidden_dim
        self.gnn = nn.ModuleList(layers)
        self.node_out = nn.Linear(hidden_dim, self.n_occ_classes)
        _init_linear_weights(self)

    def forward(self, dec_occ: torch.Tensor) -> torch.Tensor:
        b = dec_occ.shape[0]
        device = dec_occ.device
        adj = self.adj.to(device)
        pos = self.slot_pos.to(device).unsqueeze(0).expand(b, -1, -1)
        g = self.global_proj(dec_occ).unsqueeze(1).expand(-1, NUM_SLOTS, -1)
        slot_ids = torch.arange(NUM_SLOTS, device=device)
        slot_feat = self.slot_embed(slot_ids).unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([g, slot_feat, pos], dim=-1)
        for layer in self.gnn:
            x = layer(x, adj)
        return self.node_out(x)  # (B, 52, T)


# --------------------------------------------------------------------------------------
# Impedance: 231-bin chain graph along log-frequency
# --------------------------------------------------------------------------------------


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
