"""Surrogate impedance model and training script — exp037.

Direct mapping:  occ (52,) → impedance (1, 231) log-z spectrum
No VAE bottleneck → can learn exact resonance structure.

The surrogate is used during latent optimization:
    z → VAE occ decoder → occ_binary (STE) → surrogate → accurate impedance → score

Run:
    python experiments/exp037_lat_change/codes/surrogate_impedance.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ── Project root ──────────────────────────────────────────────────────────────
_here = Path(__file__).resolve()
PROJECT_ROOT = next(
    (str(p) for p in _here.parents if (p / "datasets").is_dir() and (p / "experiments").is_dir()),
    str(_here.parents[3]),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class SurrConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir:    str = "datasets/data_multifreq_norm"
    train_split: float = 0.9
    num_workers: int   = 4

    # ── Architecture ──────────────────────────────────────────────────────────
    occ_dim:      int   = 52
    freq_emb_dim: int   = 16     # PI_freq embedding dimension
    hidden_dims:  tuple = (256, 512, 512, 512, 256)
    out_channels: int   = 1      # ch0 log-z — matches VAE output
    freq_pts:     int   = 231    # number of frequency points
    dropout:      float = 0.1

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs:    int   = 200
    batch_size:    int   = 128
    learning_rate: float = 3e-4
    lr_patience:   int   = 15
    lr_factor:     float = 0.5
    lr_min:        float = 1e-5
    weight_decay:  float = 1e-4

    # ── Loss weights ──────────────────────────────────────────────────────────
    ch0_weight:      float = 1.0    # MSE on log-z impedance channel
    topk_k:          int   = 15     # supervise top-K amplitude frequencies
    topk_weight:     float = 3.0    # extra MSE at peak frequencies
    under_penalty:   float = 2.0    # asymmetric: penalise underestimate more

    # ── Paths ─────────────────────────────────────────────────────────────────
    experiment_dir: str = "experiments/exp037_lat_change"
    checkpoint_interval: int = 50
    resume: bool = False   # set True to resume from last checkpoint

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATASET
# =============================================================================

class OccImpDataset(Dataset):
    """Load occupancy → impedance (ch0) pairs. PI_freq in files is ignored (heatmap-only cond)."""

    def __init__(self, data_dir: str, indices: list[int]):
        self.data_dir = Path(data_dir)
        # Normalise PI_freq to [0,1] — range is typically 1e6..1e9
        self.indices  = indices

        # Pre-load into RAM for speed (40k × 52 + 40k × 2 × 231 ≈ 90 MB)
        print(f"  Loading {len(indices)} samples into RAM …", end="", flush=True)
        t0 = time.time()
        occ_list, imp_list, freq_list = [], [], []
        for i in indices:
            occ_list.append(np.load(self.data_dir / "Occ_map" / f"sample_{i}.npy"))
            imp_list.append(np.load(self.data_dir / "Imp"     / f"sample_{i}.npy"))
            freq_list.append(float(np.load(self.data_dir / "PI_freq" / f"sample_{i}.npy")))

        self.occ  = torch.tensor(np.stack(occ_list),  dtype=torch.float32)   # (N, 52)
        self.imp  = torch.tensor(np.stack(imp_list),  dtype=torch.float32)   # (N, 3, 231) or (N,2,231)
        self.freq = torch.tensor(freq_list,            dtype=torch.float32)   # (N,)
        # Normalise freq: log10(f) rescaled to [0,1] over [1e5, 1e9]
        self.freq = (torch.log10(self.freq.clamp(min=1e5)) - 5.0) / 4.0
        print(f" done in {time.time()-t0:.1f}s")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.occ[idx], self.imp[idx, :1]


def _get_indices(data_dir: str) -> list[int]:
    files = list((Path(data_dir) / "Occ_map").glob("sample_*.npy"))
    return sorted(int(f.stem.split("_")[1]) for f in files)


# =============================================================================
# MODEL
# =============================================================================

class SurrogateImpedanceNet(nn.Module):
    """Direct occ → impedance (1, 231). K/decap layout only — not PI_freq."""

    def __init__(self, cfg: SurrConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = SurrConfig()

        self.out_channels = cfg.out_channels
        self.freq_pts     = cfg.freq_pts

        self.occ_proj = nn.Sequential(
            nn.Linear(cfg.occ_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )

        in_dim = 128
        layers: list[nn.Module] = []
        prev = in_dim
        for h in cfg.hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(),
                nn.Dropout(cfg.dropout),
            ]
            prev = h
        # Output head: predict (out_channels * freq_pts) then reshape
        layers.append(nn.Linear(prev, cfg.out_channels * cfg.freq_pts))
        self.net = nn.Sequential(*layers)

    def forward(self, occ: torch.Tensor, pi_freq_norm: torch.Tensor | None = None) -> torch.Tensor:
        """pi_freq_norm is ignored (kept for backward-compatible call sites)."""
        o_feat = self.occ_proj(occ)
        out    = self.net(o_feat)
        return out.view(-1, self.out_channels, self.freq_pts)


# =============================================================================
# LOSS
# =============================================================================

def surrogate_loss(pred: torch.Tensor, target: torch.Tensor, cfg: SurrConfig) -> torch.Tensor:
    """Per-batch loss for (B, 1, 231) predictions."""
    loss = cfg.ch0_weight * F.mse_loss(pred[:, 0], target[:, 0])

    # Top-K frequency supervision on ch0
    if cfg.topk_weight > 0 and cfg.topk_k > 0:
        with torch.no_grad():
            topk_idx = torch.topk(target[:, 0].abs(), k=cfg.topk_k, dim=-1).indices  # (B, K)
        pred_topk   = pred[:, 0].gather(1, topk_idx)
        target_topk = target[:, 0].gather(1, topk_idx)
        loss = loss + cfg.topk_weight * F.mse_loss(pred_topk, target_topk)

    # Asymmetric under-penalty on ch0: penalise predicting lower than target more
    if cfg.under_penalty > 1.0:
        diff = target[:, 0] - pred[:, 0]   # positive where pred < target
        under_mask = (diff > 0).float()
        asym = (diff * under_mask).pow(2).mean()
        loss = loss + (cfg.under_penalty - 1.0) * asym

    return loss


# =============================================================================
# TRAINING
# =============================================================================

def train_surrogate(cfg: SurrConfig | None = None) -> None:
    if cfg is None:
        cfg = SurrConfig()

    device = torch.device(cfg.device)
    exp_dir = Path(cfg.experiment_dir)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    all_idx = _get_indices(cfg.data_dir)
    n_train = int(len(all_idx) * cfg.train_split)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(all_idx).tolist()
    train_idx, val_idx = shuffled[:n_train], shuffled[n_train:]
    print(f"Dataset: {len(all_idx)} total  |  {len(train_idx)} train  |  {len(val_idx)} val")

    print("Loading train split:")
    train_ds = OccImpDataset(cfg.data_dir, train_idx)
    print("Loading val split:")
    val_ds   = OccImpDataset(cfg.data_dir, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SurrogateImpedanceNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Surrogate params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.lr_patience,
        factor=cfg.lr_factor, min_lr=cfg.lr_min,
    )

    start_epoch = 0
    best_val    = float("inf")

    # ── Resume ────────────────────────────────────────────────────────────────
    surrogate_last = ckpt_dir / "surrogate_last.pt"
    if cfg.resume and surrogate_last.exists():
        ckpt = torch.load(surrogate_last, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"Resumed from epoch {start_epoch}  best_val={best_val:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining surrogate epochs {start_epoch+1}–{cfg.num_epochs} …\n")
    for epoch in range(start_epoch, cfg.num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for occ, imp in train_loader:
            occ, imp = occ.to(device), imp.to(device)
            optimizer.zero_grad()
            pred = model(occ)
            loss = surrogate_loss(pred, imp, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for occ, imp in val_loader:
                occ, imp = occ.to(device), imp.to(device)
                pred = model(occ)
                val_loss += surrogate_loss(pred, imp, cfg).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Ep {epoch+1:>3} | train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}",
              flush=True)

        # Checkpoint
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "cfg": cfg.__dict__,
            }, ckpt_dir / "surrogate_best.pt")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val,
            "cfg": cfg.__dict__,
        }, surrogate_last)

        if (epoch + 1) % cfg.checkpoint_interval == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_val": best_val,
                "cfg": cfg.__dict__,
            }, ckpt_dir / f"surrogate_epoch_{epoch+1}.pt")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'surrogate_best.pt'}")


def load_surrogate(ckpt_path: str | Path, device: str | torch.device = "cuda") -> SurrogateImpedanceNet:
    """Load a trained surrogate model from checkpoint.

    Used by latent_optimization_impedance.py:
        from experiments.exp037_lat_change.codes.surrogate_impedance import load_surrogate
        surrogate = load_surrogate(SURROGATE_CHECKPOINT_PATH)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})
    cfg = SurrConfig(**{k: v for k, v in cfg_dict.items() if k in SurrConfig.__dataclass_fields__})
    model = SurrogateImpedanceNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded surrogate from {ckpt_path}  (best_val={ckpt.get('best_val', '?'):.4f})")
    return model


if __name__ == "__main__":
    train_surrogate()
