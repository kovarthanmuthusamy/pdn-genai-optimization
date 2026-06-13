"""Surrogate impedance model and training script — exp038_true_multi.

Direct mapping:  occ (52,) → impedance (1, 231) log-z spectrum
PI_freq is not used (heatmap-only conditioning in the VAE).

Run:
    python experiments/exp038_true_multi/codes/surrogate_impedance.py
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

from experiments.exp038_true_multi.codes.impedance_spectrum_loss import surrogate_spectrum_loss

_here = Path(__file__).resolve()
PROJECT_ROOT = next(
    (str(p) for p in _here.parents if (p / "datasets").is_dir() and (p / "experiments").is_dir()),
    str(_here.parents[3]),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class SurrConfig:
    data_dir:    str = "/home/ubuntu/gan/datasets/data_multifreq_norm"
    train_split: float = 0.9
    num_workers: int   = 4

    occ_dim:      int   = 52
    freq_emb_dim: int   = 16
    hidden_dims:  tuple = (256, 512, 512, 512, 256)
    out_channels: int   = 1
    freq_pts:     int   = 231
    dropout:      float = 0.1

    num_epochs:    int   = 200
    batch_size:    int   = 128
    learning_rate: float = 3e-4
    lr_patience:   int   = 15
    lr_factor:     float = 0.5
    lr_min:        float = 1e-5
    weight_decay:  float = 1e-4

    ch0_weight:      float = 1.0
    topk_k:          int   = 20
    topk_weight:     float = 7.0
    under_penalty:   float = 2.8
    freq_weight_alpha: float = 2.0
    dual_topk_weight: float = 0.75
    peak_index_weight: float = 2.5
    peak_mag_weight: float = 2.0
    num_peaks:       int   = 8

    experiment_dir: str = "/home/ubuntu/gan/experiments/exp038_true_multi"
    checkpoint_interval: int = 50
    resume: bool = False

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class OccImpDataset(Dataset):
    """occ → impedance (ch0). PI_freq in files is ignored."""

    def __init__(self, data_dir: str, indices: list[int]):
        self.data_dir = Path(data_dir)
        self.indices  = indices

        print(f"  Loading {len(indices)} samples into RAM …", end="", flush=True)
        t0 = time.time()
        occ_list, imp_list = [], []
        for i in indices:
            occ_list.append(np.load(self.data_dir / "Occ_map" / f"sample_{i}.npy"))
            imp_list.append(np.load(self.data_dir / "Imp" / f"sample_{i}.npy"))

        self.occ = torch.tensor(np.stack(occ_list), dtype=torch.float32)
        self.imp = torch.tensor(np.stack(imp_list), dtype=torch.float32)
        print(f" done in {time.time()-t0:.1f}s")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.occ[idx], self.imp[idx, :1]


def _get_indices(data_dir: str) -> list[int]:
    files = list((Path(data_dir) / "Occ_map").glob("sample_*.npy"))
    return sorted(int(f.stem.split("_")[1]) for f in files)


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

        prev = 128
        layers: list[nn.Module] = []
        for h in cfg.hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(),
                nn.Dropout(cfg.dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, cfg.out_channels * cfg.freq_pts))
        self.net = nn.Sequential(*layers)

    def forward(self, occ: torch.Tensor, pi_freq_norm: torch.Tensor | None = None) -> torch.Tensor:
        del pi_freq_norm
        o_feat = self.occ_proj(occ)
        out    = self.net(o_feat)
        return out.view(-1, self.out_channels, self.freq_pts)


def surrogate_loss(pred: torch.Tensor, target: torch.Tensor, cfg: SurrConfig) -> torch.Tensor:
    return surrogate_spectrum_loss(
        pred,
        target,
        ch0_weight=cfg.ch0_weight,
        topk_k=cfg.topk_k,
        topk_weight=cfg.topk_weight,
        under_penalty=cfg.under_penalty,
        freq_weight_alpha=cfg.freq_weight_alpha,
        dual_topk_weight=cfg.dual_topk_weight,
        peak_index_weight=cfg.peak_index_weight,
        peak_mag_weight=cfg.peak_mag_weight,
        num_peaks=cfg.num_peaks,
    )


def train_surrogate(cfg: SurrConfig | None = None) -> None:
    if cfg is None:
        cfg = SurrConfig()

    device = torch.device(cfg.device)
    ckpt_dir = Path(cfg.experiment_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    model = SurrogateImpedanceNet(cfg).to(device)
    print(f"Surrogate params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.lr_patience,
        factor=cfg.lr_factor, min_lr=cfg.lr_min,
    )

    start_epoch = 0
    best_val    = float("inf")

    surrogate_last = ckpt_dir / "surrogate_last.pt"
    if cfg.resume and surrogate_last.exists():
        ckpt = torch.load(surrogate_last, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"Resumed from epoch {start_epoch}  best_val={best_val:.4f}")

    print(f"\nTraining surrogate epochs {start_epoch+1}–{cfg.num_epochs} …\n")
    for epoch in range(start_epoch, cfg.num_epochs):
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
