"""Backfill epoch-1 train/val metrics into metrics/loss.csv (no checkpoint write).

Replays one training epoch with fresh init + seed=42 (same as a new exp038 run at ep 1).

    python experiments/exp038_true_multi/codes/save_epoch1_losses.py
    python experiments/exp038_true_multi/codes/refresh_exp038_plots.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

_ROOT = Path(__file__).resolve()
PROJECT_ROOT = next(
    (str(p) for p in _ROOT.parents if (p / "src_vae").is_dir()),
    str(_ROOT.parents[3]),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders
from experiments.exp038_true_multi.codes.metrics_csv_utils import insert_epoch_row, update_metrics_loss_csv
from experiments.exp038_true_multi.codes.physics_loss import PhysicsLoss
from experiments.exp038_true_multi.codes.train_vae_simple import (
    Config,
    _adapt_amp_for_gpu,
    _amp_dtype,
    _physics_weights,
    _run_epoch,
    compute_beta,
    compute_modality_dropout,
)
from experiments.exp038_true_multi.codes.vae_multi_input_simple import MultiInputVAE
from src_vae.others.vae_logger import CSV_HEADER, LATENT_CSV_HEADER

_BACKFILL_RAM = os.getenv("EXP038_BACKFILL_RAM", "0").strip().lower() in ("1", "true", "yes")


def _seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    c = Config()
    _adapt_amp_for_gpu(c)
    _seed_all(42)

    exp = Path(c.experiment_dir)
    metrics_dir = exp / "metrics"
    log_dir = exp / "logs"
    loss_csv = metrics_dir / "loss.csv"

    print("=== Clean metrics/loss.csv (dedupe resume duplicates) ===")
    info = update_metrics_loss_csv(metrics_dir, backup=True)
    if not info["needs_epoch1"]:
        print("Epoch 1 already in loss.csv — nothing to backfill.")
        return

    stats_path = Path(c.data_dir) / "normalization_stats.json"
    imp_log_std = 1.0
    if stats_path.exists():
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
        c.background_value = raw["background_value"]
        c.physics_fg_clip_min = raw["Heatmap"]["clip_min"]
        imp_log_std = float(raw["Impedance"]["log_std"]) or 1.0

    use_ram = _BACKFILL_RAM or c.cache_in_ram
    print(f"\n=== Epoch 1 backfill | cache_in_ram={use_ram} | train_draws={c.train_samples_per_epoch} ===")

    train_ld, val_ld = create_multifreq_data_loaders(
        data_dir=c.data_dir,
        batch_size=c.batch_size,
        num_workers=c.num_workers,
        train_split=c.train_split,
        seed=42,
        pin_memory=c.is_cuda(),
        split_by_design=c.split_by_design,
        stratify_by_k=c.stratify_by_k,
        balance_k=c.balance_k,
        balance_freq=c.balance_freq,
        k_balance_power=c.k_balance_power,
        freq_balance_power=c.freq_balance_power,
        k_balance_smoothing=c.k_balance_smoothing,
        cache_in_ram=use_ram,
        train_samples_per_epoch=c.train_samples_per_epoch,
    )

    model = MultiInputVAE(
        latent_dim=c.latent_dim,
        cond_dim=c.cond_dim,
        heatmap_private_dim=c.heatmap_private_dim,
        modality_dropout=c.modality_dropout,
    ).to(c.device)

    opt = optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=1e-4, foreach=True)
    physics = None
    if c.physics_ri_weight > 0 or c.physics_critic_sup_weight > 0:
        physics = PhysicsLoss(c.background_value, c.physics_fg_clip_min).to(c.device)
        opt.add_param_group({"params": list(physics.parameters()), "lr": c.learning_rate})

    epoch, ep = 0, 1
    beta = compute_beta(epoch, c)
    md = compute_modality_dropout(epoch, c)
    pw = _physics_weights(epoch, c) if physics else None
    scaler = torch.amp.GradScaler("cuda", enabled=(_amp_dtype(c) == torch.float16))

    print("Running epoch 1 (fresh weights, seed=42) …")
    tr = _run_epoch(
        model, train_ld, c, epoch, beta, md, physics, pw, imp_log_std,
        train=True, optimizer=opt, scaler=scaler,
    )
    val = _run_epoch(
        model, val_ld, c, epoch, beta, 0.0, physics, pw, imp_log_std,
        train=False, collect_per_k=False,
    )
    print(f"  train={tr['total_loss']:.4f}  val={val['total_loss']:.4f}  β={beta:.3f}")

    insert_epoch_row(
        loss_csv,
        CSV_HEADER,
        [
            ep,
            tr["total_loss"],
            tr["recon_loss"],
            tr["kl_loss"],
            tr["heatmap_loss"],
            tr["occupancy_loss"],
            tr["impedance_loss"],
            val["total_loss"],
            val["recon_loss"],
            val["kl_loss"],
            val["heatmap_loss"],
            val["occupancy_loss"],
            val["impedance_loss"],
            val.get("expert_kl_loss", float("nan")),
            val.get("sigma_floor_loss", float("nan")),
            val.get("physics_ri_loss", float("nan")),
            val.get("physics_critic_sup_loss", float("nan")),
            val.get("physics_ar_loss", float("nan")),
        ],
    )

    ms = val.get("modality_stats") or {}
    if ms:
        insert_epoch_row(
            log_dir / "latent_stats.csv",
            LATENT_CSV_HEADER,
            [
                ep,
                f"{beta:.6f}",
                f"{ms['heatmap']['mu_mean']:.4f}",
                f"{ms['heatmap']['std_mean']:.4f}",
                f"{ms['occupancy']['mu_mean']:.4f}",
                f"{ms['occupancy']['std_mean']:.4f}",
                f"{ms['impedance']['mu_mean']:.4f}",
                f"{ms['impedance']['std_mean']:.4f}",
                f"{val['mu_mean']:.4f}",
                f"{val['mu_std']:.4f}",
                f"{val['std_mean']:.4f}",
                f"{val['mu_min']:.4f}",
                f"{val['mu_max']:.4f}",
                f"{val['kl_gaussian']:.4f}",
                f"{val['recon_loss']:.4f}",
                f"{val['total_loss']:.4f}",
            ],
        )

    update_metrics_loss_csv(metrics_dir, backup=False)
    print("\nDone — metrics/loss.csv updated. Next:")
    print("  python experiments/exp038_true_multi/codes/refresh_exp038_plots.py")


if __name__ == "__main__":
    main()
