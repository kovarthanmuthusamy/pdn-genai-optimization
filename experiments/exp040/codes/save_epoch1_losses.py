"""Backfill epoch-1 metrics and rebuild exp039 plots from metrics/loss.csv.

Use when a run was resumed from a later checkpoint (e.g. 750) so loss.csv starts
at epoch 25 and final plots omit epoch 1.

  cd ~/gan
  # Backfill epoch 1 (one GPU epoch, fresh init) + rebuild plots:
  python experiments/exp039_improved_heatmap/codes/save_epoch1_losses.py

  # Only dedupe CSV and rebuild plots (epoch 1 already in loss.csv):

Writes / updates:
  - metrics/loss.csv  (epoch=1 row when backfilled)
  - logs/latent_stats.csv  (epoch=1, when backfilled)
  - metrics/plots/*_final.png  (full curve from CSV, including ep 1 if present)
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

_EXP_DIR = _ROOT.parents[1]
os.environ.setdefault("VAE_EXPERIMENT_DIR", str(_EXP_DIR))
os.environ.setdefault("VAE_DATA_DIR", str(Path(PROJECT_ROOT) / "datasets" / "data_multifreq_norm"))

from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders  # noqa: E402
from experiments.exp038_true_multi.codes.physics_loss import PhysicsLoss  # noqa: E402
from experiments.exp038_true_multi.codes.train_vae_simple import (  # noqa: E402
    Config,
    _adapt_amp_for_gpu,
    _amp_dtype,
    _apply_yaml_config,
    _physics_weights,
    _run_epoch,
    compute_beta,
    compute_modality_dropout,
)
from experiments.exp038_true_multi.codes.vae_multi_input_simple import MultiInputVAE  # noqa: E402
from experiments.exp039_improved_heatmap.codes.metrics_csv_utils import (  # noqa: E402
    insert_epoch_row,
    update_all_metrics_csv,
)
from src_vae.others.vae_logger import CSV_HEADER, LATENT_CSV_HEADER, VAETrainingLogger


# =============================================================================
# CONFIGURATION — edit these before running: python experiments/exp040/codes/save_epoch1_losses.py
# =============================================================================

PLOTS_ONLY = False  # True = skip GPU backfill; rebuild plots from loss.csv only

# =============================================================================

_BACKFILL_RAM = os.getenv("EXP039_BACKFILL_RAM", "0").strip().lower() in ("1", "true", "yes")


def _seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rebuild_plots_from_loss_csv(c: Config) -> None:
    """Reload deduped loss.csv and write the same *_final.png plots as end of training."""
    exp = Path(c.experiment_dir)
    metrics_dir = exp / "metrics"
    plots_dir = metrics_dir / "plots"
    loss_csv = metrics_dir / "loss.csv"
    plots_dir.mkdir(parents=True, exist_ok=True)

    update_all_metrics_csv(metrics_dir, backup=False)
    logger = VAETrainingLogger(
        str(exp / "logs"),
        str(exp / "checkpoints"),
        csv_path=str(loss_csv),
        checkpoint_interval=c.checkpoint_interval,
    )
    if not logger.epochs:
        print(f"No epochs in {loss_csv} — cannot plot.")
        return

    ep_lo, ep_hi = logger.epochs[0], logger.epochs[-1]
    print(f"\n=== Rebuild plots from loss.csv (epochs {ep_lo}–{ep_hi}, n={len(logger.epochs)}) ===")
    if ep_lo > 1:
        print(f"  WARNING: first epoch is {ep_lo}, not 1 — set PLOTS_ONLY=False to backfill epoch 1.")

    logger.plot(save_path=str(plots_dir / "convergence_final.png"))
    logger.plot_loss_components(save_path=str(plots_dir / "loss_components_final.png"))
    logger.plot_overfitting(save_path=str(plots_dir / "overfitting_final.png"))
    if logger.val_physics_ri_loss and any(v == v for v in logger.val_physics_ri_loss):
        logger.plot_physics(save_path=str(plots_dir / "physics_losses_final.png"))
    logger.print_statistics()
    print(f"Plots → {plots_dir}/")


def backfill_epoch1(c: Config) -> None:
    """Run one training epoch (fresh weights) and insert epoch=1 into metrics CSVs."""
    exp = Path(c.experiment_dir)
    metrics_dir = exp / "metrics"
    log_dir = exp / "logs"
    loss_csv = metrics_dir / "loss.csv"

    stats_path = Path(c.data_dir) / "normalization_stats.json"
    imp_log_std = 1.0
    hm_log_mean = 0.0
    hm_log_std = 1.0
    if stats_path.exists():
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
        c.background_value = raw.get("background_value", c.background_value)
        c.physics_fg_clip_min = raw["Heatmap"]["clip_min"]
        imp_log_std = float(raw["Impedance"]["log_std"]) or 1.0
        hm_log_mean = float(raw["Heatmap"]["log_mean"])
        hm_log_std = float(raw["Heatmap"]["log_std"]) or 1.0

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
        cross_freq_pairs=c.cross_freq_weight > 0,
    )

    model = MultiInputVAE(
        latent_dim=c.latent_dim,
        cond_dim=c.cond_dim,
        heatmap_private_dim=c.heatmap_private_dim,
        modality_dropout=c.modality_dropout,
        freq_fourier_features=int(getattr(c, "freq_fourier_features", 8)),
        use_heatmap_film=bool(getattr(c, "use_heatmap_film", True)),
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
        hm_log_mean, hm_log_std,
        train=True, optimizer=opt, scaler=scaler,
    )
    val = _run_epoch(
        model, val_ld, c, epoch, beta, 0.0, physics, pw, imp_log_std,
        hm_log_mean, hm_log_std,
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


def main() -> None:
    c = Config()
    _apply_yaml_config(c)
    _adapt_amp_for_gpu(c)
    _seed_all(42)

    print("=== Clean metrics CSVs (dedupe resume duplicates) ===")
    info = update_all_metrics_csv(Path(c.experiment_dir) / "metrics", backup=True)

    if PLOTS_ONLY:
        rebuild_plots_from_loss_csv(c)
        return

    if not info["needs_epoch1"]:
        print("Epoch 1 already in loss.csv — skipping GPU backfill.")
        rebuild_plots_from_loss_csv(c)
        return

    backfill_epoch1(c)
    update_all_metrics_csv(Path(c.experiment_dir) / "metrics", backup=False)
    rebuild_plots_from_loss_csv(c)
    print("\nDone — loss.csv includes epoch 1 and plots were rebuilt.")


if __name__ == "__main__":
    main()
