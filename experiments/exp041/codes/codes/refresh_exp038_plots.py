"""Update metrics/loss.csv (dedupe) and rebuild exp038 plots from VAETrainingLogger.

    python experiments/exp038_true_multi/codes/refresh_exp038_plots.py

Skips epoch-1 GPU backfill; run save_epoch1_losses.py first if plots should start at ep 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
PROJECT_ROOT = next(
    (str(p) for p in _ROOT.parents if (p / "src_vae").is_dir()),
    str(_ROOT.parents[3]),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.exp038_true_multi.codes.metrics_csv_utils import dedupe_csv_by_epoch, update_metrics_loss_csv
from experiments.exp038_true_multi.codes.train_vae_simple import Config
from src_vae.others.vae_logger import LATENT_CSV_HEADER, VAETrainingLogger


def main() -> None:
    c = Config()
    exp = Path(c.experiment_dir)
    metrics_dir = exp / "metrics"
    plots_dir = metrics_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = metrics_dir / "loss.csv"
    latent_csv = exp / "logs" / "latent_stats.csv"

    print("=== Update metrics CSVs ===")
    info = update_metrics_loss_csv(metrics_dir, backup=True)
    if latent_csv.exists():
        n0, n1 = dedupe_csv_by_epoch(latent_csv, LATENT_CSV_HEADER)
        if n0:
            print(f"  latent_stats.csv: {n0} → {n1} rows")
    if info.get("needs_epoch1"):
        print("\n  NOTE: epoch 1 missing in loss.csv — run save_epoch1_losses.py for a full curve from ep 1.")

    if info.get("loss_rows", 0) == 0:
        print(f"No data in {loss_csv}")
        return

    print("\n=== Rebuild plots ===")
    logger = VAETrainingLogger(
        str(exp / "logs"),
        str(exp / "checkpoints"),
        csv_path=str(loss_csv),
        checkpoint_interval=c.checkpoint_interval,
    )
    if not logger.epochs:
        print("No epochs loaded — check loss.csv")
        return

    logger.plot(save_path=str(plots_dir / "convergence_final.png"))
    logger.plot_loss_components(save_path=str(plots_dir / "loss_components_final.png"))
    logger.plot_overfitting(save_path=str(plots_dir / "overfitting_final.png"))
    if logger.val_physics_ri_loss and any(v == v for v in logger.val_physics_ri_loss):
        logger.plot_physics(save_path=str(plots_dir / "physics_losses_final.png"))
    logger.print_statistics()
    print(f"\nUpdated {loss_csv}")
    print(f"Plots → {plots_dir}/")


if __name__ == "__main__":
    main()
