"""
Logging utilities for Variational Autoencoder training

Reference: src/logger.py
Adapted for VAE with multi-input, multi-output architecture
"""

import csv
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List


CSV_HEADER = [
    'epoch',
    'train_total_loss', 'train_recon_loss', 'train_kl_loss',
    'train_heatmap_loss', 'train_occupancy_loss', 'train_impedance_loss',
    'val_total_loss', 'val_recon_loss', 'val_kl_loss',
    'val_heatmap_loss', 'val_occupancy_loss', 'val_impedance_loss',
    'val_expert_kl_loss', 'val_sigma_floor_loss',
    'val_physics_ri_loss', 'val_physics_critic_sup_loss', 'val_physics_ar_loss',
]

LATENT_CSV_HEADER = [
    'epoch', 'beta',
    'hm_mu_mean', 'hm_sigma_mean', 'occ_mu_mean', 'occ_sigma_mean',
    'imp_mu_mean', 'imp_sigma_mean',
    'fused_mu_mean', 'fused_mu_std', 'fused_sigma_mean', 'fused_mu_min', 'fused_mu_max',
    'kl_gaussian', 'val_recon', 'val_total',
]


class VAETrainingLogger:
    """Logger for tracking and visualizing VAE training metrics"""
    
    def __init__(self, log_dir: str, checkpoint_dir: Optional[str] = None,
                 csv_path: Optional[str] = None, checkpoint_interval: int = 25,
                 log_interval: Optional[int] = None):
        """
        Initialize the VAE logger
        
        Args:
            log_dir: Directory to save logs and metrics
            checkpoint_dir: Directory to save checkpoints (default: parent of log_dir/checkpoints)
            csv_path: Path to the loss CSV file. Defaults to <log_dir>/metrics.csv
            checkpoint_interval: Log CSV/plots/latent stats every N epochs (same as training checkpoints)
            log_interval: Deprecated alias for checkpoint_interval
        """
        if log_interval is not None:
            checkpoint_interval = log_interval
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = self.log_dir.parent / "checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(csv_path) if csv_path is not None else self.log_dir / "metrics.csv"
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.log_interval = self.checkpoint_interval  # backward-compatible alias

        # Metric storage — train
        self.epochs = []
        self.total_loss = []
        self.recon_loss = []
        self.kl_loss = []
        self.heatmap_loss = []
        self.occupancy_loss = []
        self.impedance_loss = []

        # Metric storage — validation (for overfitting detection)
        self.val_total_loss = []
        self.val_recon_loss = []
        self.val_kl_loss = []
        self.val_heatmap_loss = []
        self.val_occupancy_loss = []
        self.val_impedance_loss = []
        self.val_expert_kl_loss = []
        self.val_sigma_floor_loss = []
        self.val_physics_ri_loss = []
        self.val_physics_critic_sup_loss = []
        self.val_physics_ar_loss = []

        # Load existing CSV data so resumed runs accumulate the full history.
        # Also fix up empty/missing-header CSVs so later appends stay parseable.
        if self.csv_path.exists():
            self._load_existing_csv()
            self._ensure_csv_header()
        else:
            self._write_csv_header(overwrite=True)

    def _write_csv_header(self, overwrite: bool = False):
        mode = 'w' if overwrite else 'x'
        try:
            with open(self.csv_path, mode, newline='') as f:
                csv.writer(f).writerow(CSV_HEADER)
        except FileExistsError:
            return

    def _ensure_csv_header(self):
        """Ensure CSV exists and has a header line; safe for resumed runs."""
        try:
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                self._write_csv_header(overwrite=True)
                return
            with open(self.csv_path, 'r', newline='') as f:
                first = f.readline().strip()
            if not first:
                self._write_csv_header(overwrite=True)
                return
            # Accept BOM/whitespace variants of 'epoch'
            first_key = first.split(',')[0].strip().lstrip('\ufeff')
            if first_key != 'epoch':
                # Do not overwrite a non-empty file; just warn.
                print(f"  WARNING: CSV {self.csv_path.name} has unexpected header; keeping as-is")
        except Exception as e:
            print(f"  WARNING: could not validate CSV header {self.csv_path}: {e}")
    
    def _load_existing_csv(self):
        """Load previously logged metrics from CSV into memory (for resumed runs)."""
        def _f(v):
            try:
                f = float(v)
                return float('nan') if f != f else f  # preserve NaN
            except (ValueError, TypeError):
                return float('nan')

        def _norm_key(k: str) -> str:
            return (k or '').strip().lstrip('\ufeff')

        def _norm_row(row: dict) -> dict:
            return {_norm_key(k): v for k, v in row.items()}

        try:
            with open(self.csv_path, newline='') as f:
                reader = csv.DictReader(f)
                rows = [_norm_row(r) for r in reader]
        except Exception as e:
            print(f"  WARNING: could not read existing CSV {self.csv_path}: {e} — starting fresh")
            return

        if not rows:
            print(f"  WARNING: CSV {self.csv_path.name} is empty — starting fresh")
            return

        if 'epoch' not in rows[0]:
            first_key = next(iter(rows[0])) if rows[0] else ''
            print(f"  WARNING: CSV {self.csv_path.name} missing 'epoch' column (first key '{first_key}') — starting fresh")
            return

        for row in rows:
            try:
                self.epochs.append(int(row['epoch']))
            except (ValueError, KeyError):
                continue
            self.total_loss.append(_f(row.get('train_total_loss', 'nan')))
            self.recon_loss.append(_f(row.get('train_recon_loss', 'nan')))
            self.kl_loss.append(_f(row.get('train_kl_loss', 'nan')))
            self.heatmap_loss.append(_f(row.get('train_heatmap_loss', 'nan')))
            self.occupancy_loss.append(_f(row.get('train_occupancy_loss', 'nan')))
            self.impedance_loss.append(_f(row.get('train_impedance_loss', 'nan')))
            self.val_total_loss.append(_f(row.get('val_total_loss', 'nan')))
            self.val_recon_loss.append(_f(row.get('val_recon_loss', 'nan')))
            self.val_kl_loss.append(_f(row.get('val_kl_loss', 'nan')))
            self.val_heatmap_loss.append(_f(row.get('val_heatmap_loss', 'nan')))
            self.val_occupancy_loss.append(_f(row.get('val_occupancy_loss', 'nan')))
            self.val_impedance_loss.append(_f(row.get('val_impedance_loss', 'nan')))
            self.val_expert_kl_loss.append(_f(row.get('val_expert_kl_loss', 'nan')))
            self.val_sigma_floor_loss.append(_f(row.get('val_sigma_floor_loss', 'nan')))
            self.val_physics_ri_loss.append(_f(row.get('val_physics_ri_loss', 'nan')))
            self.val_physics_critic_sup_loss.append(_f(row.get('val_physics_critic_sup_loss', 'nan')))
            self.val_physics_ar_loss.append(_f(row.get('val_physics_ar_loss', 'nan')))
        if self.epochs:
            print(f"  Loaded {len(self.epochs)} existing epochs from {self.csv_path.name} "
                  f"(ep {self.epochs[0]}–{self.epochs[-1]})")

    def at_checkpoint_epoch(self, epoch: int) -> bool:
        """True when epoch (1-based) should log metrics / align with checkpoint saves."""
        # Always include epoch 1 so fresh runs (or resumes) never miss initial metrics/plots.
        return epoch == 1 or (epoch > 0 and epoch % self.checkpoint_interval == 0)

    def log(self, epoch: int,
            total_loss: float,
            recon_loss: float,
            kl_loss: float,
            heatmap_loss: float,
            occupancy_loss: float,
            impedance_loss: float,
            val_total_loss: float = None, # type: ignore
            val_recon_loss: float = None, # type: ignore
            val_kl_loss: float = None, # type: ignore
            val_heatmap_loss: float = None, # type: ignore
            val_occupancy_loss: float = None, # type: ignore
            val_impedance_loss: float = None, # type: ignore
            val_expert_kl_loss: float = None, # type: ignore
            val_sigma_floor_loss: float = None, # type: ignore
            val_physics_ri_loss: float = None, # type: ignore
            val_physics_critic_sup_loss: float = None, # type: ignore
            val_physics_ar_loss: float = None): # type: ignore
        """
        Log train (and optionally validation) losses for one epoch.
        """
        if not self.at_checkpoint_epoch(epoch):
            return

        self._ensure_csv_header()

        # Truncate everything at or after `epoch` so a resume from any
        # checkpoint correctly overwrites stale future data even when the
        # exact epoch number was never written (off-by-one, mid-epoch crash).
        cut = next((i for i, e in enumerate(self.epochs) if e >= epoch),
                   len(self.epochs))
        if cut < len(self.epochs):
            self.epochs              = self.epochs[:cut]
            self.total_loss          = self.total_loss[:cut]
            self.recon_loss          = self.recon_loss[:cut]
            self.kl_loss             = self.kl_loss[:cut]
            self.heatmap_loss        = self.heatmap_loss[:cut]
            self.occupancy_loss      = self.occupancy_loss[:cut]
            self.impedance_loss      = self.impedance_loss[:cut]
            self.val_total_loss      = self.val_total_loss[:cut]
            self.val_recon_loss      = self.val_recon_loss[:cut]
            self.val_kl_loss         = self.val_kl_loss[:cut]
            self.val_heatmap_loss    = self.val_heatmap_loss[:cut]
            self.val_occupancy_loss  = self.val_occupancy_loss[:cut]
            self.val_impedance_loss  = self.val_impedance_loss[:cut]
            self.val_expert_kl_loss  = self.val_expert_kl_loss[:cut]
            self.val_sigma_floor_loss = self.val_sigma_floor_loss[:cut]
            self.val_physics_ri_loss          = self.val_physics_ri_loss[:cut]
            self.val_physics_critic_sup_loss  = self.val_physics_critic_sup_loss[:cut]
            self.val_physics_ar_loss          = self.val_physics_ar_loss[:cut]
            # Rewrite CSV with only the kept rows (atomic to avoid truncation on crash)
            tmp = self.csv_path.with_suffix(self.csv_path.suffix + '.tmp')
            with open(tmp, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADER)
                for i in range(len(self.epochs)):
                    w.writerow([
                        self.epochs[i],
                        self.total_loss[i], self.recon_loss[i], self.kl_loss[i],
                        self.heatmap_loss[i], self.occupancy_loss[i], self.impedance_loss[i],
                        self.val_total_loss[i], self.val_recon_loss[i], self.val_kl_loss[i],
                        self.val_heatmap_loss[i], self.val_occupancy_loss[i], self.val_impedance_loss[i],
                        self.val_expert_kl_loss[i], self.val_sigma_floor_loss[i],
                        self.val_physics_ri_loss[i],
                        self.val_physics_critic_sup_loss[i], self.val_physics_ar_loss[i],
                    ])
            tmp.replace(self.csv_path)

        self.epochs.append(epoch)
        self.total_loss.append(total_loss)
        self.recon_loss.append(recon_loss)
        self.kl_loss.append(kl_loss)
        self.heatmap_loss.append(heatmap_loss)
        self.occupancy_loss.append(occupancy_loss)
        self.impedance_loss.append(impedance_loss)

        # Validation metrics (None → NaN stored so lists stay in sync)
        self.val_total_loss.append(val_total_loss if val_total_loss is not None else float('nan'))
        self.val_recon_loss.append(val_recon_loss if val_recon_loss is not None else float('nan'))
        self.val_kl_loss.append(val_kl_loss if val_kl_loss is not None else float('nan'))
        self.val_heatmap_loss.append(val_heatmap_loss if val_heatmap_loss is not None else float('nan'))
        self.val_occupancy_loss.append(val_occupancy_loss if val_occupancy_loss is not None else float('nan'))
        self.val_impedance_loss.append(val_impedance_loss if val_impedance_loss is not None else float('nan'))
        self.val_expert_kl_loss.append(val_expert_kl_loss if val_expert_kl_loss is not None else float('nan'))
        self.val_sigma_floor_loss.append(val_sigma_floor_loss if val_sigma_floor_loss is not None else float('nan'))
        self.val_physics_ri_loss.append(val_physics_ri_loss if val_physics_ri_loss is not None else float('nan'))
        self.val_physics_critic_sup_loss.append(val_physics_critic_sup_loss if val_physics_critic_sup_loss is not None else float('nan'))
        self.val_physics_ar_loss.append(val_physics_ar_loss if val_physics_ar_loss is not None else float('nan'))

        # Append new row to CSV
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                total_loss, recon_loss, kl_loss,
                heatmap_loss, occupancy_loss, impedance_loss,
                self.val_total_loss[-1], self.val_recon_loss[-1], self.val_kl_loss[-1],
                self.val_heatmap_loss[-1], self.val_occupancy_loss[-1], self.val_impedance_loss[-1],
                self.val_expert_kl_loss[-1], self.val_sigma_floor_loss[-1],
                self.val_physics_ri_loss[-1],
                self.val_physics_critic_sup_loss[-1], self.val_physics_ar_loss[-1],
            ])
    
    def log_dict(self, epoch: int, loss_dict: dict, val_loss_dict: dict = None): # type: ignore
        """
        Log train (and optionally validation) losses from loss dictionaries.

        Args:
            epoch: Epoch number
            loss_dict: Training loss dict with keys from VAELoss
            val_loss_dict: Validation loss dict (same keys). When provided, both
                           are written as a single row so overfitting can be tracked.
        """
        val_kwargs = {}
        if val_loss_dict is not None:
            val_kwargs = {
                'val_total_loss':      val_loss_dict['total_loss'],
                'val_recon_loss':      val_loss_dict['recon_loss'],
                'val_kl_loss':         val_loss_dict['kl_loss'],
                'val_heatmap_loss':    val_loss_dict['heatmap_loss'],
                'val_occupancy_loss':  val_loss_dict['occupancy_loss'],
                'val_impedance_loss':  val_loss_dict['impedance_loss'],
                'val_expert_kl_loss':  val_loss_dict.get('expert_kl_loss'),
                'val_sigma_floor_loss': val_loss_dict.get('sigma_floor_loss'),
                'val_physics_ri_loss':         val_loss_dict.get('physics_ri_loss'),
                'val_physics_critic_sup_loss':  val_loss_dict.get('physics_critic_sup_loss'),
                'val_physics_ar_loss':          val_loss_dict.get('physics_ar_loss'),
            }
        self.log(
            epoch=epoch,
            total_loss=loss_dict['total_loss'],
            recon_loss=loss_dict['recon_loss'],
            kl_loss=loss_dict['kl_loss'],
            heatmap_loss=loss_dict['heatmap_loss'],
            occupancy_loss=loss_dict['occupancy_loss'],
            impedance_loss=loss_dict['impedance_loss'],
            **val_kwargs
        )

    # ------------------------------------------------------------------
    # Latent stats logging (separate CSV, written every N epochs)
    # ------------------------------------------------------------------

    @staticmethod
    def build_latent_stats(val: dict) -> dict:
        """Build latent_stats dict for checkpoint and inference sampling."""
        ms  = val['modality_stats']
        pds = val.get('per_dim_stats', {}).get('latent', {})
        stats = {
            mod: {'mu_mean': ms[mod]['mu_mean'], 'mu_std': ms[mod]['mu_std'],
                  'sigma_mean': ms[mod]['std_mean']}
            for mod in ('heatmap', 'occupancy', 'impedance')
        }
        stats['latent'] = {
            'mu_mean':    val['mu_mean'],
            'mu_std':     val['mu_std'],
            'sigma_mean': val['std_mean'],
            **pds,
        }
        return stats

    def log_latent_stats(self, path: Path, epoch: int, beta: float, val: dict):
        """Append one row to the latent-stats CSV (same interval as checkpoint_interval).

        Reads existing rows, drops any at or after `epoch` (resume safety),
        then rewrites header + kept rows + new row atomically.
        """
        if not self.at_checkpoint_epoch(epoch):
            return
        ms  = val['modality_stats']
        row = [
            epoch, f"{beta:.6f}",
            *[f"{ms['heatmap'][k]:.4f}"   for k in ('mu_mean', 'std_mean')],
            *[f"{ms['occupancy'][k]:.4f}" for k in ('mu_mean', 'std_mean')],
            *[f"{ms['impedance'][k]:.4f}" for k in ('mu_mean', 'std_mean')],
            f"{val['mu_mean']:.4f}", f"{val['mu_std']:.4f}", f"{val['std_mean']:.4f}",
            f"{val['mu_min']:.4f}",  f"{val['mu_max']:.4f}",
            f"{val['kl_gaussian']:.4f}", f"{val['recon_loss']:.4f}", f"{val['total_loss']:.4f}",
        ]
        existing: list = []
        if path.exists():
            with open(path, newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for r in reader:
                    try:
                        if int(r[0]) < epoch:
                            existing.append(r)
                    except (ValueError, IndexError):
                        pass
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(LATENT_CSV_HEADER)
            w.writerows(existing)
            w.writerow(row)
    
    def plot(self, save_path: Optional[str] = None):
        """
        Generate and save training visualization plots
        
        Args:
            save_path: Optional custom path to save plots
        """
        if not self.epochs:
            print("No data to plot")
            return
        
        epochs = np.array(self.epochs)
        total_loss = np.array(self.total_loss)
        recon_loss = np.array(self.recon_loss)
        kl_loss = np.array(self.kl_loss)
        heatmap_loss = np.array(self.heatmap_loss)
        occupancy_loss = np.array(self.occupancy_loss)
        impedance_loss = np.array(self.impedance_loss)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total and component losses
        axes[0, 0].plot(epochs, total_loss, 'b-', label='Total Loss', linewidth=2)
        axes[0, 0].plot(epochs, recon_loss, 'g-', label='Reconstruction Loss', linewidth=2)
        axes[0, 0].plot(epochs, kl_loss, 'r-', label='KL Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss vs Components')
        axes[0, 0].set_xlim(left=1)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Reconstruction loss components
        axes[0, 1].plot(epochs, heatmap_loss, 'orange', label='Heatmap Loss', linewidth=2)
        axes[0, 1].plot(epochs, occupancy_loss, 'purple', label='Occupancy Loss', linewidth=2)
        axes[0, 1].plot(epochs, impedance_loss, 'brown', label='Impedance Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Modality-specific Reconstruction Losses')
        axes[0, 1].set_xlim(left=1)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # KL loss trend
        axes[1, 0].plot(epochs, kl_loss, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].set_title('KL Divergence Loss (Latent Space Regularization)')
        axes[1, 0].set_xlim(left=1)
        axes[1, 0].grid(alpha=0.3)
        
        # Loss ratio (recon vs KL)
        axes[1, 1].plot(epochs, recon_loss / (kl_loss + 1e-8), linewidth=2, color='cyan')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_title('Reconstruction to KL Loss Ratio')
        axes[1, 1].set_xlim(left=1)
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = str(self.log_dir / "convergence.png")
        
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot saved: {save_path}")

    def plot_loss_components(self, save_path: Optional[str] = None):
        """
        Generate detailed loss component visualization
        
        Args:
            save_path: Optional custom path to save plots
        """
        if not self.epochs:
            print("No data to plot")
            return
        
        epochs = np.array(self.epochs)
        heatmap_loss = np.array(self.heatmap_loss)
        occupancy_loss = np.array(self.occupancy_loss)
        impedance_loss = np.array(self.impedance_loss)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(epochs, heatmap_loss, 'o-', linewidth=2, markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Heatmap Reconstruction Loss (64x64x2)')
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(epochs, occupancy_loss, 's-', linewidth=2, markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Occupancy Reconstruction Loss (7x8x1)')
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(epochs, impedance_loss, '^-', linewidth=2, markersize=4)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Impedance Reconstruction Loss (231x1)')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = str(self.log_dir / "loss_components.png")
        
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Loss components plot saved: {save_path}")

    def plot_overfitting(self, save_path = None):
        if not self.epochs:
            print("No data to plot (overfitting)"); return
        epochs = np.array(self.epochs)
        val_total = np.array(self.val_total_loss)
        val_recon = np.array(self.val_recon_loss)
        val_kl = np.array(self.val_kl_loss)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(epochs, self.total_loss, "b-", label="Train", linewidth=2)
        axes[0].plot(epochs, val_total, "r--", label="Val", linewidth=2)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Total Loss (Train vs Val)")
        axes[0].set_xlim(left=1); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(epochs, self.recon_loss, "b-", label="Train", linewidth=2)
        axes[1].plot(epochs, val_recon, "r--", label="Val", linewidth=2)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
        axes[1].set_title("Reconstruction Loss (Train vs Val)")
        axes[1].set_xlim(left=1); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].plot(epochs, self.kl_loss, "b-", label="Train", linewidth=2)
        axes[2].plot(epochs, val_kl, "r--", label="Val", linewidth=2)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss")
        axes[2].set_title("KL Loss (Train vs Val)")
        axes[2].set_xlim(left=1); axes[2].legend(); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        if save_path is None:
            save_path = str(self.log_dir / "overfitting.png")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300); plt.close()
        print(f"Overfitting plot saved: {save_path}")

    def plot_physics(self, save_path = None):
        if not self.epochs:
            print("No data to plot (physics)"); return
        epochs = np.array(self.epochs)
        ri = np.array(self.val_physics_ri_loss)
        cs = np.array(self.val_physics_critic_sup_loss)
        ar = np.array(self.val_physics_ar_loss)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(epochs, ri, "m-", linewidth=2)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Physics RI Loss"); axes[0].set_xlim(left=1); axes[0].grid(alpha=0.3)
        axes[1].plot(epochs, cs, "c-", linewidth=2)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
        axes[1].set_title("Physics Critic Supervision"); axes[1].set_xlim(left=1); axes[1].grid(alpha=0.3)
        axes[2].plot(epochs, ar, "y-", linewidth=2)
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss")
        axes[2].set_title("Physics Anti-Resonance"); axes[2].set_xlim(left=1); axes[2].grid(alpha=0.3)
        plt.tight_layout()
        if save_path is None:
            save_path = str(self.log_dir / "physics_losses.png")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300); plt.close()
        print(f"Physics plot saved: {save_path}")

    def get_statistics(self) -> dict:
        """
        Get training statistics summary
        
        Returns:
            Dictionary with min, max, mean, and final values for each loss
        """
        stats = {}
        
        for loss_name, loss_values in [
            ('total_loss', self.total_loss),
            ('recon_loss', self.recon_loss),
            ('kl_loss', self.kl_loss),
            ('heatmap_loss', self.heatmap_loss),
            ('occupancy_loss', self.occupancy_loss),
            ('impedance_loss', self.impedance_loss),
        ]:
            if loss_values:
                arr = np.array(loss_values)
                stats[loss_name] = {
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'mean': float(np.mean(arr)),
                    'final': float(arr[-1]),
                }
        
        return stats
    
    def print_statistics(self):
        """Print training statistics summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("TRAINING STATISTICS SUMMARY")
        print("="*80)
        
        for loss_name, values in stats.items():
            print(f"\n{loss_name}:")
            print(f"  Min:   {values['min']:.6f}")
            print(f"  Max:   {values['max']:.6f}")
            print(f"  Mean:  {values['mean']:.6f}")
            print(f"  Final: {values['final']:.6f}")
        
        print("\n" + "="*80)

    # ------------------------------------------------------------------
    # Training-run console helpers
    # ------------------------------------------------------------------

    def log_start(self, exp_name: str, cfg_summary: dict):
        """Print the startup banner after logger construction."""
        print("=" * 80)
        print(f"{exp_name.upper()} — MULTI-INPUT VAE")
        for key, val in cfg_summary.items():
            print(f"  {key}: {val}")
        print("=" * 80)

    def log_epoch(
        self,
        epoch: int,
        num_epochs: int,
        beta: float,
        cur_lr: float,
        tr: dict,
        val: dict,
        physics=None,
        physics_warmup_epochs: int = 0,
    ):
        """Verbose per-epoch console print (every 10th epoch)."""
        ms = val['modality_stats']
        physics_stage = 2 if (physics is not None and epoch >= physics_warmup_epochs) else 1
        print(
            f"\n  Ep {epoch}  train={tr['total_loss']:.4f}  val={val['total_loss']:.4f}"
            f"  recon={val['recon_loss']:.4f}  kl={val['kl_loss']:.4f}"
            f"  sig_reg={val['sigma_reg_loss']:.4f}  lr={cur_lr:.2e}"
        )
        if physics is not None:
            print(
                f"  Physics[s{physics_stage}]"
                f" ri={val['physics_ri_loss']:.4f}"
                f" cs={val['physics_critic_sup_loss']:.4f}"
                f" ar={val['physics_ar_loss']:.4f}"
            )
        print(
            f"  Experts"
            f"  hm:{ms['heatmap']['mu_mean']:.3f}/{ms['heatmap']['std_mean']:.3f}"
            f"  occ:{ms['occupancy']['mu_mean']:.3f}/{ms['occupancy']['std_mean']:.3f}"
            f"  imp:{ms['impedance']['mu_mean']:.3f}/{ms['impedance']['std_mean']:.3f}"
        )
        print(
            f"  Fused"
            f"  \u03bc={val['mu_mean']:.3f}\u00b1{val['mu_std']:.3f}"
            f"  \u03c3={val['std_mean']:.3f}"
            f"  [{val['mu_min']:.2f},{val['mu_max']:.2f}]"
        )

    def log_epoch_minor(self, epoch: int, num_epochs: int, beta: float,
                        cur_lr: float, tr: dict, val: dict):
        """Brief single-line console print for non-verbose epochs."""
        print(
            f"  Ep {epoch}/{num_epochs}"
            f"  train={tr['total_loss']:.4f}  val={val['total_loss']:.4f}"
            f"  \u03b2={beta:.4f}  lr={cur_lr:.2e}"
        )

    def log_complete(self, best_val: float, ckpt_path, time_str: str):
        """Print the completion summary."""
        print(f"\n{'='*80}\nTRAINING COMPLETE  ({time_str})")
        print(f"Best val: {best_val:.4f}  |  Checkpoints: {ckpt_path}\n{'='*80}")

    def save_config(self, path, config_data: dict):
        """Dump config_data dict to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        print(f"Config saved: {path}")


if __name__ == '__main__':
    # Quick smoke-test: construct logger, log two epochs, verify no crash.
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        logger = VAETrainingLogger(log_dir=tmp)
        for ep in (1, 2):
            logger.log(ep, 1.0, 0.8, 0.2, 0.4, 0.0, 0.2, 0.2)
        print("Smoke-test passed.")
