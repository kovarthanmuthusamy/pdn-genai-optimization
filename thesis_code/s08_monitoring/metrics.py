"""Metric files of a training run, under ``<experiment>/metrics/``.

    loss.csv, latent_stats.csv   written by ``logger.VAETrainingLogger``
    epoch_timing.csv             every epoch: seconds per phase, train / val loss
    impedance_split.csv          checkpoints: impedance loss = legacy (shape) + peak-index part
    heatmap_peak_split.csv       checkpoints: heatmap loss before / after the peak-location term
    cycle_consistency.csv        checkpoints: train-only cycle loss
    anticopy.csv                 checkpoints: anti-copy and cross-frequency losses
    crossfreq_eval.csv           eval epochs (crossfreq_eval.py)
    timing.json                  total training time (read back on resume)

Usage (import only):
EPOCH_TIMING_HEADER, RESUMABLE_CSVS
append_epoch_csv(path, epoch, header, row)   append, replacing any row of the same epoch
truncate_csv_after_epoch(path, epoch)        resume: drop rows past the resume epoch
log_epoch_metrics(logger, obs, dir, ep, beta, tr, val, run_val)   everything above for one epoch
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from thesis_code.s06_losses.objective import LOSS_KEYS

EPOCH_TIMING_HEADER = ["epoch", "train_sec", "val_sec", "total_sec", "val_ran", "train_loss", "val_loss"]
RESUMABLE_CSVS = (
    "loss.csv", "epoch_timing.csv", "impedance_split.csv", "heatmap_peak_split.csv",
    "latent_stats.csv", "cycle_consistency.csv", "anticopy.csv", "crossfreq_eval.csv",
)
_TRAIN_LOG_KEYS = ("total_loss", "recon_loss", "kl_loss", "heatmap_loss", "occupancy_loss", "impedance_loss",
                   "impedance_legacy_loss", "impedance_peak_loss", "cycle_consistency_loss")


def _row_epoch(row: list[str], col: int = 0) -> int | None:
    try:
        return int(float(row[col]))
    except (TypeError, ValueError, IndexError):
        return None


def append_epoch_csv(path: Path, epoch: int, header: list[str], row: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[list[str]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            existing = [r for r in reader if _row_epoch(r) not in (None, epoch)]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(existing)
        w.writerow(row)
    tmp.replace(path)


def truncate_csv_after_epoch(path: Path, max_epoch: int) -> int:
    """Drop rows with epoch > ``max_epoch``. Returns the number dropped."""
    if not path.is_file() or max_epoch < 0:
        return 0
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    if not rows:
        return 0
    header, body = rows[0], [r for r in rows[1:] if r]
    ep_idx = header.index("epoch") if "epoch" in header else 0
    kept = [r for r in body if (_row_epoch(r, ep_idx) or 0) <= max_epoch]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
    tmp.replace(path)
    return len(body) - len(kept)


def _append_split_csvs(metrics_dir: Path, epoch: int, tr: dict, val: dict | None) -> None:
    v = val or {}
    append_epoch_csv(
        metrics_dir / "impedance_split.csv", epoch,
        ["epoch", "train_impedance_loss", "train_impedance_legacy", "train_impedance_peak",
         "val_impedance_loss", "val_impedance_legacy", "val_impedance_peak"],
        [epoch, tr.get("impedance_loss", ""), tr.get("impedance_legacy_loss", ""), tr.get("impedance_peak_loss", ""),
         v.get("impedance_loss", ""), v.get("impedance_legacy_loss", ""), v.get("impedance_peak_loss", "")])
    append_epoch_csv(
        metrics_dir / "heatmap_peak_split.csv", epoch,
        ["epoch", "train_heatmap_loss", "train_tier_a", "train_peak_loc", "val_heatmap_loss", "val_tier_a", "val_peak_loc"],
        [epoch, tr.get("heatmap_loss", ""), tr.get("heatmap_loss_tier_a", ""), tr.get("heatmap_peak_loc_loss", ""),
         v.get("heatmap_loss", ""), v.get("heatmap_loss_tier_a", ""), v.get("heatmap_peak_loc_loss", "")])
    append_epoch_csv(metrics_dir / "cycle_consistency.csv", epoch, ["epoch", "train_cycle_consistency_loss"],
                     [epoch, tr.get("cycle_consistency_loss", "")])
    append_epoch_csv(
        metrics_dir / "anticopy.csv", epoch,
        ["epoch", "train_anticopy_loss", "train_cross_freq_heatmap_loss", "val_anticopy_loss"],
        [epoch, tr.get("anticopy_loss", ""), tr.get("cross_freq_heatmap_loss", ""), v.get("anticopy_loss", "")])


def log_epoch_metrics(logger, obs, metrics_dir: Path, ep: int, beta: float, tr: dict, val: dict, run_val: bool) -> None:
    """loss.csv, the event log / MLflow, the split CSVs and latent_stats.csv for one epoch."""
    # the shared logger reads a "kl_gaussian" column; it is the same value as kl_loss
    tr_d = {**{k: tr[k] for k in _TRAIN_LOG_KEYS}, "kl_gaussian": tr["kl_loss"]}
    if run_val:
        val = {**val, "kl_gaussian": val["kl_loss"]}
    val_keys = [k for k in LOSS_KEYS if k != "cycle_consistency_loss"] + ["kl_gaussian"]   # cycle is train-only
    val_d = {k: val[k] for k in val_keys} if run_val else None
    logger.log_dict(ep, loss_dict=tr_d, val_loss_dict=val_d)
    if obs is not None:
        obs.log_losses(ep, train=tr_d, val=val_d)
    _append_split_csvs(metrics_dir, ep, tr, val if run_val else None)
    logger.log_latent_stats(metrics_dir / "latent_stats.csv", ep, beta, val)
