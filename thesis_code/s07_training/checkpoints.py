"""Checkpoints, resume and fresh-run cleanup, under ``<experiment>/checkpoints/``.

    checkpoint_epoch_<N>.pt     every ``checkpoint_interval`` epochs (and epoch 1)
    best_off_anchor_model.pt    highest cross-frequency score (training/crossfreq_eval.py)
    last_model.pt               always the final epoch

A checkpoint holds the model / optimizer / scheduler states, the config, the validation metrics and
the aggregate-posterior statistics (``latent_stats``, ``per_K_latent_stats``) the latent search
samples from.

Usage (import only):
resolve_resume(c)                      resume_checkpoint (epoch / "last" / "latest" / path) -> path
load_model_weights(path, model, dev)   -> (epoch, val_loss, saved config); skips mismatched tensors
checkpoint_dict(epoch, model, opt, sched, train_loss, val, c, latent_stats, per_k)
clean_previous_run(exp_dir)            fresh run: delete checkpoints/, logs/, metrics/, visuals_*
                                       (only inside OUTPUT_DIR; anywhere else it stops)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch

from thesis_code.common import settings
from thesis_code.s06_losses.objective import LOSS_KEYS
from thesis_code.s07_training.distributed import model_state_dict


def resolve_resume(c) -> None:
    rc = c.resume_checkpoint
    d = Path(c.experiment_dir) / "checkpoints"
    if isinstance(rc, int):
        c.resume_checkpoint = str(d / f"checkpoint_epoch_{rc}.pt")
    elif isinstance(rc, str) and rc.lower() in ("last", "last_model"):
        c.resume_checkpoint = str(d / "last_model.pt")
    elif isinstance(rc, str) and rc.lower() == "latest":
        interval = sorted(d.glob("checkpoint_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        if interval:
            c.resume_checkpoint = str(interval[-1])
        elif (d / "last_model.pt").exists():
            c.resume_checkpoint = str(d / "last_model.pt")
        else:
            raise FileNotFoundError(f"No checkpoints in {d}")


def load_model_weights(path: str, model, device) -> tuple[int, float, dict]:
    """Load matching tensors (name and shape) into ``model``. Raises if none match."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    msd = model.state_dict()
    compat = {k: v for k, v in ckpt["model_state_dict"].items() if k in msd and v.shape == msd[k].shape}
    if not compat:
        raise RuntimeError(f"{path}: no tensor matches this model")
    msd.update(compat)
    model.load_state_dict(msd)
    ep, val_loss = int(ckpt.get("epoch", 0)), float(ckpt.get("val_loss", float("nan")))
    print(f"\n{'=' * 56}\n  RESUMING from checkpoint {Path(path).name}\n  completed epoch: {ep}\n"
          f"  val_loss: {val_loss:.4f}\n{'=' * 56}\n", flush=True)
    return ep, val_loss, ckpt.get("config") or {}


def checkpoint_dict(epoch, model, opt, sched, train_loss, val, c, latent_stats, per_k) -> dict:
    return {
        "epoch": epoch, "train_loss": train_loss, "val_loss": val["total_loss"],
        "val_metrics": {k: float(val[k]) for k in LOSS_KEYS if k in val},
        "config": c.to_checkpoint_dict(), "latent_stats": latent_stats, "per_K_latent_stats": per_k,
        "model_state_dict": model_state_dict(model), "optimizer_state_dict": opt.state_dict(),
        "mu_mean": val["mu_mean"], "mu_std": val["mu_std"], "mu_min": val["mu_min"], "mu_max": val["mu_max"],
        "scheduler_state_dict": sched.state_dict(),
    }


def clean_previous_run(exp: Path) -> None:
    out = Path(settings.OUTPUT_DIR).resolve()
    if out not in Path(exp).resolve().parents and Path(exp).resolve() != out:
        raise SystemExit(f"refusing to delete training outputs in {exp}: it is not inside OUTPUT_DIR ({out})")
    removed = []
    for name in ("checkpoints", "logs", "metrics"):
        if (exp / name).exists():
            shutil.rmtree(exp / name)
            removed.append(name + "/")
    for p in sorted(exp.glob("visuals_*")):
        if p.is_dir():
            shutil.rmtree(p)
            removed.append(p.name + "/")
    print(f"Fresh run -- removed previous outputs under {exp}: {', '.join(removed)}" if removed
          else f"Fresh run -- no prior training artifacts in {exp}")
