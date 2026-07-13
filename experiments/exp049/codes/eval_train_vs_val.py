"""Diagnostic: compare layout_cross vs encode_cross on TRAIN vs VAL designs.

Answers: is poor layout generation a generalization problem (train >> val)
or a capacity problem (train ~= val)?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders
from experiments.exp049.codes.eval_spatial_metrics import run_off_anchor_eval_spatial
from experiments.exp049.codes.exp049_eval_common import load_model

EXP_DIR = Path(__file__).resolve().parents[1]


def _load_cfg() -> dict:
    path = EXP_DIR / "config.yaml"
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return json.loads("\n".join(lines))


def main() -> None:
    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ep = 300
    ckpt = EXP_DIR / "checkpoints" / f"checkpoint_epoch_{ep}.pt"
    if not ckpt.is_file():
        # fall back to best spatial / last
        for alt in ("best_encode_cross_spatial.pt", "last_model.pt"):
            p = EXP_DIR / "checkpoints" / alt
            if p.is_file():
                ckpt = p
                break
    print(f"Using checkpoint: {ckpt}")

    model, mcfg, _, _ = load_model(ckpt, device)
    model.eval()

    data_dir = cfg["data_dir"]
    bg = float(cfg.get("background_value", -3.1922))
    off = tuple(float(x) for x in cfg.get("eval_off_anchor_mhz", [100.0, 175.0, 330.0, 400.0]))

    train_ld, val_ld = create_multifreq_data_loaders(
        data_dir=data_dir,
        batch_size=int(cfg.get("val_batch_size", 160)),
        num_workers=0,
        train_split=float(cfg.get("train_split", 0.9)),
        seed=42,
        pin_memory=False,
        split_by_design=bool(cfg.get("split_by_design", True)),
        balance_k=bool(cfg.get("balance_k", True)),
        balance_freq=bool(cfg.get("balance_freq", True)),
        k_balance_power=float(cfg.get("k_balance_power", 0.35)),
        freq_balance_power=float(cfg.get("freq_balance_power", 1.2)),
        k_balance_smoothing=float(cfg.get("k_balance_smoothing", 1e-3)),
        stratify_by_k=bool(cfg.get("stratify_by_k", True)),
        cache_in_ram=True,
        train_samples_per_epoch=int(cfg.get("train_samples_per_epoch", 20000)),
        cross_freq_pairs=False,
        val_batch_size=int(cfg.get("val_batch_size", 160)),
    )

    out_dir = EXP_DIR / "metrics"
    print("\n" + "=" * 70)
    print("VAL designs (held-out)")
    print("=" * 70)
    run_off_anchor_eval_spatial(
        model, val_ld, bg=bg, off_anchor_mhz=off, max_batches=32, device=device,
        out_csv=out_dir / "diag_val_split.csv", early_stop_state=None,
    )

    print("\n" + "=" * 70)
    print("TRAIN designs (seen during training)")
    print("=" * 70)
    run_off_anchor_eval_spatial(
        model, train_ld, bg=bg, off_anchor_mhz=off, max_batches=32, device=device,
        out_csv=out_dir / "diag_train_split.csv", early_stop_state=None,
    )

    print("\nDone. CSVs: diag_val_split.csv, diag_train_split.csv")


if __name__ == "__main__":
    main()
