"""Train VAE for exp039_improved_heatmap.

Uses ``experiments/exp038_true_multi/codes/train_vae_simple.py`` with overrides from
``config.yaml`` (VAE_EXPERIMENT_DIR). Adds optional synthetic mid-MHz heatmap blends.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[3]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path
setup_path()

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
from experiments.exp039_improved_heatmap.codes.synthetic_freq_blend import (
    maybe_apply_synthetic_blend,
)

_orig_prepare = _tr._prepare_batch


@dataclass
class Config(_tr.Config):
    synthetic_blend_prob: float = 0.0


def _prepare_batch(batch: dict, c: _tr.Config) -> tuple:
    return _orig_prepare(maybe_apply_synthetic_blend(batch, c), c)


def _patch_training() -> None:
    _tr.Config = Config
    _tr._prepare_batch = _prepare_batch


def train_vae() -> None:
    _patch_training()
    _tr.train_vae()


def main() -> None:
    exp_dir = Path(__file__).resolve().parents[1]
    os.environ.setdefault("VAE_EXPERIMENT_DIR", str(exp_dir))
    os.environ.setdefault("VAE_DATA_DIR", str(_PROJECT_ROOT / "datasets" / "data_multifreq_norm"))
    train_vae()


if __name__ == "__main__":
    main()
