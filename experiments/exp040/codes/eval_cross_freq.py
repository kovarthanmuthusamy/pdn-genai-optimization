"""Off-anchor eval for exp040 FactorizedFreqVAE."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_EXP = Path(__file__).resolve().parents[1]
os.environ.setdefault("VAE_EXPERIMENT_DIR", str(_EXP))

import experiments.exp038_true_multi.codes.eval_cross_freq as _ec
from experiments.exp040.codes.inference_vae import VAEInference

_ec.EXP_DIR = _EXP
_ec.DEFAULT_CKPT = _EXP / "checkpoints/last_model.pt"
_ec.OUT_CSV = _EXP / "metrics/cross_freq_eval.csv"
_ec.VAEInference = VAEInference

if __name__ == "__main__":
    _ec.main()
