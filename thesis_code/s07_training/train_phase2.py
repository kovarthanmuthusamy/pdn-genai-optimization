"""Train the placement-only encoder (second phase) on the frozen VAE. Run as a module from the repo root.

What: step 1 fits the distiller (placement -> latent of the full encoder), step 2 fine-tunes it with
    the spectrum decoder into the head NAME (the thesis recipe: w_imp 1, w_hm 1, w_anchor 0.01, 8,000
    steps). Both write to OUTPUT_DIR/s07_training/phase2/. To use the new files, copy them into
    WEIGHTS_DIR (occ_to_z_distill.pt, vae_route_finetune/<NAME>.pt) and measure the margin
    (python -m thesis_code.s09_inference.margin). Needs a GPU (about an hour on one; days on a CPU).
Usage: python -m thesis_code.s07_training.train_phase2
Config keys: NAME, DISTILL_STEPS, N_TRAIN, STEPS, W_IMP, W_HM, W_ANCHOR, ALLOW_CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from thesis_code.common import settings
from thesis_code.s07_training.phase2 import finetune, train_distiller

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
NAME = settings.HEAD
DISTILL_STEPS, N_TRAIN = 20000, 16000
STEPS = 8000
W_IMP, W_HM, W_ANCHOR = 1.0, 1.0, 0.01
ALLOW_CPU = False
# =============================================================================


def main() -> None:
    if not torch.cuda.is_available() and not ALLOW_CPU:
        raise SystemExit("no GPU visible; set ALLOW_CPU = True to train on the CPU (very slow)")
    d = train_distiller(steps=DISTILL_STEPS, n_train=N_TRAIN)
    print(json.dumps(d, indent=1))
    out = finetune(NAME, steps=STEPS, w_imp=W_IMP, w_hm=W_HM, w_anchor=W_ANCHOR, distill=Path(d["path"]))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
