"""Generate new VAE samples + a single ECADStar batch .peb for K=1..52.

This is a thin wrapper around `scrap/generate_samples_and_peb.py`.

Typical usage
    Edit the CONFIGURATION block below (optional), then run:
        python scrap/generate_samples_and_peb_all_k.py

Notes
- Creates one output folder per K:
    scrap/generated_samples/K{K}/
- Writes one combined .peb containing all samples (in K order):
    scrap/PEB/K1_to_K52.peb
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRAP_DIR = Path(__file__).resolve().parent
if str(SCRAP_DIR) not in sys.path:
    sys.path.insert(0, str(SCRAP_DIR))

from scrap.generation import generate_samples_and_peb as gsp


# ============================================================
# CONFIGURATION (optional overrides)
# ============================================================
K_MIN = 1
K_MAX = 52

# If you want a single root folder for all K outputs, set it here.
# By default this matches `generate_samples_and_peb.py`'s convention.
OUT_ROOT = Path("scrap/generated_samples_v2")  # will contain subfolders K1, K2, ..., K52

# Where to write the combined .peb file (separate from sample folders).
# Defaults to the same setting as `generate_samples_and_peb.py`.
PEB_OUT_DIR = Path(gsp.PEB_PATH) if getattr(gsp, "PEB_PATH", "") else Path("scrap/PEB")
PEB_OUT_FILE = PEB_OUT_DIR / f"K{K_MIN}_to_K{K_MAX}.peb"


def main() -> None:
    if not (0 <= K_MIN <= 52 and 0 <= K_MAX <= 52 and K_MIN <= K_MAX):
        raise SystemExit("Expected 0 <= K_MIN <= K_MAX <= 52")

    device = torch.device("cpu" if gsp.FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Ensure relative paths work regardless of where the script is launched from
    os.chdir(gsp.PROJECT_ROOT)

    engine = gsp.VAEInference(
        checkpoint_path=gsp.CHECKPOINT_PATH,
        latent_dim=gsp.MODEL_LATENT_DIM,
        device=device,
    )
    engine.load_latent_stats(gsp.LATENT_STATS_PATH or None)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PEB_OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_occupancies: list[np.ndarray] = []

    for k in range(K_MIN, K_MAX + 1):
        out_dir = OUT_ROOT / f"K{k}"
        print(f"\n=== K={k} → {out_dir} ===")

        occ = gsp.generate_save(
            engine,
            num_samples=gsp.NUM_SAMPLES,
            out_dir=out_dir,
            K=k,
            shared_temp=gsp.SHARED_TEMP,
        )

        # Keep the order: K asc, then sample index asc.
        all_occupancies.append(occ)

    occupancy_all = np.concatenate(all_occupancies, axis=0) if all_occupancies else np.zeros((0, 52), dtype=np.int8)

    gsp.generate_peb(
        occupancy=occupancy_all,
        output_path=str(PEB_OUT_FILE),
        powerbus=gsp.POWERBUS,
        freq=gsp.FREQ,
        components=gsp.COMPONENTS,
    )

    print("\n✓ Done")
    print(f"  Outputs under: {OUT_ROOT}")
    print(f"  Combined PEB:  {PEB_OUT_FILE}")


if __name__ == "__main__":
    main()
