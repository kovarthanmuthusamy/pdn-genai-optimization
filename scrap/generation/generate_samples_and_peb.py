"""Generate VAE samples and PEB for one K.

Purpose:
    Load VAE checkpoint, generate N samples with exactly K active decaps, save ``data_sample_*``
    folders, and write a matching ECADStar ``.peb``.

Run:
    python scrap/generation/generate_samples_and_peb.py

Agent notes:
    - What: Single-K VAE sample export + PEB builder (legacy exp030 path).
    - Usage: Set ``K_VALUE``, ``NUM_SAMPLES``, checkpoint paths, ``OUTPUT_DIR`` → run.
    - Config keys:
        - ``CHECKPOINT_PATH``, ``LATENT_STATS_PATH``, ``MODEL_LATENT_DIM`` — VAE load
        - ``K_VALUE``, ``NUM_SAMPLES``, ``SHARED_TEMP`` — generation
        - ``OUTPUT_DIR``, ``PEB_PATH`` — where ``.npy`` and ``.peb`` are written
    - Key symbols: ``generate_save``, ``main``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# CONFIGURATION — edit these before running: python scrap/generation/generate_samples_and_peb.py
# =============================================================================
CHECKPOINT_PATH = "experiments/exp030_adding_physic/checkpoints/checkpoint_epoch_400.pt"

# Optional: if empty, inference will use checkpoint-embedded latent stats (if present)
LATENT_STATS_PATH = ""

MODEL_LATENT_DIM = 32  # hint only; checkpoint may override via embedded config

NUM_SAMPLES = 5
K_VALUE = 5  # exactly K occupied capacitor slots per sample (0..52)
SHARED_TEMP = 1.5  # higher = more diversity; 1.0 = raw posterior stats

OUTPUT_DIR = f"scrap/generated_samples_v2/K{K_VALUE}"  # will be created if it doesn't exist

# PEB settings
# - If empty -> writes to <OUTPUT_DIR>/K{K_VALUE}.peb
# - If points to a directory (recommended) -> writes to <PEB_PATH>/K{K_VALUE}.peb
# - If points to a .peb file -> writes exactly to that file
PEB_PATH = "scrap/PEB"
POWERBUS = "Power_GND"
FREQ = "63e6"
COMPONENTS = "IC1_Port1"

FORCE_CPU = False

# =============================================================================


def _add_project_root_to_syspath() -> Path:
    from repo_paths import REPO_ROOT, setup_path
    setup_path()
    return REPO_ROOT


PROJECT_ROOT = _add_project_root_to_syspath()

# Import after sys.path tweak
from experiments.exp030_adding_physic.codes.inference_vae import VAEInference  # noqa: E402
from scrap.generation.generate_peb import generate_peb  # noqa: E402


def generate_save(
    engine: VAEInference,
    *,
    num_samples: int,
    out_dir: str | Path,
    K: int,
    shared_temp: float,
) -> np.ndarray:
    """Generate samples and save them to disk.

    Returns the stacked binary occupancy array of shape (num_samples, 52).
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        heatmap_zscore, occ_prob, impedance_norm = engine.model.inference(
            num_samples,
            engine.device,
            K=K,
            latent_stats=engine.latent_stats,
            per_K_latent_stats=engine.per_K_latent_stats,
            shared_temp=shared_temp,
        )

        # Activate exactly K highest-probability slots per sample
        occ_bin = torch.zeros_like(occ_prob)
        if K > 0:
            topk_idx = occ_prob.topk(min(K, occ_prob.shape[-1]), dim=-1).indices
            occ_bin.scatter_(-1, topk_idx, 1.0)

        impedance_log, _, _, _ = engine._denorm_impedance(impedance_norm)

        heatmap_physical = (torch.exp(heatmap_zscore * engine.hm_log_std + engine.hm_log_mean) - 1.0).clamp(min=0.0)

    # Move to CPU for saving
    heatmap_zscore_np = heatmap_zscore.cpu().numpy()
    heatmap_phys_np = heatmap_physical.cpu().numpy()
    occ_np = occ_bin.cpu().numpy()
    imp_np = impedance_log.cpu().numpy()

    # Save combined occupancy for convenience
    np.save(out_path / "occupancy.npy", occ_np.astype(np.int8))

    for i in range(num_samples):
        sample_dir = out_path / f"data_sample_{i}"
        sample_dir.mkdir(exist_ok=True)
        np.save(sample_dir / "heatmap_zscore.npy", heatmap_zscore_np[i])
        np.save(sample_dir / "heatmap_physical.npy", heatmap_phys_np[i])
        np.save(sample_dir / "occupancy_map.npy", occ_np[i].astype(np.int8))
        np.save(sample_dir / "impedance_profile.npy", imp_np[i])

    return occ_np.astype(np.int8)


def main() -> None:
    if NUM_SAMPLES <= 0:
        raise SystemExit("NUM_SAMPLES must be > 0")
    if not (0 <= K_VALUE <= 52):
        raise SystemExit("K_VALUE must be in [0, 52]")

    device = torch.device("cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Ensure relative paths work regardless of where the script is launched from
    os.chdir(PROJECT_ROOT)

    engine = VAEInference(checkpoint_path=CHECKPOINT_PATH, latent_dim=MODEL_LATENT_DIM, device=device)
    engine.load_latent_stats(LATENT_STATS_PATH or None)

    occ = generate_save(
        engine,
        num_samples=NUM_SAMPLES,
        out_dir=OUTPUT_DIR,
        K=K_VALUE,
        shared_temp=SHARED_TEMP,
    )

    out_path = Path(OUTPUT_DIR)
    # IMPORTANT: Generate the batch .peb from the saved occupancy file to guarantee
    # that the .peb ordering matches the saved sample folders:
    #   occupancy.npy row i  <->  data_sample_i/occupancy_map.npy
    occupancy_path = out_path / "occupancy.npy"
    occ_for_peb = np.load(occupancy_path)
    
    if PEB_PATH:
        peb_path_cfg = Path(PEB_PATH)
        peb_path = peb_path_cfg if peb_path_cfg.suffix.lower() == ".peb" else (peb_path_cfg / f"K{K_VALUE}.peb")
    else:
        peb_path = out_path / f"K{K_VALUE}.peb"
    peb_path.parent.mkdir(parents=True, exist_ok=True)

    generate_peb(
        occupancy=occ_for_peb,
        output_path=str(peb_path),
        powerbus=POWERBUS,
        freq=FREQ,
        components=COMPONENTS,
    )

    print("\n✓ Done")
    print(f"  Samples saved under: {out_path}")
    print(f"  Occupancy saved as:  {out_path / 'occupancy.npy'}")
    print(f"  PEB written to:      {peb_path}")


if __name__ == "__main__":
    main()
