"""Quick smoke test: exp044 VAEInference loads for sweep."""
import sys
from pathlib import Path

from repo_paths import REPO_ROOT, setup_path

setup_path()

import torch
import scrap.generation.run_multifreq_heatmap_sweep as sweep

sweep.EXPERIMENT_DIR = "experiments/exp044"
sweep.CHECKPOINT_PATH = "experiments/exp044/checkpoints/last_model.pt"
sweep.DATA_DIR = "datasets/data_multifreq_norm_z_score"

device = torch.device("cpu")
engine = sweep._load_engine(device)
print("OK:", type(engine).__module__, "latent_dim", engine.latent_dim)
