"""Quick latent-dimension traversal smoke test.

Run: python pipelines/analysis/quick_traversal.py"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from repo_paths import repo_path, setup_path

setup_path()

SOURCE_DIR = repo_path("source")
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from source.model.vae_multi_input import MultiInputVAE

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/analysis/quick_traversal.py
# =============================================================================

CHECKPOINT_PATH = repo_path("experiments/exp012/checkpoints/epoch_100.pt")
LATENT_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = repo_path("temp_visuals/quick_traversal_test")
TEST_DIMENSION = 0
TRAVERSAL_VALUES = np.linspace(-2, 2, 5)

# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = MultiInputVAE(latent_dim=LATENT_DIM)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

    z_base = torch.zeros(1, LATENT_DIM, device=DEVICE)
    results = []
    with torch.no_grad():
        for value in TRAVERSAL_VALUES:
            z = z_base.clone()
            z[0, TEST_DIMENSION] = float(value)
            hm, occ_logits, imp = model.decode(z)
            results.append({
                "value": value,
                "heatmap": hm.cpu().numpy()[0],
                "occupancy": torch.sigmoid(occ_logits).cpu().numpy()[0],
                "impedance": imp.cpu().numpy()[0],
            })

    fig, axes = plt.subplots(3, len(results), figsize=(4 * len(results), 12))
    for i, r in enumerate(results):
        v = r["value"]
        axes[0, i].imshow(r["heatmap"][0], cmap="jet", vmin=0, vmax=1)
        axes[0, i].set_title(f"Heatmap z[{TEST_DIMENSION}]={v:.1f}")
        axes[0, i].axis("off")
        axes[1, i].imshow((r["occupancy"][0] > 0.5).astype(float), cmap="Greys", vmin=0, vmax=1)
        axes[1, i].set_title(f"Occupancy")
        axes[1, i].axis("off")
        axes[2, i].plot(r["impedance"])
        axes[2, i].set_title("Impedance")
        axes[2, i].set_ylim(0, 1)

    fig.suptitle(f"Quick Traversal — dim {TEST_DIMENSION}", fontsize=16)
    fig.tight_layout()
    out = OUTPUT_DIR / "quick_test.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
