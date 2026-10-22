"""Load the trained VAE for inference: ``VAEInference().model``.

What: the model is rebuilt from the checkpoint's saved config and loaded strictly (every tensor must
    match). The normalization (map statistics per frequency, spectrum log mean / std, background
    value) is the dataset's, shipped in common/inputs/normalization_stats.json.
Usage (import only):
    vae = VAEInference(checkpoint_path=None, device=None)   # None: settings.WEIGHTS_DIR / settings.CHECKPOINT
    vae.model                           MultiInputVAE in eval mode
    vae.norm_stats / vae.hm_stats       statistics bundle / map statistics
    vae.imp_log_mean / vae.imp_log_std  spectrum normalization
    vae.denorm_heatmap_physical(hm, mhz=)   model map -> ohm (``mhz`` a tensor for mixed frequencies)
    vae.latent_stats / vae.per_K_latent_stats   aggregate posterior saved in the checkpoint
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.common.paths import need
from thesis_code.s03_normalization.stats import HeatmapNormStats, NormStatsBundle, load_norm_stats
from thesis_code.s05_model import MultiInputVAE
from thesis_code.s07_training.config import CONFIG_FILE, load_experiment_config_dict, vae_model_kwargs


class VAEInference:
    """The trained model in eval mode plus the normalization needed to read its outputs.

    Attributes: ``model``, ``device``, ``norm_stats`` / ``hm_stats``, ``imp_log_mean`` / ``imp_log_std``,
    ``background_value``, ``frequency`` (231 Hz values), ``binary_mask`` (board), ``latent_stats`` and
    ``per_K_latent_stats``.
    """

    def __init__(self, checkpoint_path: str | Path | None = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_path is None:
            checkpoint_path = need(Path(settings.WEIGHTS_DIR), "WEIGHTS_DIR") / settings.CHECKPOINT
        self.checkpoint_path = str(need(Path(checkpoint_path), "WEIGHTS_DIR / CHECKPOINT"))
        self.norm_stats: NormStatsBundle = load_norm_stats(None)
        self.hm_stats: HeatmapNormStats = self.norm_stats.heatmap
        self.imp_log_mean = self.norm_stats.impedance.log_mean
        self.imp_log_std = self.norm_stats.impedance.log_std
        self.background_value = float(self.norm_stats.raw.get("background_value", self.hm_stats.background_value))
        self.frequency = np.load(INPUTS / "Frequency_data_hz.npy").squeeze()
        self.binary_mask = np.load(INPUTS / "binary_mask.npy")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_cfg = {k: v for k, v in (ckpt.get("config") or {}).items() if v is not None}
        kw = vae_model_kwargs({**load_experiment_config_dict(CONFIG_FILE), **ckpt_cfg})
        self.model = MultiInputVAE(**kw)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.latent_dim = kw["latent_dim"]
        self.latent_stats = ckpt.get("latent_stats", None)
        self.per_K_latent_stats = ckpt.get("per_K_latent_stats", None)
        print(f"Loaded {Path(self.checkpoint_path).name}: latent_dim={kw['latent_dim']} cond_dim={kw['cond_dim']} "
              f"types={kw['n_occ_classes']}")

    def load_latent_stats(self, latent_stats_path: Optional[str] = None):
        """Replace the checkpoint's latent statistics with ``metrics/latent_stats.json`` if present."""
        if latent_stats_path is None:
            auto = Path(self.checkpoint_path).parent.parent / "metrics" / "latent_stats.json"
            latent_stats_path = str(auto) if auto.exists() else None
        if latent_stats_path and Path(latent_stats_path).exists():
            with open(latent_stats_path, encoding="utf-8") as f:
                self.latent_stats = json.load(f)

    def denorm_heatmap_physical(self, hm_norm: torch.Tensor, *, mhz: float | torch.Tensor | None = None,
                                pi_norm: torch.Tensor | None = None) -> torch.Tensor:
        """Model map output -> ohm with the statistics at ``mhz`` (or the frequency in ``pi_norm``)."""
        return self.hm_stats.norm_to_physical(hm_norm, mhz=mhz, pi_norm=pi_norm)
