"""Inference for exp040 FactorizedFreqVAE."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[3]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path
setup_path()

_EXP_DIR = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _EXP_DIR / "config.yaml"


def load_experiment_config(path: Path) -> dict:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))


def _default_data_dir() -> Path:
    if _CONFIG_PATH.is_file():
        cfg = load_experiment_config(_CONFIG_PATH)
        if cfg.get("data_dir"):
            return Path(cfg["data_dir"])
    return PROJECT_ROOT / "datasets" / "data_multifreq_norm"


def _norm_stats_path() -> Path:
    p = _default_data_dir() / "normalization_stats.json"
    if p.is_file():
        return p
    raise FileNotFoundError(f"normalization_stats.json not found at {p}")


from experiments.exp040.codes.vae_factorized_freq import FactorizedFreqVAE  # noqa: E402

exp = "experiments/exp040"
CHECKPOINT_PATH = f"{exp}/checkpoints/last_model.pt"
LATENT_STATS_PATH = f"{exp}/metrics/latent_stats.json"
MODEL_LATENT_DIM = 42
SHARED_TEMP = 1.5


class VAEInference:
    """Load FactorizedFreqVAE and run layout / anchor_blend inference."""

    def __init__(
        self,
        checkpoint_path: str,
        latent_dim: int = MODEL_LATENT_DIM,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        cfg_yaml = load_experiment_config(_CONFIG_PATH) if _CONFIG_PATH.is_file() else {}
        with open(_norm_stats_path(), encoding="utf-8") as f:
            ns = json.load(f)
        hm = ns["Heatmap"]
        self.hm_log_mean = hm["log_mean"]
        self.hm_log_std = hm["log_std"]
        self.hm_z_clip = (
            float(hm["clip_min"]),
            float(hm["clip_max"]),
        ) if hm.get("clip_min") is not None and hm.get("clip_max") is not None else None
        self.imp_log_std = ns["Impedance"]["log_std"]
        self.imp_log_mean = ns["Impedance"]["log_mean"]
        self.background_value = float(cfg_yaml.get("background_value", -2.9669))

        cfg_dir = PROJECT_ROOT / "configs"
        self.frequency = np.load(cfg_dir / "Frequency_data_hz.npy").squeeze()
        self.target_impedance = np.load(cfg_dir / "target_impedance.npy").squeeze()
        self.binary_mask = np.load(cfg_dir / "binary_mask.npy")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        ld = int(cfg.get("latent_dim", latent_dim))
        self.model = FactorizedFreqVAE(
            latent_dim=ld,
            cond_dim=int(cfg.get("cond_dim", 8)),
            heatmap_private_dim=int(cfg.get("heatmap_private_dim", 8)),
            modality_dropout=float(cfg.get("modality_dropout", 0.0)),
            freq_fourier_features=int(cfg.get("freq_fourier_features", 8)),
            use_heatmap_film=bool(cfg.get("use_heatmap_film", True)),
            num_freq_modes=int(cfg.get("num_freq_modes", cfg_yaml.get("num_freq_modes", 6))),
            freq_alpha_softmax=bool(cfg.get("freq_alpha_softmax", False)),
            use_freq_residual=bool(cfg.get("use_freq_residual", True)),
            residual_gain_init=float(cfg.get("residual_gain_init", 0.15)),
        )
        state = ckpt["model_state_dict"]
        ms = self.model.state_dict()
        compat = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
        self.model.load_state_dict(compat, strict=False)
        fac_only = bool(cfg.get("inference_factorized_only", cfg_yaml.get("inference_factorized_only", True)))
        self.model.inference_factorized_only = fac_only
        self.model.to(self.device).eval()
        print(
            f"Loaded FactorizedFreqVAE {len(compat)}/{len(state)} tensors from {checkpoint_path}"
            f"  (inference_factorized_only={fac_only})",
        )

        self.latent_stats = ckpt.get("latent_stats")
        self.per_K_latent_stats = ckpt.get("per_K_latent_stats")

    def load_latent_stats(self, latent_stats_path: Optional[str] = None) -> None:
        if latent_stats_path is None:
            auto = Path(self.checkpoint_path).parent.parent / "metrics" / "latent_stats.json"
            if auto.is_file():
                latent_stats_path = str(auto)
        if latent_stats_path and Path(latent_stats_path).is_file():
            with open(latent_stats_path, encoding="utf-8") as f:
                self.latent_stats = json.load(f)
            print(f"Loaded latent stats: {latent_stats_path}")
