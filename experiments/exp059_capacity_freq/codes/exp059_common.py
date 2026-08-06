"""exp059 shared paths, yaml config, and VAE constructor kwargs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parents[1]


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    path = path or (EXP_DIR / "config.yaml")
    if not path.is_file():
        return {}
    lines = [
        ln for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return json.loads("\n".join(lines))


def _cfg_get(cfg: dict[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def vae_model_kwargs(cfg: dict[str, Any] | Any) -> dict[str, Any]:
    g = lambda k, d=None: _cfg_get(cfg, k, d)
    return {
        "latent_dim": int(g("latent_dim", 42)),
        "cond_dim": int(g("cond_dim", 8)),
        "heatmap_private_dim": int(g("heatmap_private_dim", 8)),
        "modality_dropout": float(g("modality_dropout", 0.0)),
        "freq_fourier_features": int(g("freq_fourier_features", 8)),
        "use_heatmap_film": bool(g("use_heatmap_film", True)),
        "use_multiscale_film": bool(g("use_multiscale_film", True)),
        "use_freq_poe_expert": bool(g("use_freq_poe_expert", True)),
        "use_occ_spatial_decoder": bool(g("use_occ_spatial_decoder", True)),
        "use_occ_spatial_tower": bool(g("use_occ_spatial_tower", True)),
        "occ_spatial_ch": int(g("occ_spatial_ch", 8)),
        "use_layout_private_head": bool(g("use_layout_private_head", True)),
        "layout_private_hidden": int(g("layout_private_hidden", 384)),
        "use_layout_private_freq_film": bool(g("use_layout_private_freq_film", True)),
        "graph_hidden_dim": int(g("graph_hidden_dim", 128)),
        "graph_num_layers": int(g("graph_num_layers", 3)),
        "graph_dropout": float(g("graph_dropout", 0.1)),
        "imp_peak_dim": int(g("imp_peak_dim", 8)),
    }
