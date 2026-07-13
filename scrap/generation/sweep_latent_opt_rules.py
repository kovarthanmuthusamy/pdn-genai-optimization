"""Hard rules aligning multifreq sweep QC with ``pipelines/latent/optimize.py``.

The sweep is a *quality check* of the trained VAE — not a random layout sampler.
These rules mirror the production latent-optimization → heatmap path:

1. **Occupancy / impedance** come from a fixed layout (val set or latent-run export),
   never from marginal z sampling (unless ``ALLOW_RANDOM_LAYOUT=True``).
2. **Layout encode** uses ``PI_REF_MHZ``; **heatmap decode** changes only ``PI_freq``
   per sweep MHz (matches ``model.inference`` layout mode and post-opt ``decode``).
3. **``latent_z`` mode** — load ``best_latent.npy``, decode heatmaps at each MHz without
   re-encoding (matches heatmaps after latent optimization).
4. **``layout_hybrid`` mode** — ``encode_layout_latent_full(occ, imp)`` for private dims,
   then ``z[:, :shared] = z_opt[:, :shared]`` (recommended post-opt variant).
5. **Top-K occupancy** for PEB uses hard top-K (same as optimize ``_topk_occ``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_sweep_val_loader_cache: dict[tuple, DataLoader] = {}

# Inference modes that satisfy QC (fixed layout, no marginal draw)
QC_INFERENCE_MODES = frozenset({"layout_qc", "latent_z", "layout_hybrid"})

# Legacy modes that may draw random occ/imp when not explicitly provided
RANDOM_LAYOUT_MODES = frozenset({"marginal", "layout", "anchor_blend", "encode"})

LATENT_RUN_FILES = {
    "z": "best_latent.npy",
    "occ_topk": "best_occupancy_topk.npy",
    "occ_prob": "best_occupancy_prob.npy",
    "imp_log": "best_impedance_log.npy",
}


@dataclass
class SweepQCConfig:
    qc_sweep: bool = True
    inference_mode: str = "layout_qc"
    layout_source: str = "val"  # val | latent_run | npy
    k_value: int = 30
    num_samples: int = 2
    pi_ref_mhz: float = 200.0
    seed: int = 42
    latent_run_dir: str | None = None
    explicit_z_npy: str | None = None
    explicit_occ_npy: str | None = None
    explicit_imp_npy: str | None = None
    allow_random_layout: bool = False


class SweepQCError(ValueError):
    """Raised when sweep config violates latent-opt alignment rules."""


def validate_sweep_qc(cfg: SweepQCConfig) -> None:
    """Enforce QC rules before generate."""
    mode = cfg.inference_mode.strip().lower()

    if not cfg.qc_sweep:
        if mode in QC_INFERENCE_MODES and cfg.layout_source == "marginal":
            raise SweepQCError(
                f"QC_SWEEP=False but inference_mode={mode!r} requires a fixed layout source."
            )
        return

    if mode in RANDOM_LAYOUT_MODES and not cfg.allow_random_layout:
        raise SweepQCError(
            f"QC_SWEEP=True blocks inference_mode={mode!r} — draws random layouts from "
            f"marginal sampling. Use layout_qc | latent_z | layout_hybrid, or set "
            f"ALLOW_RANDOM_LAYOUT=True to override."
        )

    if mode not in QC_INFERENCE_MODES:
        raise SweepQCError(
            f"Unknown or disallowed inference_mode={mode!r}. "
            f"QC modes: {sorted(QC_INFERENCE_MODES)}"
        )

    if mode == "layout_qc":
        if cfg.layout_source not in ("val", "npy"):
            raise SweepQCError(
                f"layout_qc requires LAYOUT_SOURCE='val' or 'npy', got {cfg.layout_source!r}"
            )
    elif mode in ("latent_z", "layout_hybrid"):
        if cfg.layout_source not in ("latent_run", "npy"):
            raise SweepQCError(
                f"{mode} requires LAYOUT_SOURCE='latent_run' or 'npy', got {cfg.layout_source!r}"
            )
        if cfg.layout_source == "latent_run" and not cfg.latent_run_dir:
            raise SweepQCError(
                f"{mode} with LAYOUT_SOURCE='latent_run' requires LATENT_RUN_DIR "
                f"(e.g. data/latent_runs/exp047/0/K30)"
            )


def topk_occ_binary(occ_prob: torch.Tensor, k: int) -> torch.Tensor:
    """Hard top-K mask — matches ``optimize._topk_occ``."""
    out = torch.zeros_like(occ_prob)
    if k > 0:
        out.scatter_(-1, occ_prob.topk(min(k, occ_prob.shape[-1]), dim=-1).indices, 1.0)
    return out


def _imp_log_to_norm(imp_log: torch.Tensor, imp_log_mean: float, imp_log_std: float) -> torch.Tensor:
    """Convert log-Ω impedance to model normalized space (B, 1, 231)."""
    if imp_log.dim() == 1:
        imp_log = imp_log.unsqueeze(0)
    if imp_log.dim() == 2:
        imp_log = imp_log.unsqueeze(1)
    return (imp_log - imp_log_mean) / max(imp_log_std, 1e-12)


def latent_run_k_dir(run_dir: Path, k: int) -> Path:
    """``.../K30`` folder inside a latent optimization run."""
    run_dir = Path(run_dir)
    tagged = run_dir / f"K{int(k):02d}"
    if tagged.is_dir():
        return tagged
    plain = run_dir / f"K{int(k)}"
    if plain.is_dir():
        return plain
    return tagged


def load_latent_run_bundle(
    run_dir: Path | str,
    k: int,
    device: torch.device,
    *,
    imp_log_mean: float,
    imp_log_std: float,
) -> dict[str, torch.Tensor]:
    """Load optimize.py exports for one K."""
    k_dir = latent_run_k_dir(Path(run_dir), k)
    if not k_dir.is_dir():
        raise FileNotFoundError(f"Latent run K folder not found: {k_dir}")

    z_path = k_dir / LATENT_RUN_FILES["z"]
    if not z_path.is_file():
        raise FileNotFoundError(f"Missing {z_path}")

    z = torch.tensor(np.load(z_path), dtype=torch.float32, device=device)
    if z.dim() == 1:
        z = z.unsqueeze(0)

    occ_topk_np = np.load(k_dir / LATENT_RUN_FILES["occ_topk"])
    occ_topk = torch.tensor(occ_topk_np, dtype=torch.float32, device=device)
    if occ_topk.dim() == 1:
        occ_topk = occ_topk.unsqueeze(0)

    occ_prob_path = k_dir / LATENT_RUN_FILES["occ_prob"]
    if occ_prob_path.is_file():
        occ_prob = torch.tensor(np.load(occ_prob_path), dtype=torch.float32, device=device)
        if occ_prob.dim() == 1:
            occ_prob = occ_prob.unsqueeze(0)
    else:
        occ_prob = occ_topk.clone()

    imp_log_path = k_dir / LATENT_RUN_FILES["imp_log"]
    if imp_log_path.is_file():
        imp_log = torch.tensor(np.load(imp_log_path), dtype=torch.float32, device=device)
        imp_norm = _imp_log_to_norm(imp_log, imp_log_mean, imp_log_std)
    else:
        imp_norm = None

    k_t = torch.full((z.shape[0],), int(k), dtype=torch.long, device=device)
    return {
        "z": z,
        "occ_prob": occ_prob,
        "occ_topk": occ_topk,
        "imp_norm": imp_norm,
        "K": k_t,
        "k_dir": k_dir,
    }


def load_explicit_npy_bundle(
    *,
    z_npy: str | None,
    occ_npy: str | None,
    imp_npy: str | None,
    k: int,
    device: torch.device,
    imp_log_mean: float,
    imp_log_std: float,
) -> dict[str, torch.Tensor]:
    """Load explicit .npy paths (latent-run style layout)."""
    if z_npy is None:
        raise SweepQCError("EXPLICIT_Z_NPY required for npy layout source in latent_z/layout_hybrid")
    z = torch.tensor(np.load(z_npy), dtype=torch.float32, device=device)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    n = z.shape[0]

    if occ_npy is None:
        raise SweepQCError("EXPLICIT_OCC_NPY required for npy layout source")
    occ = torch.tensor(np.load(occ_npy), dtype=torch.float32, device=device)
    if occ.dim() == 1:
        occ = occ.unsqueeze(0)

    if imp_npy is not None:
        imp_raw = torch.tensor(np.load(imp_npy), dtype=torch.float32, device=device)
        if imp_raw.dim() == 1:
            imp_raw = imp_raw.unsqueeze(0)
        if imp_raw.shape[-1] == 231 and imp_raw.abs().max() > 20:
            imp_norm = _imp_log_to_norm(imp_raw, imp_log_mean, imp_log_std)
        else:
            imp_norm = imp_raw.unsqueeze(1) if imp_raw.dim() == 2 else imp_raw
    else:
        imp_norm = None

    k_t = torch.full((n,), int(k), dtype=torch.long, device=device)
    return {
        "z": z,
        "occ_prob": occ.float(),
        "occ_topk": topk_occ_binary(occ.float(), k) if occ.max() <= 1.0 else occ.float(),
        "imp_norm": imp_norm,
        "K": k_t,
    }


def _sweep_val_loader_key(data_dir: Path | str, experiment_cfg: dict[str, Any]) -> tuple:
    return (
        str(Path(data_dir).resolve()),
        int(experiment_cfg.get("batch_size", 64)),
        float(experiment_cfg.get("train_split", 0.9)),
        bool(experiment_cfg.get("split_by_design", True)),
        bool(experiment_cfg.get("stratify_by_k", True)),
        bool(experiment_cfg.get("cache_in_ram", True)),
    )


def k_values_in_val_loader(val_ld: DataLoader) -> set[int]:
    """Return K values present in the val split (one pass over ``val_ld``)."""
    found: set[int] = set()
    for batch in val_ld:
        found.update(int(k) for k in batch["K"].numpy().tolist())
    return found


def filter_k_for_layout_qc(
    k_values: list[int],
    val_ld: DataLoader,
) -> list[int]:
    """Drop K not in val; raise if none remain."""
    available = k_values_in_val_loader(val_ld)
    kept = [k for k in k_values if k in available]
    dropped = [k for k in k_values if k not in available]
    if dropped:
        preview = ", ".join(str(k) for k in sorted(available)[:15])
        suffix = "..." if len(available) > 15 else ""
        print(
            f"  layout_qc: dropping K not in val split: {dropped} "
            f"(available: {preview}{suffix})"
        )
    if not kept:
        raise SweepQCError(
            f"No requested K values found in val split (requested {k_values}). "
            f"Available K: {sorted(available)}. Check DATA_DIR and K_VALUE."
        )
    return kept


def get_sweep_val_loader(
    *,
    data_dir: Path | str,
    experiment_cfg: dict[str, Any],
    device: torch.device,
) -> DataLoader:
    """Shared val ``DataLoader`` for sweep/QC — built once per dataset + split config."""
    from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders

    key = _sweep_val_loader_key(data_dir, experiment_cfg)
    cached = _sweep_val_loader_cache.get(key)
    if cached is not None:
        return cached

    _, val_ld = create_multifreq_data_loaders(
        data_dir=str(data_dir),
        batch_size=int(experiment_cfg.get("batch_size", 64)),
        num_workers=2,
        train_split=float(experiment_cfg.get("train_split", 0.9)),
        seed=42,
        pin_memory=device.type == "cuda",
        split_by_design=experiment_cfg.get("split_by_design", True),
        stratify_by_k=experiment_cfg.get("stratify_by_k", True),
        balance_k=False,
        balance_freq=False,
        cache_in_ram=experiment_cfg.get("cache_in_ram", True),
    )
    _sweep_val_loader_cache[key] = val_ld
    return val_ld


def load_val_layout_samples(
    *,
    data_dir: Path | str,
    experiment_cfg: dict[str, Any],
    k_value: int,
    num_samples: int,
    seed: int,
    device: torch.device,
    with_ground_truth: bool = False,
    val_ld: DataLoader | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray
]:
    """Collect ``num_samples`` real val layouts with fixed K (matches eval_real_data_sweep).

    If ``with_ground_truth``, also returns ``heatmap_norm`` (B,1,H,W) and native ``pi_mhz`` (B,).
    """
    if val_ld is None:
        val_ld = get_sweep_val_loader(
            data_dir=data_dir,
            experiment_cfg=experiment_cfg,
            device=device,
        )

    occ_chunks: list[torch.Tensor] = []
    imp_chunks: list[torch.Tensor] = []
    k_chunks: list[torch.Tensor] = []
    rng = np.random.default_rng(seed)

    for batch in val_ld:
        k_mask = batch["K"].numpy() == int(k_value)
        if not k_mask.any():
            continue
        idx = np.where(k_mask)[0]
        rng.shuffle(idx)
        for i in idx:
            occ_chunks.append(batch["occupancy"][i])
            imp_chunks.append(batch["impedance"][i])
            k_chunks.append(batch["K"][i])
            if len(occ_chunks) >= num_samples:
                break
        if len(occ_chunks) >= num_samples:
            break

    if not occ_chunks:
        raise SweepQCError(
            f"No val layouts found for K={k_value}. Check DATA_DIR and K_VALUE."
        )

    occ = torch.stack(occ_chunks[:num_samples]).to(device).float()
    imp = torch.stack(imp_chunks[:num_samples]).to(device).float()
    if imp.dim() == 2:
        imp = imp.unsqueeze(1)
    elif imp.dim() == 3:
        imp = imp[:, :1]
    k_t = torch.stack(k_chunks[:num_samples]).to(device).long()
    return occ, imp, k_t


def load_val_anchor_samples(
    *,
    data_dir: Path | str,
    experiment_cfg: dict[str, Any],
    k_value: int,
    anchor_mhz: float,
    num_samples: int,
    seed: int,
    device: torch.device,
    mhz_tol: float = 2.0,
    val_ld: DataLoader | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Val layouts with real heatmaps at a training-anchor PI_freq."""
    from src_vae.others.pi_freq_utils import _LOG10_MIN, _LOG10_RANGE

    def _norm_to_mhz(norm: float) -> float:
        log10_hz = float(norm) * _LOG10_RANGE + _LOG10_MIN
        return float(10.0 ** log10_hz / 1e6)

    if val_ld is None:
        val_ld = get_sweep_val_loader(
            data_dir=data_dir,
            experiment_cfg=experiment_cfg,
            device=device,
        )

    occ_chunks: list[torch.Tensor] = []
    imp_chunks: list[torch.Tensor] = []
    hm_chunks: list[torch.Tensor] = []
    k_chunks: list[torch.Tensor] = []
    rng = np.random.default_rng(seed + int(round(anchor_mhz)))

    for batch in val_ld:
        pi_mhz = np.array([_norm_to_mhz(float(x)) for x in batch["PI_freq"].numpy()])
        k_arr = batch["K"].numpy()
        sel = (k_arr == int(k_value)) & (np.abs(pi_mhz - float(anchor_mhz)) < mhz_tol)
        if not sel.any():
            continue
        idx = np.where(sel)[0]
        rng.shuffle(idx)
        for i in idx:
            occ_chunks.append(batch["occupancy"][i])
            imp_chunks.append(batch["impedance"][i])
            hm_chunks.append(batch["heatmap_norm"][i])
            k_chunks.append(batch["K"][i])
            if len(occ_chunks) >= num_samples:
                break
        if len(occ_chunks) >= num_samples:
            break

    if not occ_chunks:
        raise ValueError(f"No val samples at anchor {anchor_mhz} MHz, K={k_value}")

    occ = torch.stack(occ_chunks[:num_samples]).to(device).float()
    imp = torch.stack(imp_chunks[:num_samples]).to(device).float()
    hm = torch.stack(hm_chunks[:num_samples]).to(device).float()
    if imp.dim() == 2:
        imp = imp.unsqueeze(1)
    elif imp.dim() == 3:
        imp = imp[:, :1]
    k_t = torch.stack(k_chunks[:num_samples]).to(device).long()
    return occ, imp, hm, k_t


@torch.inference_mode()
def build_hybrid_z(
    model: torch.nn.Module,
    z_opt: torch.Tensor,
    occ_prob: torch.Tensor,
    imp_norm: torch.Tensor,
    k: torch.Tensor,
    pi_ref_mhz: float,
    device: torch.device,
) -> torch.Tensor:
    """Private dims from layout head; shared dims from optimized z."""
    from src_vae.others.pi_freq_utils import pi_freq_norm_for_model

    n = z_opt.shape[0]
    pi_ref = pi_freq_norm_for_model(pi_ref_mhz, n, unit="mhz", device=device)
    z_hm, _, _ = model.encode_layout_latent_full(occ_prob, imp_norm, k, pi_ref)
    shared = int(getattr(model, "shared_latent_dim", z_opt.shape[1]))
    z_hm[:, :shared] = z_opt[:, :shared]
    return z_hm


@torch.inference_mode()
def decode_heatmap_sweep_mhz(
    model: torch.nn.Module,
    z: torch.Tensor,
    k: torch.Tensor,
    mhz_list: list[float],
    device: torch.device,
    *,
    occ_for_decode: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Decode heatmaps at each MHz — z fixed (latent-opt path). PI_freq only changes."""
    from src_vae.others.pi_freq_utils import pi_freq_norm_for_model

    out: list[torch.Tensor] = []
    for mhz in mhz_list:
        pi_t = pi_freq_norm_for_model(mhz, z.shape[0], unit="mhz", device=device)
        hm, _, _ = model.decode(z, k, pi_t, occupancy=occ_for_decode)
        out.append(hm)
    return out


def qc_manifest_extra(cfg: SweepQCConfig) -> dict[str, Any]:
    """Extra fields for sweep_freq_manifest.json."""
    return {
        "qc_sweep": cfg.qc_sweep,
        "inference_mode": cfg.inference_mode,
        "layout_source": cfg.layout_source,
        "pi_ref_mhz": cfg.pi_ref_mhz,
        "latent_run_dir": cfg.latent_run_dir,
    }
