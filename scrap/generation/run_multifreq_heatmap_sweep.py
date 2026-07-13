"""Multifreq Heatmap Sweep at Fixed K (or explicit K list).

Purpose: Decode VAE samples at multiple PI frequencies for one or more K values for
    PI-Distribution heatmaps; write per-freq folders, combined .peb, sweep manifest,
    and summary plots.
Run: python scrap/generation/run_multifreq_heatmap_sweep.py
Inputs / outputs: CHECKPOINT_PATH, DATA_DIR, SWEEP, K_VALUE (int or list[int])
    → {OUTPUT_ROOT}/freq_*MHz/K{k}/; PEB_OUT_FILE (.peb), sweep_freq_manifest.json;
    optional PEB_COPY_DEST.
Dependencies: torch, matplotlib, numpy; dynamic VAEInference from EXPERIMENT_DIR.
Agent notes:
    - Type: CLI generator (does not replace run_all_k K-sweep workflow)
    - Key symbols: main, run_generate, run_val, exported_freq_mhz_list,
    - Config keys: ``MODE``, ``EXPERIMENT_DIR``, ``CHECKPOINT_PATH``, ``DATA_DIR``, ``OUTPUT_ROOT``, ``DENSE_N_POINTS``, ``K_VALUE``, ``NUM_SAMPLES``, ``SHARED_TEMP``, ``SEED``, ``HEATMAP_ONLY_PEB``, ``PEB_OUT_FILE``, ``POWERBUS``, ``FORCE_CPU``, ``VAL_MAX_BATCHES``, ``BACKGROUND_MARGIN``, ``INFERENCE_MODE``, ``PI_REF_MHZ``, ``CALIBRATE_FG_MAX``, ``INFERENCE_FACTORIZED_ONLY``
      write_sweep_freq_manifest, HEATMAP_ONLY_PEB, K_VALUE, OUTPUT_ROOT
"""
from __future__ import annotations

import csv
import dataclasses
import importlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path
setup_path()

from experiments.exp038_true_multi.codes.dataloader_multifreq import (  # noqa: E402
    ANCHOR_MHZ,
    create_multifreq_data_loaders,
)
from scrap.generation.generate_peb import generate_peb  # noqa: E402
from scrap.peb_copy import copy_peb_to_folder  # noqa: E402
from scrap.generation.sweep_latent_opt_rules import (  # noqa: E402
    SweepQCConfig,
    SweepQCError,
    build_hybrid_z,
    decode_heatmap_sweep_mhz,
    load_explicit_npy_bundle,
    load_latent_run_bundle,
    filter_k_for_layout_qc,
    get_sweep_val_loader,
    load_val_layout_samples,
    qc_manifest_extra,
    topk_occ_binary,
    validate_sweep_qc,
)
from scrap.generation.sweep_qc_eval import run_sweep_qc_eval  # noqa: E402
from experiments.exp038_true_multi.codes.freq_inference_utils import (  # noqa: E402
    calibrate_heatmap_physical,
    interp_anchor_value,
    is_training_anchor,
    load_anchor_fg_max_table,
)
from src_vae.others.pi_freq_utils import (  # noqa: E402
    _LOG10_MIN,
    _LOG10_RANGE,
    pi_freq_mhz_to_norm,
    pi_freq_norm_for_model,
)

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

# MODE = "generate"  # generate | val | both
MODE = "generate"

EXPERIMENT_DIR = "experiments/exp050"
CHECKPOINT_PATH = f"{EXPERIMENT_DIR}/checkpoints/last_model.pt"
DATA_DIR = "data_multi_norm_unbounded"
OUTPUT_ROOT = f"{EXPERIMENT_DIR}/multifreq_heatmap_sweep_30"

# SWEEP = "anchors"  # anchors | dense | explicit MHz list
# SWEEP = "dense"
SWEEP: list[float] | str = [10, 70, 120, 270, 400]
DENSE_N_POINTS = 24

K_VALUE: int | list[int] = 30  # single int or explicit list, e.g. [10, 20, 30]
NUM_SAMPLES = 2
SHARED_TEMP = 1.5
SEED = 42

# Run mode — single PI kind per sample, mutually exclusive:
#   "heatmap"   → PEB emits PI-Distribution groups; SWEEP MHz list drives per-freq folders
#   "impedance" → PEB emits PI-Spectrum groups; SWEEP ignored (decode@PI_REF_MHZ, flat K{k}/)
RUN_MODE = "heatmap"
HEATMAP_ONLY_PEB = True  # 1 PI per sample (both run modes emit a single PI kind)
PEB_OUT_FILE = f"{OUTPUT_ROOT}/pi_distribution_K30_freq_sweep.peb"
POWERBUS = "Power_GND"
COMPONENTS = "IC1_Port1"  # CreatePISpectrum IC port(s) used when RUN_MODE == "impedance"
PEB_COPY_DEST: str | None = r"C:\Users\muthusamy\Desktop\design\PEB"
REPORT_COPY_DEST: str | None = r"C:\Users\muthusamy\Desktop\reports"

FORCE_CPU = False
VAL_MAX_BATCHES = 0  # 0 = full val loader
BACKGROUND_MARGIN = 0.5

# QC sweep — aligned with pipelines/latent/optimize.py (no random marginal layouts)
QC_SWEEP = True
# layout_qc: real val (occ, imp) → encode_layout@PI_REF → decode@each MHz  [model QC]
# latent_z:   best_latent.npy → decode@each MHz only                        [post-opt path]
# layout_hybrid: encode_layout_full + shared z from optimize                [post-opt hybrid]
INFERENCE_MODE = "layout_qc"
LAYOUT_SOURCE = "val"  # val | latent_run | npy
LATENT_RUN_DIR: str | None = None  # e.g. data/latent_runs/exp050/0
EXPLICIT_Z_NPY: str | None = None
EXPLICIT_OCC_NPY: str | None = None
EXPLICIT_IMP_NPY: str | None = None
ALLOW_RANDOM_LAYOUT = False  # must be True to use marginal / anchor_blend / layout w/o val

RUN_SWEEP_QC_EVAL = True  # write sweep_qc_report.md + print agent copy block

PI_REF_MHZ = 200.0
CALIBRATE_FG_MAX = False
INFERENCE_FACTORIZED_ONLY = False

# =============================================================================


def normalize_k_values(k: int | list[int] | tuple[int, ...]) -> list[int]:
    """Return sorted unique K values in [0, 52] from a scalar or list."""
    if isinstance(k, int):
        vals = [k]
    else:
        vals = [int(x) for x in k]
    ks = sorted(set(vals))
    if not ks:
        raise ValueError("K_VALUE must be a non-empty int or list of ints")
    if not all(0 <= x <= 52 for x in ks):
        raise ValueError(f"Each K must be in [0, 52], got {ks}")
    return ks


def k_output_tag(k: int | list[int] | tuple[int, ...]) -> str:
    """Folder tag suffix, e.g. 30 or 10_20_30."""
    return "_".join(str(x) for x in normalize_k_values(k))


def _peb_kind() -> str:
    """PEB analysis kind for the active run mode."""
    return "spectrum" if is_impedance_run_mode() else "distribution"


def is_impedance_run_mode() -> bool:
    return str(RUN_MODE).strip().lower() == "impedance"


def uses_flat_k_layout(out_root: str | Path | None = None) -> bool:
    """True when outputs live under ``K{k}/`` (impedance) rather than ``freq_*MHz/K{k}/``."""
    if is_impedance_run_mode():
        return True
    manifest = sweep_freq_manifest_path(out_root)
    if manifest.is_file():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return str(raw.get("run_mode", "")).strip().lower() == "impedance"
    return False


def peb_basename_for_k(k: int | list[int] | tuple[int, ...]) -> str:
    """Combined multifreq sweep PEB filename for one or more K values.

    ``pi_distribution_K..._freq_sweep.peb`` in heatmap mode,
    ``pi_spectrum_K..._freq_sweep.peb`` in impedance mode.
    """
    return f"pi_{_peb_kind()}_K{k_output_tag(k)}_freq_sweep.peb"


def _k_values_from_generate_summary(root: Path) -> list[int] | None:
    """K values actually generated (authoritative when layout_qc drops K)."""
    csv_path = root / "generate_summary.csv"
    if not csv_path.is_file():
        return None
    found: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("k"):
                found.add(int(row["k"]))
    return sorted(found) if found else None


def _k_values_from_output_dirs(root: Path) -> list[int] | None:
    """Infer K from ``freq_*MHz/K*/data_sample_*`` folders."""
    found: set[int] = set()
    for sample_dir in root.glob("freq_*MHz/K*/data_sample_*"):
        if not sample_dir.is_dir():
            continue
        m = re.match(r"^K(\d+)$", sample_dir.parent.name)
        if m:
            found.add(int(m.group(1)))
    return sorted(found) if found else None


def exported_k_values(out_root: str | Path | None = None) -> list[int]:
    """Effective K list for move/compare — matches what generate + PEB actually contain."""
    root = Path(out_root if out_root is not None else OUTPUT_ROOT)

    from_csv = _k_values_from_generate_summary(root)
    if from_csv:
        manifest = sweep_freq_manifest_path(root)
        if manifest.is_file():
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "k_values" in raw:
                manifest_ks = normalize_k_values(raw["k_values"])
                dropped = [k for k in manifest_ks if k not in from_csv]
                if dropped:
                    print(
                        f"  effective K from generate_summary.csv: {from_csv[0]}–{from_csv[-1]} "
                        f"({len(from_csv)} values; manifest omitted {dropped})"
                    )
        return from_csv

    from_dirs = _k_values_from_output_dirs(root)
    if from_dirs:
        return from_dirs

    manifest = sweep_freq_manifest_path(root)
    if manifest.is_file():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if "k_values" in raw:
                return normalize_k_values(raw["k_values"])
            if "k" in raw:
                return normalize_k_values(int(raw["k"]))
    return normalize_k_values(K_VALUE)


def _sync_derived_paths() -> None:
    """Keep PEB_OUT_FILE aligned with OUTPUT_ROOT and K_VALUE."""
    global PEB_OUT_FILE
    PEB_OUT_FILE = f"{OUTPUT_ROOT}/{peb_basename_for_k(K_VALUE)}"


SWEEP_FREQ_MANIFEST = "sweep_frequencies_mhz.json"


def sweep_freq_manifest_path(out_root: str | Path | None = None) -> Path:
    return Path(out_root if out_root is not None else OUTPUT_ROOT) / SWEEP_FREQ_MANIFEST


def write_sweep_freq_manifest(
    mhz_list: list[float],
    out_root: str | Path | None = None,
    *,
    k_values: list[int] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist MHz + effective K list from the last generate run (move/compare read this)."""
    path = sweep_freq_manifest_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    requested = normalize_k_values(K_VALUE)
    effective = normalize_k_values(k_values if k_values is not None else K_VALUE)
    dropped = [k for k in requested if k not in effective]
    payload: dict[str, Any] = {
        "mhz": [float(m) for m in mhz_list],
        "run_mode": str(RUN_MODE).strip().lower(),
        "k": effective[0] if len(effective) == 1 else effective,
        "k_values": effective,
        "k_values_requested": requested,
        "k_values_dropped": dropped,
        "num_samples": int(NUM_SAMPLES),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_sweep_freq_mhz_list(out_root: str | Path | None = None) -> list[float] | None:
    """MHz from manifest or generate_summary.csv; None if neither exists."""
    root = Path(out_root if out_root is not None else OUTPUT_ROOT)
    manifest = sweep_freq_manifest_path(root)
    if manifest.is_file():
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [float(x) for x in raw]
        if str(raw.get("run_mode", "")).strip().lower() == "impedance":
            return []
        return [float(x) for x in raw.get("mhz", [])]

    csv_path = root / "generate_summary.csv"
    if csv_path.is_file():
        mhz_vals: list[float] = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "mhz" in row and row["mhz"]:
                    mhz_vals.append(float(row["mhz"]))
        if mhz_vals:
            return mhz_vals
    return None


def exported_freq_mhz_list(
    sweep: str | list | tuple | None = None,
    out_root: str | Path | None = None,
) -> list[int]:
    """Integer MHz list for move/compare (matches folder tags under OUTPUT_ROOT).

    Empty for impedance runs — PI-Spectrum is broadband; layout is flat ``K{k}/``.
    """
    recorded = load_sweep_freq_mhz_list(out_root)
    if recorded is not None:
        return [int(round(m)) for m in recorded]
    if is_impedance_run_mode() or uses_flat_k_layout(out_root):
        return []
    return [int(round(m)) for m in _build_mhz_list(sweep if sweep is not None else SWEEP)]


def _norm_to_mhz(norm: float) -> float:
    log10_hz = float(norm) * _LOG10_RANGE + _LOG10_MIN
    return float(10.0 ** log10_hz / 1e6)


def _load_experiment_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    lines = [
        ln for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return json.loads("\n".join(lines))


def _load_vae_inference(experiment_dir: str | None = None):
    """Import VAEInference from the experiment package."""
    exp = experiment_dir or EXPERIMENT_DIR
    exp_mod = exp.replace("/", ".").replace("\\", ".") + ".codes.inference_vae"
    try:
        return importlib.import_module(exp_mod).VAEInference
    except ModuleNotFoundError:
        fallback = "experiments.exp038_true_multi.codes.inference_vae"
        print(f"WARNING: {exp_mod} not found — using {fallback}")
        return importlib.import_module(fallback).VAEInference


def _resolve_vae_inference():
    """Resolve VAEInference for current EXPERIMENT_DIR (re-read after config overrides)."""
    return _load_vae_inference(EXPERIMENT_DIR)


def _build_mhz_list(sweep: str | list[int] | tuple[float, ...]) -> list[float]:
    if isinstance(sweep, (list, tuple)):
        return [float(f) for f in sweep]
    if sweep == "anchors":
        return list(ANCHOR_MHZ)
    if sweep == "dense":
        return [float(x) for x in np.geomspace(1.0, 600.0, DENSE_N_POINTS)]
    raise ValueError(f"Unknown SWEEP={sweep!r}; use 'anchors', 'dense', or a list of MHz")


def _freq_tag(mhz: float) -> str:
    return f"freq_{int(round(mhz))}MHz"


def _hm_physical(hm_z: torch.Tensor, engine: Any, mhz: float | None = None) -> torch.Tensor:
    """Denorm heatmap tensor to Ω (robust per-MHz, global-max, or legacy z-score)."""
    if hasattr(engine, "denorm_heatmap_physical"):
        return engine.denorm_heatmap_physical(hm_z, mhz=mhz)
    from src_vae.others.heatmap_z_clip import heatmap_norm_to_physical, heatmap_z_to_physical

    hm_stats = getattr(engine, "hm_stats", None)
    if hm_stats is None:
        hm_stats = getattr(getattr(engine, "norm_stats", None), "heatmap", None)
    clip = getattr(engine, "hm_z_clip", None)
    lo, hi = (clip[0], clip[1]) if clip is not None else (None, None)
    if hm_stats is not None and getattr(hm_stats, "is_global_max", lambda: False)():
        raw = hm_stats if isinstance(hm_stats, dict) else None
        if raw is None and hasattr(hm_stats, "global_max_ohm"):
            raw = {"norm_mode": "global_max", "global_max_ohm": hm_stats.global_max_ohm}
        if raw is not None:
            return heatmap_norm_to_physical(hm_z, raw, clip_lo=lo, clip_hi=hi)
    return heatmap_z_to_physical(
        hm_z,
        engine.hm_log_mean,
        engine.hm_log_std,
        hm_stats=hm_stats,
        mhz=mhz,
        clip_lo=lo,
        clip_hi=hi,
    )


def _impedance_log_from_norm(engine: Any, imp_norm: torch.Tensor) -> torch.Tensor:
    """Blended log-Ω impedance ``(B, N)`` from the model's normalized output.

    Impedance is z-scored in log space (see ``sweep_latent_opt_rules._imp_log_to_norm``):
    ``norm = (imp_log - imp_log_mean) / imp_log_std``. Prefer the engine's own
    ``_denorm_impedance`` when present (matches generate_samples_and_peb.py).
    """
    if hasattr(engine, "_denorm_impedance"):
        out = engine._denorm_impedance(imp_norm)
        imp_log = out[0] if isinstance(out, (tuple, list)) else out
    else:
        std = float(getattr(engine, "imp_log_std", 1.0))
        mean = float(getattr(engine, "imp_log_mean", 0.0))
        x = imp_norm
        if x.dim() == 3:  # (B, C, N) → primary channel
            x = x[:, 0]
        imp_log = x * std + mean
    return imp_log.reshape(imp_log.shape[0], -1)


def _fg_threshold(engine: Any, cfg: dict[str, Any] | None = None) -> float:
    """Foreground threshold in normalized heatmap space."""
    if hasattr(engine, "fg_threshold"):
        return float(engine.fg_threshold())
    cfg = cfg or {}
    if cfg.get("heatmap_fg_threshold") is not None:
        return float(cfg["heatmap_fg_threshold"])
    return float(getattr(engine, "background_value", -2.9669)) + BACKGROUND_MARGIN


def _fg_mse(recon: torch.Tensor, target: torch.Tensor, fg_thr: float) -> torch.Tensor:
    """Foreground MSE per sample (B,) in normalized heatmap space."""
    fg = (target > fg_thr).float()
    err = (recon - target).pow(2)
    n = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (err * fg).sum(dim=(1, 2, 3)) / n


def _hm_stats(hm_phys: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    fg = hm_phys[0][mask] if hm_phys.ndim == 3 else hm_phys[mask]
    if fg.size == 0:
        return {"max": 0.0, "mean_fg": 0.0, "p95": 0.0}
    return {
        "max": float(fg.max()),
        "mean_fg": float(fg.mean()),
        "p95": float(np.percentile(fg, 95)),
    }


def _override_inference_data_dir(inf_mod: Any) -> Path:
    """Use pipeline DATA_DIR for norm stats (resolved under REPO_ROOT)."""
    from repo_paths import resolve_repo_path

    data_root = resolve_repo_path(DATA_DIR)
    stats_file = data_root / "normalization_stats.json"
    if not stats_file.is_file():
        raise SystemExit(
            "normalization_stats.json not found under pipeline DATA_DIR.\n"
            f"  Expected: {stats_file}\n"
            f"  Set DATA_DIR in run_multifreq_heatmap_sweep.py or the pipeline config."
        )
    inf_mod._default_data_dir = lambda: data_root
    return data_root


def _load_engine(device: torch.device) -> Any:
    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    VAEInference = _resolve_vae_inference()
    inf_mod = importlib.import_module(VAEInference.__module__)
    data_root = _override_inference_data_dir(inf_mod)
    engine = VAEInference(
        checkpoint_path=str(ckpt),
        device=device,
    )
    if hasattr(engine.model, "inference_factorized_only"):
        engine.model.inference_factorized_only = bool(INFERENCE_FACTORIZED_ONLY)
    stats_auto = PROJECT_ROOT / EXPERIMENT_DIR / "metrics" / "latent_stats.json"
    engine.load_latent_stats(str(stats_auto) if stats_auto.is_file() else None)
    if hasattr(engine, "hm_stats"):
        norm_desc = engine.hm_stats.describe()
    elif getattr(engine, "use_global_max", False):
        norm_desc = f"global-max gmax={engine.global_max_ohm:.4g}Ω"
    else:
        norm_desc = f"z-score μ={engine.hm_log_mean:.3g} σ={engine.hm_log_std:.3g}"
    print(f"  model loaded  ckpt={ckpt.name}  norm={norm_desc}")
    return engine


def _copy_peb_if_configured(peb_file: Path) -> None:
    if not PEB_COPY_DEST:
        return
    try:
        target = copy_peb_to_folder(peb_file, PEB_COPY_DEST)
        print(f"  Copied PEB    : {target}")
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"  ⚠ PEB copy skipped: {exc}")


def _build_qc_config() -> SweepQCConfig:
    return SweepQCConfig(
        qc_sweep=bool(QC_SWEEP),
        inference_mode=str(INFERENCE_MODE),
        layout_source=str(LAYOUT_SOURCE),
        k_value=normalize_k_values(K_VALUE)[0],
        num_samples=int(NUM_SAMPLES),
        pi_ref_mhz=float(PI_REF_MHZ),
        seed=int(SEED),
        latent_run_dir=LATENT_RUN_DIR,
        explicit_z_npy=EXPLICIT_Z_NPY,
        explicit_occ_npy=EXPLICIT_OCC_NPY,
        explicit_imp_npy=EXPLICIT_IMP_NPY,
        allow_random_layout=bool(ALLOW_RANDOM_LAYOUT),
    )


def _legacy_marginal_layout_draw(
    engine: Any,
    device: torch.device,
    num_samples: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random occ/imp from marginal z — only when ALLOW_RANDOM_LAYOUT=True."""
    with torch.no_grad():
        _, occ_prob0, imp0 = engine.model.inference(
            num_samples,
            device,
            K=k,
            PI_freq=PI_REF_MHZ,
            pi_freq_unit="mhz",
            latent_stats=engine.latent_stats,
            per_K_latent_stats=engine.per_K_latent_stats,
            shared_temp=SHARED_TEMP,
            mode="marginal",
        )
    return occ_prob0, imp0


@dataclass
class _KGenerateContext:
    n_batch: int
    shared_occ: torch.Tensor | None = None
    shared_imp: torch.Tensor | None = None
    shared_k: torch.Tensor | None = None
    z_fixed: torch.Tensor | None = None


def _load_k_generate_context(
    engine: Any,
    device: torch.device,
    qc_cfg: SweepQCConfig,
    k: int,
    *,
    exp_cfg: dict[str, Any] | None,
    sweep_val_ld: Any,
    quiet: bool = False,
) -> _KGenerateContext:
    """Load fixed layout / latent bundle for one K (cached per run_generate call)."""
    mode = qc_cfg.inference_mode.strip().lower()
    n_batch = int(NUM_SAMPLES)

    if mode == "layout_qc":
        if exp_cfg is None:
            exp_cfg = _load_experiment_config(PROJECT_ROOT / EXPERIMENT_DIR / "config.yaml")
        if sweep_val_ld is None:
            sweep_val_ld = get_sweep_val_loader(
                data_dir=PROJECT_ROOT / DATA_DIR,
                experiment_cfg=exp_cfg,
                device=device,
            )
        shared_occ, shared_imp, shared_k = load_val_layout_samples(
            data_dir=PROJECT_ROOT / DATA_DIR,
            experiment_cfg=exp_cfg,
            k_value=k,
            num_samples=n_batch,
            seed=SEED,
            device=device,
            val_ld=sweep_val_ld,
        )
        n_batch = int(shared_occ.shape[0])
        if not quiet:
            print(f"  layout_qc: {n_batch} val layout(s) at K={k}")
        return _KGenerateContext(
            n_batch=n_batch,
            shared_occ=shared_occ,
            shared_imp=shared_imp,
            shared_k=shared_k,
        )

    if mode in ("latent_z", "layout_hybrid"):
        if qc_cfg.layout_source == "latent_run":
            bundle = load_latent_run_bundle(
                qc_cfg.latent_run_dir or "",
                k,
                device,
                imp_log_mean=float(engine.imp_log_mean),
                imp_log_std=float(engine.imp_log_std),
            )
        else:
            bundle = load_explicit_npy_bundle(
                z_npy=qc_cfg.explicit_z_npy,
                occ_npy=qc_cfg.explicit_occ_npy,
                imp_npy=qc_cfg.explicit_imp_npy,
                k=k,
                device=device,
                imp_log_mean=float(engine.imp_log_mean),
                imp_log_std=float(engine.imp_log_std),
            )
        z_fixed = bundle["z"]
        shared_occ = bundle["occ_prob"]
        shared_imp = bundle["imp_norm"]
        shared_k = bundle["K"]
        n_batch = int(z_fixed.shape[0])
        if not quiet:
            print(f"  {mode}: z {tuple(z_fixed.shape)} from {qc_cfg.layout_source} K={k}")

        if mode == "layout_hybrid":
            if shared_imp is None:
                raise SystemExit(
                    "layout_hybrid requires impedance (best_impedance_log.npy or EXPLICIT_IMP_NPY)"
                )
            z_fixed = build_hybrid_z(
                engine.model,
                z_fixed,
                shared_occ,
                shared_imp,
                shared_k,
                PI_REF_MHZ,
                device,
            )
            if not quiet:
                print("  layout_hybrid: merged optimized shared + layout private head")

        return _KGenerateContext(
            n_batch=n_batch,
            shared_occ=shared_occ,
            shared_imp=shared_imp,
            shared_k=shared_k,
            z_fixed=z_fixed,
        )

    return _KGenerateContext(n_batch=n_batch)


def _decode_at_mhz_k(
    engine: Any,
    device: torch.device,
    qc_cfg: SweepQCConfig,
    ctx: _KGenerateContext,
    *,
    mhz: float,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run model inference for one (MHz, K) pair using a prepared layout context."""
    mode = qc_cfg.inference_mode.strip().lower()
    n_batch = ctx.n_batch

    with torch.no_grad():
        if mode == "layout_qc":
            assert ctx.shared_occ is not None and ctx.shared_imp is not None
            hm_z, occ_prob, imp_norm = engine.model.inference(
                n_batch,
                device,
                K=k,
                PI_freq=mhz,
                pi_freq_unit="mhz",
                latent_stats=engine.latent_stats,
                per_K_latent_stats=engine.per_K_latent_stats,
                shared_temp=SHARED_TEMP,
                mode="layout",
                pi_ref_mhz=PI_REF_MHZ,
                occupancy=ctx.shared_occ,
                impedance=ctx.shared_imp,
            )
        elif mode in ("latent_z", "layout_hybrid"):
            assert ctx.z_fixed is not None and ctx.shared_k is not None
            occ_decode = ctx.shared_occ
            hm_list = decode_heatmap_sweep_mhz(
                engine.model,
                ctx.z_fixed,
                ctx.shared_k,
                [mhz],
                device,
                occ_for_decode=occ_decode,
            )
            hm_z = hm_list[0]
            pi_t = pi_freq_norm_for_model(mhz, n_batch, unit="mhz", device=device)
            _, occ_logits, imp_norm = engine.model.decode(
                ctx.z_fixed, ctx.shared_k, pi_t, occupancy=occ_decode,
            )
            occ_prob = torch.sigmoid(occ_logits)
        elif mode in ("layout", "anchor_blend", "marginal", "encode"):
            if not ALLOW_RANDOM_LAYOUT:
                raise SystemExit(f"Mode {mode!r} blocked — set ALLOW_RANDOM_LAYOUT=True")
            inf_kw: dict[str, Any] = dict(
                latent_stats=engine.latent_stats,
                per_K_latent_stats=engine.per_K_latent_stats,
                shared_temp=SHARED_TEMP,
                mode=mode,
                pi_ref_mhz=PI_REF_MHZ,
            )
            if mode in ("layout", "anchor_blend"):
                occ0, imp0 = _legacy_marginal_layout_draw(engine, device, n_batch, k)
                inf_kw["occupancy"] = occ0
                inf_kw["impedance"] = imp0
            hm_z, occ_prob, imp_norm = engine.model.inference(
                n_batch,
                device,
                K=k,
                PI_freq=mhz,
                pi_freq_unit="mhz",
                **inf_kw,
            )
        else:
            raise SystemExit(f"Unhandled inference mode: {mode!r}")

    return hm_z, occ_prob, imp_norm


def run_generate(engine: Any, mhz_list: list[float], out_root: Path, device: torch.device) -> Path:
    """Decode at each (MHz, K); save heatmaps, occupancy, and combined PEB."""
    k_values = normalize_k_values(K_VALUE)
    qc_cfg = _build_qc_config()
    try:
        for k in k_values:
            validate_sweep_qc(dataclasses.replace(qc_cfg, k_value=k))
    except SweepQCError as exc:
        raise SystemExit(f"Sweep QC config error: {exc}") from exc

    mode = qc_cfg.inference_mode.strip().lower()
    impedance_run = is_impedance_run_mode()
    out_root.mkdir(parents=True, exist_ok=True)
    mask = engine.binary_mask
    rows: list[dict[str, Any]] = []
    all_occupancies: list[np.ndarray] = []
    all_peb_freqs: list[str] = []

    torch.manual_seed(SEED)

    exp_cfg: dict[str, Any] | None = None
    sweep_val_ld = None
    if mode == "layout_qc":
        exp_cfg = _load_experiment_config(PROJECT_ROOT / EXPERIMENT_DIR / "config.yaml")
        sweep_val_ld = get_sweep_val_loader(
            data_dir=PROJECT_ROOT / DATA_DIR,
            experiment_cfg=exp_cfg,
            device=device,
        )
        k_values = filter_k_for_layout_qc(k_values, sweep_val_ld)

    k_label = k_output_tag(k_values)
    compact_log = (not impedance_run) and len(k_values) * len(mhz_list) > 15
    if impedance_run:
        print(
            f"\nGenerate  K={k_label}  PI frequencies ignored (decode@{PI_REF_MHZ:g} MHz only)  "
            f"samples={NUM_SAMPLES}  mode={mode}/{qc_cfg.layout_source}"
        )
    else:
        print(
            f"\nGenerate  K={k_label}  freqs={len(mhz_list)}  samples={NUM_SAMPLES}  "
            f"mode={mode}/{qc_cfg.layout_source}"
        )

    layout_by_k: dict[int, _KGenerateContext] = {}
    for k in k_values:
        layout_by_k[k] = _load_k_generate_context(
            engine,
            device,
            dataclasses.replace(qc_cfg, k_value=k),
            k,
            exp_cfg=exp_cfg,
            sweep_val_ld=sweep_val_ld,
            quiet=compact_log,
        )
    if compact_log and mode == "layout_qc":
        print(f"  val layouts: {len(k_values)} K × {NUM_SAMPLES} sample(s)")

    fg_max_table = load_anchor_fg_max_table() if CALIBRATE_FG_MAX else {}

    def _save_sample_bundle(
        *,
        out_dir: Path,
        hm_z: torch.Tensor,
        hm_phys: np.ndarray,
        occ_np: np.ndarray,
        mhz: float,
        imp_log_np: np.ndarray | None,
        n_batch: int,
    ) -> None:
        for i in range(n_batch):
            sd = out_dir / f"data_sample_{i}"
            sd.mkdir(parents=True, exist_ok=True)
            hm_norm_np = (
                engine.heatmap_train_to_disk(hm_z[i]).cpu().numpy()
                if hasattr(engine, "heatmap_train_to_disk")
                else hm_z[i].cpu().numpy()
            )
            np.save(sd / "heatmap_norm.npy", hm_norm_np)
            np.save(sd / "heatmap_zscore.npy", hm_norm_np)
            np.save(sd / "heatmap_physical.npy", hm_phys[i])
            np.save(sd / "occupancy_map.npy", occ_np[i])
            np.save(sd / "pi_freq_mhz.npy", np.array(mhz, dtype=np.float32))
            if imp_log_np is not None:
                np.save(sd / "impedance_profile.npy", imp_log_np[i].reshape(-1))

    if impedance_run:
        decode_mhz = float(PI_REF_MHZ)
        for k in k_values:
            ctx = layout_by_k[k]
            n_batch = ctx.n_batch
            out_dir = out_root / f"K{k}"
            out_dir.mkdir(parents=True, exist_ok=True)

            hm_z, occ_prob, imp_norm = _decode_at_mhz_k(
                engine, device, qc_cfg, ctx, mhz=decode_mhz, k=k,
            )
            occ_bin = topk_occ_binary(occ_prob, k).cpu().numpy().astype(np.int8)
            hm_phys = _hm_physical(hm_z, engine, mhz=decode_mhz).cpu().numpy()
            if CALIBRATE_FG_MAX and fg_max_table and not is_training_anchor(decode_mhz):
                target_max = interp_anchor_value(decode_mhz, fg_max_table)
                for i in range(hm_phys.shape[0]):
                    hm_phys[i] = calibrate_heatmap_physical(
                        hm_phys[i], mask, target_max,
                    )
            occ_np = occ_bin
            np.save(out_dir / "occupancy.npy", occ_np)
            all_occupancies.append(occ_np)
            imp_log_np = _impedance_log_from_norm(engine, imp_norm).cpu().numpy()
            _save_sample_bundle(
                out_dir=out_dir,
                hm_z=hm_z,
                hm_phys=hm_phys,
                occ_np=occ_np,
                mhz=decode_mhz,
                imp_log_np=imp_log_np,
                n_batch=n_batch,
            )
            if not compact_log:
                print(f"  K={k:2d}  impedance profile saved ({n_batch} sample(s))")
    else:
        for mhz in mhz_list:
            tag = _freq_tag(mhz)
            mhz_maxes: list[float] = []
            for k in k_values:
                ctx = layout_by_k[k]
                n_batch = ctx.n_batch
                out_dir = out_root / tag / f"K{k}"
                out_dir.mkdir(parents=True, exist_ok=True)

                hm_z, occ_prob, imp_norm = _decode_at_mhz_k(
                    engine, device, qc_cfg, ctx, mhz=mhz, k=k,
                )

                occ_bin = topk_occ_binary(occ_prob, k).cpu().numpy().astype(np.int8)

                hm_phys = _hm_physical(hm_z, engine, mhz=mhz).cpu().numpy()
                if CALIBRATE_FG_MAX and fg_max_table and not is_training_anchor(mhz):
                    target_max = interp_anchor_value(mhz, fg_max_table)
                    for i in range(hm_phys.shape[0]):
                        hm_phys[i] = calibrate_heatmap_physical(
                            hm_phys[i], mask, target_max,
                        )
                occ_np = occ_bin
                np.save(out_dir / "occupancy.npy", occ_np)
                all_occupancies.append(occ_np)
                mhz_int = int(round(mhz))
                all_peb_freqs.extend([f"{mhz_int}e6"] * n_batch)

                _save_sample_bundle(
                    out_dir=out_dir,
                    hm_z=hm_z,
                    hm_phys=hm_phys,
                    occ_np=occ_np,
                    mhz=mhz,
                    imp_log_np=None,
                    n_batch=n_batch,
                )

                st = _hm_stats(hm_phys[0], mask)
                rows.append({"mhz": mhz, "k": k, "pi_norm": pi_freq_mhz_to_norm(mhz), **st})
                mhz_maxes.append(st["max"])
                if not compact_log:
                    print(f"  {tag:12s} K={k:2d}  max={st['max']:.4f}  mean_fg={st['mean_fg']:.4f}")
            if compact_log and mhz_maxes:
                print(
                    f"  {tag:12s}  K×{len(k_values)}  "
                    f"max {min(mhz_maxes):.3f}–{max(mhz_maxes):.3f} Ω"
                )

    occupancy_all = (
        np.concatenate(all_occupancies, axis=0)
        if all_occupancies
        else np.zeros((0, 52), dtype=np.int8)
    )
    peb_file = Path(PEB_OUT_FILE)
    peb_file.parent.mkdir(parents=True, exist_ok=True)
    peb_kw: dict[str, Any] = dict(
        occupancy=occupancy_all,
        output_path=str(peb_file),
        powerbus=POWERBUS,
        components=COMPONENTS,
        include_distribution=not impedance_run,
        include_spectrum=impedance_run,
    )
    if impedance_run:
        peb_kw["freq"] = f"{int(round(PI_REF_MHZ))}e6"
    else:
        peb_kw["freq"] = f"{int(round(mhz_list[0]))}e6"
        peb_kw["per_sample_freqs"] = all_peb_freqs
    generate_peb(**peb_kw)
    print(f"  PEB ({_peb_kind()}) → {peb_file.name}  ({len(occupancy_all)} configs)")
    _copy_peb_if_configured(peb_file)

    csv_path = out_root / "generate_summary.csv"
    if impedance_run:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["k", "num_samples", "decode_mhz"])
            w.writeheader()
            for k in k_values:
                w.writerow({
                    "k": k,
                    "num_samples": int(layout_by_k[k].n_batch),
                    "decode_mhz": decode_mhz,
                })
    else:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["mhz", "k", "pi_norm", "max", "mean_fg", "p95"])
            w.writeheader()
            w.writerows(rows)

        mhz_arr = np.array([r["mhz"] for r in rows])
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        if len(k_values) == 1:
            ax.plot(mhz_arr, [r["max"] for r in rows], "o-", label="FG max")
            ax.plot(mhz_arr, [r["mean_fg"] for r in rows], "s-", label="FG mean")
        else:
            for k in k_values:
                sub = [r for r in rows if r["k"] == k]
                if not sub:
                    continue
                ax.plot(
                    [r["mhz"] for r in sub],
                    [r["max"] for r in sub],
                    "o-",
                    label=f"K={k} max",
                )
        for a in ANCHOR_MHZ:
            ax.axvline(a, color="gray", ls="--", alpha=0.35, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("PI frequency (MHz)")
        ax.set_ylabel("Heatmap (physical)")
        ax.set_title(f"Generated heatmap vs PI_freq (K={k_label}, shared_temp={SHARED_TEMP})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(out_root / "generate_heatmap_vs_mhz.png", dpi=200)
        plt.close(fig)

    manifest_mhz = [] if impedance_run else mhz_list
    manifest = write_sweep_freq_manifest(
        manifest_mhz,
        out_root,
        k_values=k_values,
        extra=qc_manifest_extra(qc_cfg),
    )
    if k_values != normalize_k_values(K_VALUE):
        dropped = [k for k in normalize_k_values(K_VALUE) if k not in k_values]
        print(f"  manifest K: {len(k_values)} effective (dropped not in val: {dropped})")

    if RUN_SWEEP_QC_EVAL and not impedance_run:
        if exp_cfg is None:
            exp_cfg = _load_experiment_config(PROJECT_ROOT / EXPERIMENT_DIR / "config.yaml")
        if (
            sweep_val_ld is None
            and qc_cfg.inference_mode == "layout_qc"
            and qc_cfg.layout_source == "val"
        ):
            sweep_val_ld = get_sweep_val_loader(
                data_dir=PROJECT_ROOT / DATA_DIR,
                experiment_cfg=exp_cfg,
                device=device,
            )
        run_sweep_qc_eval(
            engine,
            device,
            out_root=out_root,
            experiment_dir=EXPERIMENT_DIR,
            checkpoint_path=CHECKPOINT_PATH,
            data_dir=PROJECT_ROOT / DATA_DIR,
            experiment_cfg=exp_cfg,
            qc_cfg=qc_cfg,
            gen_rows=rows,
            sweep_mhz=mhz_list,
            k_values=k_values,
            val_ld=sweep_val_ld,
        )

    print(f"\n  out: {out_root}")
    print(f"  csv: {csv_path.name}  manifest: {manifest.name}")
    return csv_path


@torch.inference_mode()
def run_val(
    engine: Any,
    mhz_list: list[float],
    out_root: Path,
    device: torch.device,
    max_batches: int,
) -> Path:
    """Val heatmap recon at native PI_freq + decode same batch at every sweep MHz."""
    cfg_path = PROJECT_ROOT / EXPERIMENT_DIR / "config.yaml"
    cfg = _load_experiment_config(cfg_path)

    _, val_ld = create_multifreq_data_loaders(
        data_dir=str(PROJECT_ROOT / DATA_DIR),
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=2,
        train_split=float(cfg.get("train_split", 0.9)),
        seed=42,
        pin_memory=device.type == "cuda",
        split_by_design=cfg.get("split_by_design", True),
        stratify_by_k=cfg.get("stratify_by_k", True),
        balance_k=False,
        balance_freq=False,
        cache_in_ram=cfg.get("cache_in_ram", True),
    )

    fg_thr = _fg_threshold(engine, cfg)
    model = engine.model
    model.eval()
    norm_label = "global-max" if getattr(engine, "use_global_max", False) else "z-score"

    native_by_anchor: dict[float, list[float]] = {a: [] for a in ANCHOR_MHZ}
    cross_by_mhz: dict[float, list[float]] = defaultdict(list)

    print(
        f"\n=== Val heatmap recon ({len(val_ld)} batches, max_batches={max_batches or 'all'}, "
        f"norm={norm_label}, fg_thr={fg_thr:.4f}) ==="
    )

    for bi, batch in enumerate(val_ld):
        if max_batches > 0 and bi >= max_batches:
            break

        hm = batch["heatmap_norm"].to(device)
        occ = batch["occupancy"].to(device)
        imp = batch["impedance"].to(device)
        K = batch["K"].to(device)
        pi_native = batch["PI_freq"].to(device)
        if hm.dim() == 3:
            hm = hm.unsqueeze(1)
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        elif imp.dim() == 3:
            imp = imp[:, :1]
        hm_enc = hm.masked_fill(hm < fg_thr, 0.0)

        rh_native, _, _, _, _, _ = model(hm_enc, occ, imp, K, pi_native)
        mse_native = _fg_mse(rh_native, hm, fg_thr)
        for j in range(mse_native.shape[0]):
            mhz_j = _norm_to_mhz(float(pi_native[j].item()))
            anchor = min(ANCHOR_MHZ, key=lambda a: abs(a - mhz_j))
            native_by_anchor[anchor].append(float(mse_native[j].item()))

        for mhz in mhz_list:
            pi_t = torch.full((hm.shape[0],), pi_freq_mhz_to_norm(mhz), device=device)
            rh, _, _, _, _, _ = model(hm_enc, occ, imp, K, pi_t)
            mse = _fg_mse(rh, hm, fg_thr)
            cross_by_mhz[mhz].extend(mse.cpu().tolist())

    rows: list[dict[str, Any]] = []
    for anchor in ANCHOR_MHZ:
        vals = native_by_anchor[anchor]
        if vals:
            rows.append({
                "mhz": anchor,
                "kind": "native_recon",
                "n": len(vals),
                "hm_fg_mse_mean": float(np.mean(vals)),
                "hm_fg_mse_p50": float(np.median(vals)),
            })

    for mhz in mhz_list:
        vals = cross_by_mhz[mhz]
        rows.append({
            "mhz": mhz,
            "kind": "cross_freq_decode",
            "n": len(vals),
            "hm_fg_mse_mean": float(np.mean(vals)),
            "hm_fg_mse_p50": float(np.median(vals)),
        })

    csv_path = out_root / "val_heatmap_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mhz", "kind", "n", "hm_fg_mse_mean", "hm_fg_mse_p50"])
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    nat = [r for r in rows if r["kind"] == "native_recon"]
    if nat:
        ax.plot(
            [r["mhz"] for r in nat],
            [r["hm_fg_mse_mean"] for r in nat],
            "o-",
            color="green",
            label="Native PI_freq recon",
            lw=2,
        )
    cross = [r for r in rows if r["kind"] == "cross_freq_decode"]
    ax.plot(
        [r["mhz"] for r in cross],
        [r["hm_fg_mse_mean"] for r in cross],
        ".-",
        color="steelblue",
        alpha=0.85,
        label="Decode at sweep MHz (same enc input)",
    )
    for a in ANCHOR_MHZ:
        ax.axvline(a, color="gray", ls="--", alpha=0.35, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("PI frequency (MHz)")
    ax.set_ylabel(f"Val heatmap FG MSE ({norm_label})")
    ax.set_title("Heatmap generalization across PI_freq (exp038)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_root / "val_heatmap_mse_vs_mhz.png", dpi=200)
    plt.close(fig)

    print(f"\n  Val summary: {csv_path}")
    if nat:
        best = min(nat, key=lambda r: r["hm_fg_mse_mean"])
        print(f"  Best native anchor: {best['mhz']} MHz  MSE={best['hm_fg_mse_mean']:.4f}")
    return csv_path


def _path_for_report(path: Path) -> str:
    """Repo-relative posix path for README (handles relative vs absolute paths)."""
    root = PROJECT_ROOT.resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_report(out_root: Path, mhz_list: list[float], modes: list[str]) -> None:
    freqs_preview = ", ".join(
        f"{int(round(m))}" if m == round(m) else f"{m:.2g}" for m in mhz_list[:12]
    ) + (" …" if len(mhz_list) > 12 else "")
    lines = [
        "# multifreq heatmap sweep",
        "",
        f"Checkpoint: `{CHECKPOINT_PATH}`",
        f"K values: **{k_output_tag(K_VALUE)}**",
        f"PI frequencies ({len(mhz_list)}): {freqs_preview}",
        f"Training anchors: {list(ANCHOR_MHZ)} MHz",
        "",
        "## CAD workflow",
        "",
        "1. Run `run_multifreq_heatmap_sweep.py` (generate mode) — creates samples + `.peb`.",
        "2. Batch-simulate the `.peb` in ECADStar (manual).",
        "3. `python scrap/multifreq_move_and_compare.py` after ECADStar batch simulate.",
        "",
        f"PEB path: `{PEB_OUT_FILE}` (PI-{_peb_kind().capitalize()} only, run mode = {str(RUN_MODE).strip().lower()}).",
        "",
        "## How to read results",
        "",
        "- **`freq_*MHz/K{n}/`** — generated heatmaps per frequency (compare vs `Real/` after move).",
        "- **`generate_heatmap_vs_mhz.png`** — if max/mean are flat, heatmap branch may ignore PI_freq.",
        "- **`val_heatmap_summary.csv`** — optional offline val metrics.",
        "",
        "## Outputs",
        "",
    ]
    for name in (
        "generate_summary.csv",
        "generate_heatmap_vs_mhz.png",
        "val_heatmap_summary.csv",
        "val_heatmap_mse_vs_mhz.png",
        Path(PEB_OUT_FILE).name,
    ):
        p = out_root / name if name != Path(PEB_OUT_FILE).name else Path(PEB_OUT_FILE)
        if p.exists():
            lines.append(f"- `{_path_for_report(p)}`")
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_from_config() -> None:
    """Run sweep using module-level CONFIG constants (also used by pipeline orchestrator)."""
    if str(RUN_MODE).strip().lower() not in ("heatmap", "impedance"):
        raise SystemExit(f"RUN_MODE must be 'heatmap' or 'impedance', got {RUN_MODE!r}")
    _sync_derived_paths()
    os.chdir(PROJECT_ROOT)
    print(f"run mode: {str(RUN_MODE).strip().lower()}  (PEB kind: PI-{_peb_kind().capitalize()})")
    mhz_list = _build_mhz_list(SWEEP)
    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    k_values = normalize_k_values(K_VALUE)
    mhz_desc = f"{mhz_list[0]:.4g} MHz" if len(mhz_list) == 1 else f"{len(mhz_list)} freqs ({mhz_list[0]:.4g}–{mhz_list[-1]:.4g} MHz)"
    print(f"device={device}  K={k_values[0]}–{k_values[-1]} ({len(k_values)})  {mhz_desc}")

    engine = _load_engine(device)
    modes_run: list[str] = []
    if MODE in ("generate", "both"):
        run_generate(engine, mhz_list, out_root, device)
        modes_run.append("generate")
    if MODE in ("val", "both"):
        run_val(engine, mhz_list, out_root, device, VAL_MAX_BATCHES)
        modes_run.append("val")

    write_report(out_root, mhz_list, modes_run)
    print(f"done → {out_root.resolve()}")


def main() -> None:
    run_from_config()


if __name__ == "__main__":
    main()
