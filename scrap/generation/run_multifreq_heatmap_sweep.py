"""Multifreq Heatmap Sweep at Fixed K.

Purpose: Decode VAE samples at multiple PI frequencies (fixed K) for PI-Distribution
    heatmaps; write per-freq folders, combined .peb, sweep manifest, and summary plots.
Run: python scrap/generation/run_multifreq_heatmap_sweep.py
Inputs / outputs: CHECKPOINT_PATH, DATA_DIR, SWEEP, K_VALUE → {OUTPUT_ROOT}/freq_*MHz/K{K}/;
    PEB_OUT_FILE (.peb), sweep_freq_manifest.json; optional PEB_COPY_DEST.
Dependencies: torch, matplotlib, numpy; dynamic VAEInference from EXPERIMENT_DIR.
Agent notes:
    - Type: CLI generator (does not replace run_all_k K-sweep workflow)
    - Key symbols: main, run_generate, run_val, exported_freq_mhz_list,
    - Config keys: ``MODE``, ``EXPERIMENT_DIR``, ``CHECKPOINT_PATH``, ``DATA_DIR``, ``OUTPUT_ROOT``, ``DENSE_N_POINTS``, ``K_VALUE``, ``NUM_SAMPLES``, ``SHARED_TEMP``, ``SEED``, ``HEATMAP_ONLY_PEB``, ``PEB_OUT_FILE``, ``POWERBUS``, ``FORCE_CPU``, ``VAL_MAX_BATCHES``, ``BACKGROUND_MARGIN``, ``INFERENCE_MODE``, ``PI_REF_MHZ``, ``CALIBRATE_FG_MAX``, ``INFERENCE_FACTORIZED_ONLY``
      write_sweep_freq_manifest, HEATMAP_ONLY_PEB, K_VALUE, OUTPUT_ROOT
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

K_VALUE = 30
NUM_SAMPLES = 2
SHARED_TEMP = 1.5
SEED = 42

HEATMAP_ONLY_PEB = True
PEB_OUT_FILE = f"{OUTPUT_ROOT}/pi_distribution_K{K_VALUE}_freq_sweep.peb"
POWERBUS = "Power_GND"
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


def _sync_derived_paths() -> None:
    """Keep PEB_OUT_FILE aligned with OUTPUT_ROOT and K_VALUE."""
    global PEB_OUT_FILE
    PEB_OUT_FILE = f"{OUTPUT_ROOT}/pi_distribution_K{K_VALUE}_freq_sweep.peb"


SWEEP_FREQ_MANIFEST = "sweep_frequencies_mhz.json"


def sweep_freq_manifest_path(out_root: str | Path | None = None) -> Path:
    return Path(out_root if out_root is not None else OUTPUT_ROOT) / SWEEP_FREQ_MANIFEST


def write_sweep_freq_manifest(
    mhz_list: list[float],
    out_root: str | Path | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist MHz list used by the last generate run (move/compare read this)."""
    path = sweep_freq_manifest_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "mhz": [float(m) for m in mhz_list],
        "k": int(K_VALUE),
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
    """Integer MHz list for move/compare (matches folder tags under OUTPUT_ROOT)."""
    recorded = load_sweep_freq_mhz_list(out_root)
    if recorded is not None:
        return [int(round(m)) for m in recorded]
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


def _load_engine(device: torch.device) -> Any:
    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    VAEInference = _resolve_vae_inference()
    print(f"  VAEInference : {VAEInference.__module__}.{VAEInference.__qualname__}")
    engine = VAEInference(
        checkpoint_path=str(ckpt),
        device=device,
    )
    if hasattr(engine.model, "inference_factorized_only"):
        engine.model.inference_factorized_only = bool(INFERENCE_FACTORIZED_ONLY)
        print(f"  inference_factorized_only={engine.model.inference_factorized_only}")
    stats_auto = PROJECT_ROOT / EXPERIMENT_DIR / "metrics" / "latent_stats.json"
    engine.load_latent_stats(str(stats_auto) if stats_auto.is_file() else None)
    if getattr(engine, "use_global_max", False):
        print(
            f"  Heatmap norm: global-max  gmax={engine.global_max_ohm:.4f} Ω  "
            f"fg_thr={engine.fg_threshold():.4f}"
        )
    elif hasattr(engine, "hm_stats"):
        print(f"  Heatmap norm: {engine.hm_stats.describe()}")
    else:
        print(
            f"  Heatmap norm: z-score  log_mean={engine.hm_log_mean:.4f}  "
            f"log_std={engine.hm_log_std:.4f}"
        )
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
        k_value=int(K_VALUE),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random occ/imp from marginal z — only when ALLOW_RANDOM_LAYOUT=True."""
    with torch.no_grad():
        _, occ_prob0, imp0 = engine.model.inference(
            num_samples,
            device,
            K=K_VALUE,
            PI_freq=PI_REF_MHZ,
            pi_freq_unit="mhz",
            latent_stats=engine.latent_stats,
            per_K_latent_stats=engine.per_K_latent_stats,
            shared_temp=SHARED_TEMP,
            mode="marginal",
        )
    return occ_prob0, imp0


def run_generate(engine: Any, mhz_list: list[float], out_root: Path, device: torch.device) -> Path:
    """Decode at each MHz with fixed K; save heatmaps, occupancy, and combined PEB."""
    qc_cfg = _build_qc_config()
    try:
        validate_sweep_qc(qc_cfg)
    except SweepQCError as exc:
        raise SystemExit(f"Sweep QC config error: {exc}") from exc

    mode = qc_cfg.inference_mode.strip().lower()
    out_root.mkdir(parents=True, exist_ok=True)
    mask = engine.binary_mask
    rows: list[dict[str, Any]] = []
    all_occupancies: list[np.ndarray] = []
    all_peb_freqs: list[str] = []

    torch.manual_seed(SEED)
    print(
        f"\n=== Generate PI-freq sweep: K={K_VALUE} (fixed), {len(mhz_list)} frequencies, "
        f"{NUM_SAMPLES} sample(s), qc_mode={mode}, layout_source={qc_cfg.layout_source} ==="
    )

    shared_occ: torch.Tensor | None = None
    shared_imp: torch.Tensor | None = None
    shared_k: torch.Tensor | None = None
    z_fixed: torch.Tensor | None = None
    n_batch = int(NUM_SAMPLES)
    exp_cfg: dict[str, Any] | None = None
    sweep_val_ld = None

    if mode == "layout_qc":
        exp_cfg = _load_experiment_config(PROJECT_ROOT / EXPERIMENT_DIR / "config.yaml")
        sweep_val_ld = get_sweep_val_loader(
            data_dir=PROJECT_ROOT / DATA_DIR,
            experiment_cfg=exp_cfg,
            device=device,
        )
        shared_occ, shared_imp, shared_k = load_val_layout_samples(
            data_dir=PROJECT_ROOT / DATA_DIR,
            experiment_cfg=exp_cfg,
            k_value=K_VALUE,
            num_samples=n_batch,
            seed=SEED,
            device=device,
            val_ld=sweep_val_ld,
        )
        n_batch = int(shared_occ.shape[0])
        print(f"  layout_qc: {n_batch} val layout(s) at K={K_VALUE}")

    elif mode in ("latent_z", "layout_hybrid"):
        if qc_cfg.layout_source == "latent_run":
            bundle = load_latent_run_bundle(
                qc_cfg.latent_run_dir or "",
                K_VALUE,
                device,
                imp_log_mean=float(engine.imp_log_mean),
                imp_log_std=float(engine.imp_log_std),
            )
        else:
            bundle = load_explicit_npy_bundle(
                z_npy=qc_cfg.explicit_z_npy,
                occ_npy=qc_cfg.explicit_occ_npy,
                imp_npy=qc_cfg.explicit_imp_npy,
                k=K_VALUE,
                device=device,
                imp_log_mean=float(engine.imp_log_mean),
                imp_log_std=float(engine.imp_log_std),
            )
        z_fixed = bundle["z"]
        shared_occ = bundle["occ_prob"]
        shared_imp = bundle["imp_norm"]
        shared_k = bundle["K"]
        n_batch = int(z_fixed.shape[0])
        print(f"  {mode}: loaded z {tuple(z_fixed.shape)} from {qc_cfg.layout_source}")

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
            print(f"  layout_hybrid: merged optimized shared + layout private head")

    fg_max_table = load_anchor_fg_max_table() if CALIBRATE_FG_MAX else {}

    for mhz in mhz_list:
        tag = _freq_tag(mhz)
        out_dir = out_root / tag / f"K{K_VALUE}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            if mode == "layout_qc":
                assert shared_occ is not None and shared_imp is not None
                hm_z, occ_prob, imp_norm = engine.model.inference(
                    n_batch,
                    device,
                    K=K_VALUE,
                    PI_freq=mhz,
                    pi_freq_unit="mhz",
                    latent_stats=engine.latent_stats,
                    per_K_latent_stats=engine.per_K_latent_stats,
                    shared_temp=SHARED_TEMP,
                    mode="layout",
                    pi_ref_mhz=PI_REF_MHZ,
                    occupancy=shared_occ,
                    impedance=shared_imp,
                )
            elif mode in ("latent_z", "layout_hybrid"):
                assert z_fixed is not None and shared_k is not None
                occ_decode = shared_occ
                hm_list = decode_heatmap_sweep_mhz(
                    engine.model,
                    z_fixed,
                    shared_k,
                    [mhz],
                    device,
                    occ_for_decode=occ_decode,
                )
                hm_z = hm_list[0]
                pi_t = pi_freq_norm_for_model(mhz, n_batch, unit="mhz", device=device)
                _, occ_logits, imp_norm = engine.model.decode(
                    z_fixed, shared_k, pi_t, occupancy=occ_decode,
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
                    occ0, imp0 = _legacy_marginal_layout_draw(engine, device, n_batch)
                    inf_kw["occupancy"] = occ0
                    inf_kw["impedance"] = imp0
                hm_z, occ_prob, imp_norm = engine.model.inference(
                    n_batch,
                    device,
                    K=K_VALUE,
                    PI_freq=mhz,
                    pi_freq_unit="mhz",
                    **inf_kw,
                )
            else:
                raise SystemExit(f"Unhandled inference mode: {mode!r}")

        occ_bin = topk_occ_binary(occ_prob, K_VALUE).cpu().numpy().astype(np.int8)

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

        st = _hm_stats(hm_phys[0], mask)
        rows.append({"mhz": mhz, "pi_norm": pi_freq_mhz_to_norm(mhz), **st})
        print(f"  {tag:16s}  max={st['max']:.4f}  mean_fg={st['mean_fg']:.4f}")

    occupancy_all = (
        np.concatenate(all_occupancies, axis=0)
        if all_occupancies
        else np.zeros((0, 52), dtype=np.int8)
    )
    peb_file = Path(PEB_OUT_FILE)
    peb_file.parent.mkdir(parents=True, exist_ok=True)
    generate_peb(
        occupancy=occupancy_all,
        output_path=str(peb_file),
        powerbus=POWERBUS,
        freq=f"{int(round(mhz_list[0]))}e6",
        per_sample_freqs=all_peb_freqs,
        include_distribution=True,
        include_spectrum=not HEATMAP_ONLY_PEB,
    )
    print(f"  Combined PEB  : {peb_file}  ({len(occupancy_all)} configs, PI-Distribution only)")
    _copy_peb_if_configured(peb_file)

    csv_path = out_root / "generate_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mhz", "pi_norm", "max", "mean_fg", "p95"])
        w.writeheader()
        w.writerows(rows)

    mhz_arr = np.array([r["mhz"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(mhz_arr, [r["max"] for r in rows], "o-", label="FG max")
    ax.plot(mhz_arr, [r["mean_fg"] for r in rows], "s-", label="FG mean")
    for a in ANCHOR_MHZ:
        ax.axvline(a, color="gray", ls="--", alpha=0.35, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("PI frequency (MHz)")
    ax.set_ylabel("Heatmap (physical)")
    ax.set_title(f"Generated heatmap vs PI_freq (K={K_VALUE}, shared_temp={SHARED_TEMP})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_root / "generate_heatmap_vs_mhz.png", dpi=200)
    plt.close(fig)

    manifest = write_sweep_freq_manifest(mhz_list, out_root, extra=qc_manifest_extra(qc_cfg))
    print(f"  Freq manifest: {manifest}")

    if RUN_SWEEP_QC_EVAL:
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
            val_ld=sweep_val_ld,
        )

    print(f"\n  Samples under: {out_root}")
    print(f"  Summary CSV : {csv_path}")
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
        "# exp038 multifreq heatmap sweep (fixed K)",
        "",
        f"Checkpoint: `{CHECKPOINT_PATH}`",
        f"Fixed K: **{K_VALUE}**",
        f"PI frequencies ({len(mhz_list)}): {freqs_preview}",
        f"Training anchors: {list(ANCHOR_MHZ)} MHz",
        "",
        "## CAD workflow",
        "",
        "1. Run `run_multifreq_heatmap_sweep.py` (generate mode) — creates samples + `.peb`.",
        "2. Batch-simulate the `.peb` in ECADStar (manual).",
        "3. `python scrap/multifreq_move_and_compare.py` after ECADStar batch simulate.",
        "",
        f"PEB path: `{PEB_OUT_FILE}` (PI-Distribution only, no PI-Spectrum).",
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
    _sync_derived_paths()
    os.chdir(PROJECT_ROOT)
    mhz_list = _build_mhz_list(SWEEP)
    out_root = Path(OUTPUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Fixed K={K_VALUE}")
    if len(mhz_list) == 1:
        print(f"Frequencies: {mhz_list[0]:.4g} MHz")
    else:
        print(f"Frequencies ({len(mhz_list)}): {mhz_list[0]:.4g} … {mhz_list[-1]:.4g} MHz")

    engine = _load_engine(device)
    modes_run: list[str] = []
    if MODE in ("generate", "both"):
        run_generate(engine, mhz_list, out_root, device)
        modes_run.append("generate")
    if MODE in ("val", "both"):
        run_val(engine, mhz_list, out_root, device, VAL_MAX_BATCHES)
        modes_run.append("val")

    write_report(out_root, mhz_list, modes_run)
    print(f"\nDone → {out_root.resolve()}")
    print(f"  Report: {out_root / 'README.md'}")


def main() -> None:
    run_from_config()


if __name__ == "__main__":
    main()
