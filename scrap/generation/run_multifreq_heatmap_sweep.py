"""exp038 — PI frequency sweep at **fixed K** (heatmap / PI-Distribution only).

Tests whether the multifreq-trained VAE generalizes across PI frequency for spatial
PI maps (PI-Distribution). K is held constant; only PI_freq is swept.

Workflow
--------
Full pipeline (generate + ECADStar + move/compare/report)::

    python scrap/run_multifreq_sweep_pipeline.py

Or step-by-step:
1. ``python scrap/generation/run_multifreq_heatmap_sweep.py``
   or ``python scrap/generation/run_multifreq_heatmap_sweep.py --mhz 80 250``
   → samples under ``{OUTPUT_ROOT}/freq_*MHz/K{K}/`` + combined ``.peb`` (PI-Distribution only)
2. Run the ``.peb`` in ECADStar (manual batch simulate), or use the pipeline script.
3. ``python scrap/multifreq_move_and_compare.py``  (move PI-* + heatmap compare)
   — or separately: ``move_pi_to_real.py`` then ``comparison/compare.py``

Does **not** replace ``scrap/generation/run_all_k.py`` (K sweep + spectrum PEB).

Modes:
    generate — decode at each MHz (fixed K), write PEB, summary plots
    val      — val-set heatmap recon (optional offline metric)
    both     — generate then val
"""

from __future__ import annotations

import argparse
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

# ── Configuration (edit here or override via CLI) ─────────────────────────────
EXPERIMENT_DIR = "experiments/exp043"
CHECKPOINT_PATH = f"{EXPERIMENT_DIR}/checkpoints/last_model.pt"
DATA_DIR = "datasets/data_multifreq_gmax"
OUTPUT_ROOT = f"{EXPERIMENT_DIR}/multifreq_heatmap_sweep_29"  # updated by CLI args; also encoded in PEB filename

# PI sweep: "anchors" (training bins), "dense" (log-spaced 1–600 MHz), or explicit list
SWEEP = [10, 70, 120, 270, 400]  # "anchors" | "dense" | [10, 63, 100, ...]
DENSE_N_POINTS = 24

K_VALUE = 29  # fixed — this sweep does not vary K
NUM_SAMPLES = 2
SHARED_TEMP = 1.5
SEED = 42

# PEB: PI-Distribution only (no CreatePISpectrum) — one PI-* output per sample in CAD
HEATMAP_ONLY_PEB = True
PEB_OUT_FILE = f"{OUTPUT_ROOT}/pi_distribution_K{K_VALUE}_freq_sweep.peb"
POWERBUS = "Power_GND"
PEB_COPY_DEST: str | None = r"C:\Users\muthusamy\Desktop\design\PEB"
REPORT_COPY_DEST: str | None = r"C:\Users\muthusamy\Desktop\reports"

FORCE_CPU = False
VAL_MAX_BATCHES = 0  # 0 = full val loader
BACKGROUND_MARGIN = 0.5

# Inference: "anchor_blend" (off-anchor MHz) | "layout" | "marginal" | "encode" (val only)
INFERENCE_MODE = "anchor_blend"
PI_REF_MHZ = 200.0  # layout occ/imp reference before per-MHz decode
# Scale physical heatmap FG max toward anchor statistics (helps unseen MHz e.g. 80, 250).
# Set False when debugging — calibration can amplify a wrong single-pixel blob to ~anchor max.
CALIBRATE_FG_MAX = False
# Only used when the loaded model defines inference_factorized_only (exp040); ignored for exp039.
INFERENCE_FACTORIZED_ONLY = False
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp038_true_multi.codes.dataloader_multifreq import (  # noqa: E402
    ANCHOR_MHZ,
    create_multifreq_data_loaders,
)
from scrap.generation.generate_peb import generate_peb  # noqa: E402
from scrap.peb_copy import copy_peb_to_folder  # noqa: E402
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
)


SWEEP_FREQ_MANIFEST = "sweep_frequencies_mhz.json"


def sweep_freq_manifest_path(out_root: str | Path | None = None) -> Path:
    return Path(out_root if out_root is not None else OUTPUT_ROOT) / SWEEP_FREQ_MANIFEST


def write_sweep_freq_manifest(
    mhz_list: list[float],
    out_root: str | Path | None = None,
) -> Path:
    """Persist MHz list used by the last generate run (move/compare read this)."""
    path = sweep_freq_manifest_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mhz": [float(m) for m in mhz_list],
        "k": int(K_VALUE),
        "num_samples": int(NUM_SAMPLES),
    }
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


def _load_vae_inference():
    """Import VAEInference from the experiment package (creates module if missing)."""
    exp_mod = EXPERIMENT_DIR.replace("/", ".").replace("\\", ".") + ".codes.inference_vae"
    try:
        return importlib.import_module(exp_mod).VAEInference
    except ModuleNotFoundError:
        fallback = "experiments.exp038_true_multi.codes.inference_vae"
        print(f"WARNING: {exp_mod} not found — using {fallback}")
        return importlib.import_module(fallback).VAEInference


VAEInference = _load_vae_inference()


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


def _hm_physical(hm_z: torch.Tensor, engine: Any) -> torch.Tensor:
    """Denorm heatmap tensor to Ω (supports exp043 global-max and exp042 z-score)."""
    if hasattr(engine, "denorm_heatmap_physical"):
        return engine.denorm_heatmap_physical(hm_z)
    from src_vae.others.heatmap_z_clip import heatmap_norm_to_physical, heatmap_z_to_physical

    hm_stats = getattr(engine, "hm_stats", None)
    clip = getattr(engine, "hm_z_clip", None)
    lo, hi = (clip[0], clip[1]) if clip is not None else (None, None)
    if hm_stats is not None and (
        hm_stats.get("norm_mode") == "global_max" or hm_stats.get("global_max_ohm") is not None
    ):
        return heatmap_norm_to_physical(hm_z, hm_stats, clip_lo=lo, clip_hi=hi)
    return heatmap_z_to_physical(
        hm_z, engine.hm_log_mean, engine.hm_log_std, clip_lo=lo, clip_hi=hi,
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


def run_generate(engine: Any, mhz_list: list[float], out_root: Path, device: torch.device) -> Path:
    """Decode at each MHz with fixed K; save heatmaps, occupancy, and combined PEB."""
    out_root.mkdir(parents=True, exist_ok=True)
    mask = engine.binary_mask
    rows: list[dict[str, Any]] = []
    all_occupancies: list[np.ndarray] = []
    all_peb_freqs: list[str] = []

    torch.manual_seed(SEED)
    print(
        f"\n=== Generate PI-freq sweep: K={K_VALUE} (fixed), {len(mhz_list)} frequencies, "
        f"{NUM_SAMPLES} sample(s) each, inference={INFERENCE_MODE} ==="
    )

    shared_occ: torch.Tensor | None = None
    shared_imp: torch.Tensor | None = None
    fg_max_table = load_anchor_fg_max_table() if CALIBRATE_FG_MAX else {}
    if INFERENCE_MODE in ("layout", "anchor_blend"):
        with torch.no_grad():
            _, occ_prob0, imp0 = engine.model.inference(
                NUM_SAMPLES,
                device,
                K=K_VALUE,
                PI_freq=PI_REF_MHZ,
                pi_freq_unit="mhz",
                latent_stats=engine.latent_stats,
                per_K_latent_stats=engine.per_K_latent_stats,
                shared_temp=SHARED_TEMP,
                mode="marginal",
            )
            shared_occ = occ_prob0
            shared_imp = imp0

    for mhz in mhz_list:
        tag = _freq_tag(mhz)
        out_dir = out_root / tag / f"K{K_VALUE}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            inf_kw: dict[str, Any] = dict(
                latent_stats=engine.latent_stats,
                per_K_latent_stats=engine.per_K_latent_stats,
                shared_temp=SHARED_TEMP,
                mode=INFERENCE_MODE,
                pi_ref_mhz=PI_REF_MHZ,
            )
            if INFERENCE_MODE in ("layout", "anchor_blend") and shared_occ is not None:
                inf_kw["occupancy"] = shared_occ
                inf_kw["impedance"] = shared_imp
            hm_z, occ_prob, imp_norm = engine.model.inference(
                NUM_SAMPLES,
                device,
                K=K_VALUE,
                PI_freq=mhz,
                pi_freq_unit="mhz",
                **inf_kw,
            )

        occ_bin = torch.zeros_like(occ_prob)
        if K_VALUE > 0:
            topk_idx = occ_prob.topk(min(K_VALUE, occ_prob.shape[-1]), dim=-1).indices
            occ_bin.scatter_(-1, topk_idx, 1.0)

        hm_phys = _hm_physical(hm_z, engine).cpu().numpy()
        if CALIBRATE_FG_MAX and fg_max_table and not is_training_anchor(mhz):
            target_max = interp_anchor_value(mhz, fg_max_table)
            for i in range(hm_phys.shape[0]):
                hm_phys[i] = calibrate_heatmap_physical(
                    hm_phys[i], mask, target_max,
                )
        occ_np = occ_bin.cpu().numpy().astype(np.int8)
        np.save(out_dir / "occupancy.npy", occ_np)
        all_occupancies.append(occ_np)
        mhz_int = int(round(mhz))
        all_peb_freqs.extend([f"{mhz_int}e6"] * NUM_SAMPLES)

        for i in range(NUM_SAMPLES):
            sd = out_dir / f"data_sample_{i}"
            sd.mkdir(parents=True, exist_ok=True)
            hm_norm_np = hm_z[i].cpu().numpy()
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

    manifest = write_sweep_freq_manifest(mhz_list, out_root)
    print(f"  Freq manifest: {manifest}")

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


def main() -> None:
    ap = argparse.ArgumentParser(description="exp038 PI_freq heatmap sweep at fixed K")
    ap.add_argument("--mode", choices=("generate", "val", "both"), default="generate")
    ap.add_argument("--sweep", default=None, help="anchors | dense | ignored if SWEEP is a list in file")
    ap.add_argument(
        "--mhz",
        nargs="+",
        type=float,
        default=None,
        metavar="MHZ",
        help="Explicit PI frequencies in MHz (overrides SWEEP and --sweep), e.g. --mhz 80 250",
    )
    ap.add_argument("--max-batches", type=int, default=VAL_MAX_BATCHES)
    ap.add_argument("--k", type=int, default=None, help="fixed K (default from config)")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--inference-mode",
        choices=("marginal", "layout", "anchor_blend", "encode"),
        default=None,
    )
    ap.add_argument("--no-calibrate-fg", action="store_true", help="Disable FG max calibration")
    ap.add_argument("--pi-ref-mhz", type=float, default=None)
    args = ap.parse_args()

    global K_VALUE, SWEEP, OUTPUT_ROOT, PEB_OUT_FILE, INFERENCE_MODE, PI_REF_MHZ, CALIBRATE_FG_MAX
    if args.k is not None:
        K_VALUE = args.k
    if args.out:
        OUTPUT_ROOT = args.out
    if args.mhz is not None:
        SWEEP = [float(m) for m in args.mhz]
    elif args.sweep:
        SWEEP = args.sweep
    if args.inference_mode:
        INFERENCE_MODE = args.inference_mode
    if args.pi_ref_mhz is not None:
        PI_REF_MHZ = args.pi_ref_mhz
    if args.no_calibrate_fg:
        CALIBRATE_FG_MAX = False
    PEB_OUT_FILE = f"{OUTPUT_ROOT}/pi_distribution_K{K_VALUE}_freq_sweep.peb"

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
    if args.mode in ("generate", "both"):
        run_generate(engine, mhz_list, out_root, device)
        modes_run.append("generate")
    if args.mode in ("val", "both"):
        run_val(engine, mhz_list, out_root, device, args.max_batches)
        modes_run.append("val")

    write_report(out_root, mhz_list, modes_run)
    print(f"\nDone → {out_root.resolve()}")
    print(f"  Report: {out_root / 'README.md'}")


if __name__ == "__main__":
    main()
