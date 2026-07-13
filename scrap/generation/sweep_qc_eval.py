"""Lightweight sweep QC metrics + agent-copy report.

Runs after ``run_generate`` — only metrics needed to judge whether to proceed training.
"""
from __future__ import annotations

import importlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.exp038_true_multi.codes.dataloader_multifreq import ANCHOR_MHZ
from scrap.generation.sweep_latent_opt_rules import (
    SweepQCConfig,
    get_sweep_val_loader,
    load_val_anchor_samples,
)
from src_vae.others.heatmap_z_clip import heatmap_z_to_physical
from src_vae.others.pi_freq_utils import pi_freq_norm_for_model


def _spatial_metrics(experiment_dir: str):
    mod_path = experiment_dir.replace("/", ".").replace("\\", ".") + ".codes.spatial_metrics"
    try:
        return importlib.import_module(mod_path)
    except ModuleNotFoundError:
        from experiments.exp050.codes import spatial_metrics

        return spatial_metrics


def _fg_mse(recon: torch.Tensor, target: torch.Tensor, fg_thr: float) -> torch.Tensor:
    fg = (target > fg_thr).float()
    err = (recon - target).pow(2)
    n = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (err * fg).sum(dim=(1, 2, 3)) / n


def clip_ceiling_ohm(engine: Any, mhz: float | None = None) -> float | None:
    """Physical Ω reference ceiling for QC (p99.5 soft ref when unbounded)."""
    hm_stats = getattr(engine, "hm_stats", None)
    if hm_stats is not None:
        if getattr(hm_stats, "is_unbounded", lambda: False)():
            fn = getattr(hm_stats, "physical_ceiling_ohm_soft", hm_stats.physical_ceiling_ohm)
            return fn(mhz)
        return hm_stats.physical_ceiling_ohm(mhz)
    norm = getattr(engine, "norm_stats", None)
    if norm is not None:
        return norm.heatmap.physical_ceiling_ohm(mhz)
    clip = getattr(engine, "hm_z_clip", None)
    if clip is None:
        return None
    hi = float(clip[1])
    t = torch.tensor([[[[hi]]]], dtype=torch.float32)
    phys = heatmap_z_to_physical(
        t,
        engine.hm_log_mean,
        engine.hm_log_std,
        hm_stats=getattr(engine, "hm_stats", None),
        mhz=mhz,
        clip_lo=None,
        clip_hi=None,
    )
    return float(phys.item())


def _is_near_clip(gen_max: float, ceiling: float | None, *, frac: float = 0.9) -> bool:
    return ceiling is not None and gen_max >= frac * ceiling


def _sort_gen_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (int(r.get("k", 0)), float(r["mhz"])))


def _k_label(k_values: list[int]) -> str:
    if not k_values:
        return "—"
    if len(k_values) == 1:
        return str(k_values[0])
    return f"{k_values[0]}–{k_values[-1]} ({len(k_values)} values)"


def _multi_k(k_values: list[int]) -> bool:
    return len(k_values) > 1


def _fmt_metric(value: Any, *, width: int = 7, prec: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{'n/a':>{width}}"
    if not math.isfinite(v):
        return f"{'n/a':>{width}}"
    return f"{v:{width}.{prec}f}"


def _fmt_metric_plain(value: Any, *, prec: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{prec}f}"


def _row_k_prefix(r: dict[str, Any], multi_k: bool) -> str:
    if not multi_k:
        return ""
    return f"K={int(r.get('k', 0))} "


@torch.inference_mode()
def eval_layout_vs_real_anchors(
    engine: Any,
    device: torch.device,
    *,
    data_dir: Path,
    experiment_cfg: dict[str, Any],
    experiment_dir: str,
    k_value: int,
    anchor_mhz_list: list[float],
    num_samples: int,
    seed: int,
    fg_thr: float,
    mhz_tol: float = 2.0,
    val_ld: DataLoader | None = None,
) -> list[dict[str, Any]]:
    """Layout path vs real val heatmaps at training-anchor MHz only."""
    sm = _spatial_metrics(experiment_dir)
    model = engine.model
    model.eval()
    clip_lo = engine.hm_z_clip[0] if getattr(engine, "hm_z_clip", None) else None
    clip_hi = engine.hm_z_clip[1] if getattr(engine, "hm_z_clip", None) else None
    rows: list[dict[str, Any]] = []

    if val_ld is None:
        val_ld = get_sweep_val_loader(
            data_dir=data_dir,
            experiment_cfg=experiment_cfg,
            device=device,
        )

    for anchor in anchor_mhz_list:
        try:
            occ, imp, hm, k_t = load_val_anchor_samples(
                data_dir=data_dir,
                experiment_cfg=experiment_cfg,
                k_value=k_value,
                anchor_mhz=anchor,
                num_samples=num_samples,
                seed=seed,
                device=device,
                mhz_tol=mhz_tol,
                val_ld=val_ld,
            )
        except ValueError:
            continue
        if hm.dim() == 3:
            hm = hm.unsqueeze(1)
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        elif imp.dim() == 3:
            imp = imp[:, :1]

        pi_t = pi_freq_norm_for_model(anchor, occ.shape[0], unit="mhz", device=device)
        hm_enc = hm.masked_fill(hm < fg_thr, 0.0)
        z_lay = model.encode_layout_latent(occ, imp, k_t, pi_t)
        rh, _, _ = model.decode(z_lay, k_t, pi_t, occupancy=occ)

        pr = sm.pearson_fg(rh, hm, fg_thr)
        mse = _fg_mse(rh, hm, fg_thr)
        pl = sm.peak_loc_err(rh, hm, fg_thr)

        if hasattr(engine, "denorm_heatmap_physical"):
            rh_phys = engine.denorm_heatmap_physical(rh, mhz=anchor)
            hm_phys = engine.denorm_heatmap_physical(hm, mhz=anchor)
        else:
            hm_stats = getattr(engine, "hm_stats", None)
            rh_phys = heatmap_z_to_physical(
                rh,
                engine.hm_log_mean,
                engine.hm_log_std,
                hm_stats=hm_stats,
                mhz=anchor,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
            )
            hm_phys = heatmap_z_to_physical(
                hm,
                engine.hm_log_mean,
                engine.hm_log_std,
                hm_stats=hm_stats,
                mhz=anchor,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
            )
        mask = engine.binary_mask
        real_maxes, gen_maxes, maes = [], [], []
        for j in range(occ.shape[0]):
            r_fg = hm_phys[j, 0].cpu().numpy()[mask]
            g_fg = rh_phys[j, 0].cpu().numpy()[mask]
            real_maxes.append(float(r_fg.max()) if r_fg.size else 0.0)
            gen_maxes.append(float(g_fg.max()) if g_fg.size else 0.0)
            maes.append(float(np.mean(np.abs(r_fg - g_fg))) if r_fg.size else 0.0)

        gen_max_mean = float(np.mean(gen_maxes))
        real_max_mean = float(np.mean(real_maxes))
        rows.append({
            "mhz": float(anchor),
            "n": int(occ.shape[0]),
            "pearson_r": float(pr.mean().cpu()),
            "fg_mse": float(mse.mean().cpu()),
            "peak_loc_err": float(pl.mean().cpu()),
            "mae_phys": float(np.mean(maes)),
            "real_max": real_max_mean,
            "gen_max": gen_max_mean,
            "max_ratio": gen_max_mean / max(real_max_mean, 1e-6),
        })
    return rows


def _flags(
    gen_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    engine: Any,
    *,
    multi_k: bool = False,
) -> list[str]:
    flags: list[str] = []
    for r in gen_rows:
        mhz = float(r["mhz"])
        gmax = float(r["max"])
        ceil = clip_ceiling_ohm(engine, mhz=mhz)
        if ceil is not None and _is_near_clip(gmax, ceil):
            flags.append(
                f"[WARN] {_row_k_prefix(r, multi_k)}{mhz:.0f} MHz: gen_max={gmax:.2f}Ω "
                f"(≥90% per-MHz clip ceiling {ceil:.2f}Ω) — magnitude saturation"
            )
    for r in anchor_rows:
        mhz = r["mhz"]
        pr = float(r["pearson_r"])
        ratio = float(r["max_ratio"])
        if pr < 0.75:
            flags.append(
                f"[WARN] {_row_k_prefix(r, multi_k)}{mhz:.0f} MHz anchor: "
                f"pearson_r={pr:.3f} (<0.75) — weak spatial pattern"
            )
        if ratio > 1.5:
            flags.append(
                f"[WARN] {_row_k_prefix(r, multi_k)}{mhz:.0f} MHz anchor: "
                f"gen_max/real_max={ratio:.2f} — magnitude overshoot"
            )
        if pr >= 0.85 and ratio <= 1.3:
            flags.append(
                f"[OK]   {_row_k_prefix(r, multi_k)}{mhz:.0f} MHz anchor: "
                f"pearson_r={pr:.3f}, max_ratio={ratio:.2f}"
            )
    if not flags:
        flags.append("[INFO] No auto flags — review tables below.")
    return flags


def format_agent_copy_block(
    *,
    experiment_dir: str,
    checkpoint_path: str,
    qc_cfg: SweepQCConfig,
    gen_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    flags: list[str],
    engine: Any,
    k_values: list[int],
) -> str:
    """Single block to paste into an agent chat."""
    multi_k = _multi_k(k_values)
    ref_ceil = clip_ceiling_ohm(engine, mhz=qc_cfg.pi_ref_mhz)
    lines = [
        "=== SWEEP QC (copy to agent) ===",
        f"experiment: {experiment_dir}",
        f"checkpoint: {checkpoint_path}",
        f"mode: {qc_cfg.inference_mode}  layout_source: {qc_cfg.layout_source}",
        f"K: {_k_label(k_values)}  samples: {qc_cfg.num_samples}  pi_ref: {qc_cfg.pi_ref_mhz} MHz",
        (
            f"z_clip_ceiling_ohm@{qc_cfg.pi_ref_mhz:.0f}MHz: {ref_ceil:.2f}"
            if ref_ceil is not None
            else "z_clip_ceiling_ohm: n/a"
        ),
        "",
        "Per-MHz generation (layout decode, no GT):",
    ]
    if multi_k:
        lines.append("  K | MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip")
    else:
        lines.append("MHz | anchor | gen_max | mean_fg | p95 | ceil_ohm | near_clip")
    anchor_set = {float(a) for a in ANCHOR_MHZ}
    for r in gen_rows:
        mhz = float(r["mhz"])
        ceil = clip_ceiling_ohm(engine, mhz=mhz)
        near = "yes" if ceil is not None and _is_near_clip(float(r["max"]), ceil) else "no"
        is_anchor = "yes" if any(abs(mhz - a) < 1.0 for a in anchor_set) else "no"
        ceil_s = f"{ceil:.2f}" if ceil is not None else "n/a"
        if multi_k:
            lines.append(
                f"{int(r.get('k', 0)):3d} | {mhz:6.0f} | {is_anchor:6s} | "
                f"{_fmt_metric(r['max'])} | {_fmt_metric(r['mean_fg'])} | {_fmt_metric(r['p95'])} | "
                f"{ceil_s:>8} | {near}"
            )
        else:
            lines.append(
                f"{mhz:6.0f} | {is_anchor:6s} | {_fmt_metric(r['max'])} | "
                f"{_fmt_metric(r['mean_fg'])} | {_fmt_metric(r['p95'])} | {ceil_s:>8} | {near}"
            )

    if anchor_rows:
        lines += [
            "",
            "Layout vs real val heatmap (training anchors only):",
        ]
        if multi_k:
            lines.append("  K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio")
        else:
            lines.append("MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio")
        for r in anchor_rows:
            if multi_k:
                lines.append(
                    f"{int(r.get('k', 0)):3d} | {r['mhz']:4.0f} | {r['n']} | {r['pearson_r']:.3f} | "
                    f"{r['fg_mse']:.4f} | {r['mae_phys']:.3f} | {r['real_max']:.2f} | "
                    f"{r['gen_max']:.2f} | {r['max_ratio']:.2f}"
                )
            else:
                lines.append(
                    f"{r['mhz']:4.0f} | {r['n']} | {r['pearson_r']:.3f} | {r['fg_mse']:.4f} | "
                    f"{r['mae_phys']:.3f} | {r['real_max']:.2f} | {r['gen_max']:.2f} | {r['max_ratio']:.2f}"
                )
    else:
        lines += ["", "Layout vs real: skipped (not layout_qc or no anchor val samples)."]

    lines += ["", "Flags:"]
    lines.extend(flags)
    lines.append("=== END SWEEP QC ===")
    return "\n".join(lines)


def write_sweep_qc_report(
    out_root: Path,
    *,
    experiment_dir: str,
    checkpoint_path: str,
    qc_cfg: SweepQCConfig,
    gen_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    engine: Any,
    k_values: list[int],
    print_copy_block: bool = True,
) -> tuple[Path, Path]:
    """Write ``sweep_qc_report.md``, ``sweep_qc_metrics.json``, optionally print copy block."""
    out_root = Path(out_root)
    gen_rows = _sort_gen_rows(gen_rows)
    anchor_rows = sorted(anchor_rows, key=lambda r: (int(r.get("k", 0)), float(r["mhz"])))
    multi_k = _multi_k(k_values)
    flags = _flags(gen_rows, anchor_rows, engine, multi_k=multi_k)
    copy_block = format_agent_copy_block(
        experiment_dir=experiment_dir,
        checkpoint_path=checkpoint_path,
        qc_cfg=qc_cfg,
        gen_rows=gen_rows,
        anchor_rows=anchor_rows,
        flags=flags,
        engine=engine,
        k_values=k_values,
    )

    per_mhz_ceilings = {
        f"K{int(r.get('k', 0))}@{float(r['mhz']):.0f}"
        if multi_k
        else str(r["mhz"]): clip_ceiling_ohm(engine, mhz=float(r["mhz"]))
        for r in gen_rows
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": experiment_dir,
        "checkpoint_path": checkpoint_path,
        "qc_config": {
            "inference_mode": qc_cfg.inference_mode,
            "layout_source": qc_cfg.layout_source,
            "k_values": k_values,
            "pi_ref_mhz": qc_cfg.pi_ref_mhz,
        },
        "per_mhz_clip_ceiling_ohm": per_mhz_ceilings,
        "per_mhz_generation": gen_rows,
        "anchor_layout_vs_real": anchor_rows,
        "flags": flags,
        "agent_copy_block": copy_block,
    }

    json_path = out_root / "sweep_qc_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    ref_ceil = clip_ceiling_ohm(engine, mhz=qc_cfg.pi_ref_mhz)
    md_lines = [
        "# Sweep QC Report",
        "",
        f"- **Experiment:** `{experiment_dir}`",
        f"- **Checkpoint:** `{checkpoint_path}`",
        f"- **Mode:** `{qc_cfg.inference_mode}` / `{qc_cfg.layout_source}`",
        f"- **K:** {_k_label(k_values)} | **PI ref:** {qc_cfg.pi_ref_mhz} MHz",
        (
            f"- **Clip ceiling @ {qc_cfg.pi_ref_mhz:.0f} MHz (approx):** {ref_ceil:.2f} Ω"
            if ref_ceil is not None
            else "- **Clip ceiling:** n/a"
        ),
        "",
        "## Per-MHz generation",
        "",
    ]
    if multi_k:
        md_lines += [
            "| K | MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |",
            "|---|-----|------------------|-----------|---------|-----|------------|",
        ]
    else:
        md_lines += [
            "| MHz | training anchor? | gen_max Ω | mean_fg | p95 | near clip? |",
            "|-----|------------------|-----------|---------|-----|------------|",
        ]
    anchor_set = {float(a) for a in ANCHOR_MHZ}
    for r in gen_rows:
        mhz = float(r["mhz"])
        is_anchor = any(abs(mhz - a) < 1.0 for a in anchor_set)
        ceil = clip_ceiling_ohm(engine, mhz=mhz)
        near = _is_near_clip(float(r["max"]), ceil)
        if multi_k:
            md_lines.append(
                f"| {int(r.get('k', 0))} | {mhz:.0f} | {'yes' if is_anchor else 'no'} | "
                f"{_fmt_metric_plain(r['max'])} | {_fmt_metric_plain(r['mean_fg'])} | "
                f"{_fmt_metric_plain(r['p95'])} | {'yes' if near else 'no'} |"
            )
        else:
            md_lines.append(
                f"| {mhz:.0f} | {'yes' if is_anchor else 'no'} | "
                f"{_fmt_metric_plain(r['max'])} | {_fmt_metric_plain(r['mean_fg'])} | "
                f"{_fmt_metric_plain(r['p95'])} | {'yes' if near else 'no'} |"
            )

    if anchor_rows:
        md_lines += ["", "## Layout vs real (val, training anchors)", ""]
        if multi_k:
            md_lines += [
                "| K | MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |",
                "|---|-----|---|-----------|--------|----------|----------|---------|-----------|",
            ]
        else:
            md_lines += [
                "| MHz | n | pearson_r | fg_mse | mae_phys | real_max | gen_max | max_ratio |",
                "|-----|---|-----------|--------|----------|----------|---------|-----------|",
            ]
        for r in anchor_rows:
            if multi_k:
                md_lines.append(
                    f"| {int(r.get('k', 0))} | {r['mhz']:.0f} | {r['n']} | {r['pearson_r']:.3f} | "
                    f"{r['fg_mse']:.4f} | {r['mae_phys']:.3f} | {r['real_max']:.2f} | "
                    f"{r['gen_max']:.2f} | {r['max_ratio']:.2f} |"
                )
            else:
                md_lines.append(
                    f"| {r['mhz']:.0f} | {r['n']} | {r['pearson_r']:.3f} | {r['fg_mse']:.4f} | "
                    f"{r['mae_phys']:.3f} | {r['real_max']:.2f} | {r['gen_max']:.2f} | {r['max_ratio']:.2f} |"
                )

    md_lines += ["", "## Flags", ""]
    for f in flags:
        md_lines.append(f"- {f}")

    md_lines += [
        "",
        "## Copy block for agent",
        "",
        "```",
        copy_block,
        "```",
    ]

    md_path = out_root / "sweep_qc_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if print_copy_block:
        print("\n" + copy_block)
    print(f"  QC → {md_path.name}, {json_path.name}")
    return md_path, json_path


def _anchors_in_sweep(sweep_mhz: list[float]) -> list[float]:
    return [
        float(a)
        for a in ANCHOR_MHZ
        if any(abs(float(a) - float(m)) < 2.0 for m in sweep_mhz)
    ]


def _collect_anchor_rows(
    engine: Any,
    device: torch.device,
    *,
    data_dir: Path,
    experiment_cfg: dict[str, Any],
    experiment_dir: str,
    k_values: list[int],
    anchor_mhz_list: list[float],
    qc_cfg: SweepQCConfig,
    val_ld: DataLoader | None,
) -> list[dict[str, Any]]:
    """Anchor GT eval for each K — one progress line per K, no per-K report spam."""
    if val_ld is None:
        val_ld = get_sweep_val_loader(
            data_dir=data_dir,
            experiment_cfg=experiment_cfg,
            device=device,
        )
    fg_thr = float(engine.fg_threshold()) if hasattr(engine, "fg_threshold") else -2.47
    anchor_rows: list[dict[str, Any]] = []
    print(
        f"  QC anchor eval: {len(anchor_mhz_list)} anchor(s) × {len(k_values)} K "
        f"({', '.join(f'{a:.0f}' for a in anchor_mhz_list)} MHz)…"
    )
    for k in k_values:
        rows = eval_layout_vs_real_anchors(
            engine,
            device,
            data_dir=data_dir,
            experiment_cfg=experiment_cfg,
            experiment_dir=experiment_dir,
            k_value=k,
            anchor_mhz_list=anchor_mhz_list,
            num_samples=qc_cfg.num_samples,
            seed=qc_cfg.seed,
            fg_thr=fg_thr,
            val_ld=val_ld,
        )
        for r in rows:
            r["k"] = k
        anchor_rows.extend(rows)
        if rows:
            summary = ", ".join(f"{r['mhz']:.0f}MHz pr={r['pearson_r']:.2f}" for r in rows)
            print(f"    K={k:2d}: {summary}")
        else:
            print(f"    K={k:2d}: no val anchor samples")
    return anchor_rows


def run_sweep_qc_eval(
    engine: Any,
    device: torch.device,
    *,
    out_root: Path,
    experiment_dir: str,
    checkpoint_path: str,
    data_dir: Path,
    experiment_cfg: dict[str, Any],
    qc_cfg: SweepQCConfig,
    gen_rows: list[dict[str, Any]],
    sweep_mhz: list[float],
    k_values: list[int] | None = None,
    val_ld: DataLoader | None = None,
) -> tuple[Path, Path] | None:
    """Run anchor GT eval (layout_qc only) and write one combined agent report."""
    kv = k_values or [int(qc_cfg.k_value)]
    anchor_rows: list[dict[str, Any]] = []
    if qc_cfg.inference_mode == "layout_qc" and qc_cfg.layout_source == "val":
        anchors_in_sweep = _anchors_in_sweep(sweep_mhz)
        if anchors_in_sweep:
            anchor_rows = _collect_anchor_rows(
                engine,
                device,
                data_dir=data_dir,
                experiment_cfg=experiment_cfg,
                experiment_dir=experiment_dir,
                k_values=kv,
                anchor_mhz_list=anchors_in_sweep,
                qc_cfg=qc_cfg,
                val_ld=val_ld,
            )

    return write_sweep_qc_report(
        out_root,
        experiment_dir=experiment_dir,
        checkpoint_path=checkpoint_path,
        qc_cfg=qc_cfg,
        gen_rows=gen_rows,
        anchor_rows=anchor_rows,
        engine=engine,
        k_values=kv,
    )
