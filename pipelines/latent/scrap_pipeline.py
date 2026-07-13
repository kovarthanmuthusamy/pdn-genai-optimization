"""Latent scrap pipeline — shared export, PEB, and impedance-compare helpers.

Run:
    Not run directly — imported by latent_run_export_peb.py and"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

EXPORT_SUBDIR = "exported_samples"
REPORT_PLOTS_SUBDIR = "report_plots"
PEB_NAME = "latent_run.peb"
CMP_IMPEDANCE = "generated_vs_real_impedance_profile.png"
CMP_GEN_TARGET = "generated_vs_target_impedance.png"
PI_FREQ_MHZ_DEFAULT = 100
CMP_HEATMAP = "generated_vs_real_heatmap.png"
CMP_OCCUPANCY = "generated_occupancy.png"

# Comparison PNGs under exported_samples/K{n}/ (cleared before new PEB)
_COMPARE_PLOT_NAMES = (
    CMP_IMPEDANCE,
    CMP_GEN_TARGET,
    CMP_HEATMAP,
    CMP_OCCUPANCY,
)


def export_root(run_dir: Path) -> Path:
    return run_dir / EXPORT_SUBDIR


def active_decaps_summary(run_dir: Path, k: int) -> str | None:
    """Short line listing decap components turned on for this K result."""
    metrics_p = run_dir / f"K{k}" / "best_metrics.json"
    if metrics_p.exists():
        try:
            data = json.loads(metrics_p.read_text(encoding="utf-8"))
            names = data.get("active_components")
            if isinstance(names, list) and names:
                n = int(data.get("n_active", len(names)))
                return f"**{n} decaps on:** {', '.join(str(x) for x in names)}"
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    for occ_p in (
        export_root(run_dir) / f"K{k}" / "occupancy.npy",
        run_dir / f"K{k}" / "best_occupancy_topk.npy",
        export_root(run_dir) / f"K{k}" / "data_sample_0" / "occupancy_map.npy",
    ):
        if not occ_p.is_file():
            continue
        vec = np.asarray(np.load(occ_p)).reshape(-1)[:52]
        on = [f"C{i + 1}" for i, v in enumerate(vec) if int(v) == 1]
        if on:
            return f"**{len(on)} decaps on:** {', '.join(on)}"
    return None


def collect_existing_compare_pngs(run_dir: Path, k_values: list[int]) -> tuple[dict[int, Path], dict[int, Path]]:
    """Find existing comparison PNGs under exported_samples/K{n}/ (for --report-only)."""
    root = export_root(run_dir)
    impedance: dict[int, Path] = {}
    heatmap: dict[int, Path] = {}
    for k in sorted(k_values):
        k_dir = root / f"K{k}"
        if not k_dir.is_dir():
            continue
        for name in (CMP_IMPEDANCE, CMP_GEN_TARGET):
            p = k_dir / name
            if p.is_file():
                impedance[k] = p
                break
        hm = k_dir / CMP_HEATMAP
        if hm.is_file():
            heatmap[k] = hm
    return impedance, heatmap


def discover_latent_k_dirs(run_dir: Path) -> dict[int, Path]:
    """Map K → latent solution folder (run_dir/K##/ with best_latent.npy)."""
    out: dict[int, Path] = {}
    for p in sorted(run_dir.iterdir()):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"K(\d+)", p.name)
        if not m or not (p / "best_latent.npy").exists():
            continue
        out[int(m.group(1))] = p
    return out


def peb_pi_layout(peb_path: Path) -> tuple[int, str | None]:
    """Return (pis_per_sample, pi_output_kind) from PEB content.

    pi_output_kind is ``"heatmap"`` | ``"impedance"`` when pps==1, else None (both types).
    """
    if not peb_path.exists():
        return 1, "impedance"
    text = peb_path.read_text(encoding="utf-8", errors="ignore")
    has_dist = "EditPIDistribution" in text or "PI-Distribution" in text
    has_spec = "CreatePISpectrum" in text
    if has_spec and not has_dist:
        return 1, "impedance"
    if has_dist and not has_spec:
        return 1, "heatmap"
    if has_dist and has_spec:
        return 2, None
    return 1, "heatmap"


def peb_pis_per_sample(peb_path: Path) -> int:
    """Backward-compatible: PI count per sample inferred from PEB."""
    return peb_pi_layout(peb_path)[0]


def load_pi_freq_mhz(run_dir: Path) -> float:
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("pi_freq_mhz", "PI_freq_mhz"):
            if key in cfg:
                return float(cfg[key])
    import os

    return float(os.getenv("LATENT_OPT_PLOT_PI_FREQ_MHZ", str(PI_FREQ_MHZ_DEFAULT)))


def export_k_from_latent(latent_k_dir: Path, out_k_dir: Path) -> np.ndarray:
    """Write run_all_k-compatible files (occupancy + impedance log only)."""
    out_k_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_k_dir / "data_sample_0"
    sample_dir.mkdir(parents=True, exist_ok=True)

    topk_p = latent_k_dir / "best_occupancy_topk.npy"
    if not topk_p.exists():
        raise FileNotFoundError(topk_p)
    occ = np.load(topk_p).astype(np.int8).reshape(-1)

    np.save(out_k_dir / "occupancy.npy", occ.reshape(1, 52))
    np.save(sample_dir / "occupancy_map.npy", occ)

    imp_log_p = latent_k_dir / "best_impedance_log.npy"
    if imp_log_p.exists():
        np.save(sample_dir / "impedance_profile.npy", np.load(imp_log_p))

    return occ


def move_pi_from_emc(
    run_dir: Path,
    k_values: list[int],
    *,
    source_emc_dir: str,
    peb_path: Path | None = None,
) -> list[Path]:
    """Move ECADStar PI-* outputs into exported_samples/K{n}/Real/ (scrap/move_pi_to_real)."""
    import scrap.orchestration.move_pi_to_real as mpi

    if not k_values:
        return []
    base = export_root(run_dir).resolve()
    peb = peb_path or (run_dir / PEB_NAME)
    pps, pi_kind = peb_pi_layout(peb)
    mpi.SEARCH_RECURSIVE = True
    kind_note = f", rename→{pi_kind}" if pi_kind else " (distribution+spectrum)"
    print(f"  source .emc: {source_emc_dir}")
    print(f"  target base: {base}")
    print(f"  PEB: {peb.name}  →  {pps} PI output(s)/sample{kind_note}")
    resolved = mpi.resolve_pi_source_dir(source_emc_dir, recursive=True)
    print(f"  PI source dir: {resolved}")
    dests = mpi.move_pi_outputs_for_k_list(
        k_values,
        source_emc_dir=str(resolved),
        base_generated_dir=base,
        pis_per_sample=pps,
        pi_output_kind=pi_kind,
    )
    if pi_kind == "impedance" or pps == 1:
        for k, dest in zip(sorted(k_values), dests):
            n = mpi._infer_num_samples(base / f"K{k}")
            mpi.ensure_real_impedance_renamed(dest, n)
    return dests


def export_all(run_dir: Path) -> list[int]:
    """Export every K with a latent solution. Returns sorted K list."""
    k_dirs = discover_latent_k_dirs(run_dir)
    if not k_dirs:
        raise FileNotFoundError(f"No K##/best_latent.npy under {run_dir}")

    root = export_root(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    for k_val, latent_k in k_dirs.items():
        out_k = root / f"K{k_val}"
        export_k_from_latent(latent_k, out_k)
        print(f"  exported K{k_val:02d} → {out_k.relative_to(run_dir)}")
    return sorted(k_dirs.keys())


def clear_exported_real_and_plots(run_dir: Path, k_values: list[int] | None = None) -> None:
    """Remove stale ECADStar Real/ data and comparison PNGs under exported_samples/K{n}/."""
    root = export_root(run_dir)
    if not root.is_dir():
        return

    if k_values is None:
        k_dirs = sorted(
            (p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"K\d+", p.name)),
            key=lambda p: int(p.name[1:]),
        )
    else:
        k_dirs = [root / f"K{k}" for k in sorted(k_values)]

    cleared_real = 0
    cleared_plots = 0
    for k_dir in k_dirs:
        if not k_dir.is_dir():
            continue
        real_dir = k_dir / "Real"
        if real_dir.is_dir():
            shutil.rmtree(real_dir)
            cleared_real += 1
        for name in _COMPARE_PLOT_NAMES:
            p = k_dir / name
            if p.is_file():
                p.unlink()
                cleared_plots += 1
        for p in k_dir.glob("generated_vs_*__*.png"):
            if p.is_file():
                p.unlink()
                cleared_plots += 1

    if cleared_real or cleared_plots:
        print(
            f"  cleared stale CAD/plots under {root.name}/ "
            f"({cleared_real} Real/ folder(s), {cleared_plots} plot file(s))"
        )


def generate_peb_for_run(
    run_dir: Path,
    k_values: list[int],
    *,
    pi_freq_mhz: float,
    powerbus: str = "Power_GND",
    components: str = "IC1_Port1",
) -> Path:
    from scrap.generation.generate_peb import generate_peb

    clear_exported_real_and_plots(run_dir, k_values)

    root = export_root(run_dir)
    rows: list[np.ndarray] = []
    for k in sorted(k_values):
        occ_path = root / f"K{k}" / "occupancy.npy"
        if not occ_path.exists():
            continue
        occ = np.load(occ_path)
        rows.append(occ.reshape(-1) if occ.ndim == 1 else occ[0])

    if not rows:
        raise FileNotFoundError(f"No exported occupancy under {root}")

    stack = np.stack(rows, axis=0).astype(np.int8)
    peb_path = run_dir / PEB_NAME
    generate_peb(
        occupancy=stack,
        output_path=str(peb_path),
        powerbus=powerbus,
        freq=f"{int(pi_freq_mhz)}e6",
        components=components,
        include_distribution=False,
        include_spectrum=True,
    )
    return peb_path


def _real_has_impedance(real_dir: Path) -> bool:
    if not real_dir.is_dir():
        return False
    if any(real_dir.glob("Imp_Real*")):
        return True
    for p in real_dir.iterdir():
        if re.fullmatch(r"PI-\d+(?:\..+)?", p.name) and p.is_dir():
            if any(p.rglob("*.csv")):
                return True
    return False


def _sample_has_gen_heatmap(sample_dir: Path) -> bool:
    return (sample_dir / "heatmap_physical.npy").exists()


def run_compare_k_like_scrap(
    k: int,
    *,
    repo_root: Path,
    base_generated_dir: Path,
    pi_freq_mhz: float,
) -> dict[str, Path]:
    """Run scrap/comparison/compare._run_single_k (Target + Real + Generated)."""
    import matplotlib

    matplotlib.use("Agg")
    import scrap.comparison.compare as cmp

    k_dir = (repo_root / base_generated_dir / f"K{k}").resolve()
    run_heatmap = any(
        _sample_has_gen_heatmap(k_dir / f"data_sample_{i}")
        for i in range(cmp._infer_num_samples(k_dir))
    )
    saved: dict[str, object] = {}
    for key in ("BASE_GENERATED_DIR", "FREQ_LABEL", "RUN_HEATMAP", "RUN_IMPEDANCE", "RUN_OCCUPANCY"):
        if hasattr(cmp, key):
            saved[key] = getattr(cmp, key)
    try:
        rel = base_generated_dir
        if rel.is_absolute():
            try:
                rel = rel.relative_to(repo_root)
            except ValueError:
                rel = rel.as_posix()
        cmp.BASE_GENERATED_DIR = rel.as_posix() if hasattr(rel, "as_posix") else str(rel)
        cmp.FREQ_LABEL = f"{int(pi_freq_mhz)} MHz"
        cmp.RUN_HEATMAP = run_heatmap
        cmp.RUN_IMPEDANCE = True
        cmp.RUN_OCCUPANCY = False
        cmp._run_single_k(k, repo_root=repo_root)
    finally:
        for key, val in saved.items():
            setattr(cmp, key, val)

    out: dict[str, Path] = {}
    if (k_dir / cmp.IMPEDANCE_OUT_NAME).exists():
        out["impedance"] = k_dir / cmp.IMPEDANCE_OUT_NAME
    if run_heatmap and (k_dir / cmp.HEATMAP_OUT_NAME).exists():
        out["heatmap"] = k_dir / cmp.HEATMAP_OUT_NAME
    return out


def _load_gen_impedance_row(cmp: Any, k_dir: Path, k: int, i: int) -> dict[str, Any]:
    sample_dir = k_dir / f"data_sample_{i}"
    gen_blended_log = cmp._load_generated_impedance_log(sample_dir / "impedance_profile.npy")
    gen_blended_ohm = np.exp(gen_blended_log)
    label = f"K{k}/sample_{i}"
    return {"label": label, "gen_blended_ohm": gen_blended_ohm}


def run_impedance_compare_k(
    k: int,
    *,
    repo_root: Path,
    base_generated_dir: Path,
    pi_freq_mhz: float,
) -> Path | None:
    """Impedance Target + Real + Generated (same as scrap/comparison/compare.py)."""
    plots = run_compare_k_like_scrap(
        k,
        repo_root=repo_root,
        base_generated_dir=base_generated_dir,
        pi_freq_mhz=pi_freq_mhz,
    )
    return plots.get("impedance")


def run_impedance_gen_vs_target_k(
    k: int,
    *,
    repo_root: Path,
    base_generated_dir: Path,
    pi_freq_mhz: float,
) -> Path:
    """Plot generated impedance vs target (no Real/ data required)."""
    import matplotlib.pyplot as plt
    import scrap.comparison.compare as cmp

    k_dir = (repo_root / base_generated_dir / f"K{k}").resolve()
    num_samples = cmp._infer_num_samples(k_dir)
    frequency = np.load(repo_root / cmp.FREQUENCY_PATH).squeeze()
    target_impedance = np.load(repo_root / cmp.TARGET_IMPEDANCE_PATH).squeeze()

    comparisons = [_load_gen_impedance_row(cmp, k_dir, k, i) for i in range(num_samples)]
    n = len(comparisons)
    fig, axes = plt.subplots(n, 1, figsize=(11, 5.5 * n), layout="constrained", squeeze=False)
    flat = list(np.atleast_1d(axes).flat)

    for ax, item in zip(flat, comparisons):
        ax.loglog(frequency, target_impedance, "--", lw=2.0, label="Target", color="red")
        ax.loglog(frequency, item["gen_blended_ohm"], "-", lw=2.0, label="Generated", color="#1565C0")
        ax.set_ylim(1e-3, 1e2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Impedance (Ohm)")
        ax.set_title(item["label"])
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(f"K={k} — Generated vs target [{int(pi_freq_mhz)} MHz]")
    out_path = k_dir / CMP_GEN_TARGET
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved gen vs target: {out_path}")
    return out_path


def run_impedance_compare_all(
    run_dir: Path,
    repo_root: Path,
    k_values: list[int],
    *,
    pi_freq_mhz: float,
) -> tuple[dict[int, Path], dict[int, Path]]:
    base = export_root(run_dir)
    rel = base.relative_to(repo_root)
    out: dict[int, Path] = {}
    heatmap_pngs: dict[int, Path] = {}

    if not any(
        _real_has_impedance((repo_root / rel / f"K{k}" / "Real").resolve())
        for k in k_values
    ):
        print("  No Imp_Real* under any K/Real/ yet.")
        print("  Run ECADStar on latent_run.peb, set LATENT_SOURCE_EMC_DIR, re-run this script.")
        print("  (Plots will use generated vs target only until Real/ has Imp_Real*.)\n")

    for k in sorted(k_values):
        k_dir = (repo_root / rel / f"K{k}").resolve()
        real_dir = k_dir / "Real"
        try:
            if _real_has_impedance(real_dir):
                plots = run_compare_k_like_scrap(
                    k,
                    repo_root=repo_root,
                    base_generated_dir=rel,
                    pi_freq_mhz=pi_freq_mhz,
                )
                p = plots.get("impedance")
                if hm := plots.get("heatmap"):
                    heatmap_pngs[k] = hm
                    print(f"  heatmap K{k:02d} → {hm.relative_to(run_dir)}")
            else:
                if real_dir.is_dir() and any(real_dir.iterdir()):
                    print(
                        f"  K{k:02d}: Real/ exists but no Imp_Real* "
                        f"(heatmap-only PEB or move_pi rename issue) — gen vs target"
                    )
                p = run_impedance_gen_vs_target_k(
                    k, repo_root=repo_root, base_generated_dir=rel, pi_freq_mhz=pi_freq_mhz,
                )
            if p and p.exists():
                out[k] = p
                print(f"  compare K{k:02d} → {p.relative_to(run_dir)}")
        except SystemExit as e:
            print(f"  compare K{k:02d} skipped: {e}")
        except Exception as e:
            print(f"  compare K{k:02d} failed: {e!r}")

    return out, heatmap_pngs


def _prepare_report_plot(run_dir: Path, src: Path, dest_name: str) -> Path:
    """Copy plot into run_dir/report_plots/ and return the destination path."""
    plots_dir = run_dir / REPORT_PLOTS_SUBDIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    dest = plots_dir / dest_name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def _png_data_uri(path: Path) -> str:
    payload = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _markdown_image(alt: str, img_path: Path, run_dir: Path, *, embed: bool) -> str:
    """Markdown image line that works in Cursor/VS Code preview (incl. WSL UNC workspaces)."""
    if embed and img_path.is_file():
        return f"![{alt}]({_png_data_uri(img_path)})"
    try:
        rel = img_path.relative_to(run_dir).as_posix()
    except ValueError:
        rel = img_path.as_posix()
    if not rel.startswith("."):
        rel = f"./{rel}"
    return f"![{alt}]({rel})"


def _write_html_report(
    run_dir: Path,
    *,
    title: str,
    sections: list[tuple[str, Path, str | None]],
) -> Path:
    """Standalone HTML report with embedded PNGs (always renders in a browser)."""
    html_path = run_dir / "run_report.html"
    body_parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;}",
        "h2{margin-top:2.5rem;} img{max-width:100%;height:auto;border:1px solid #ddd;}",
        ".note{color:#444;margin:0.25rem 0 0.75rem;}",
        "</style></head><body>",
        f"<h1>{title}</h1>",
    ]
    for heading, img_path, note in sections:
        if not img_path.is_file():
            continue
        body_parts.append(f"<h2>{heading}</h2>")
        if note:
            body_parts.append(f'<p class="note">{note}</p>')
        body_parts.append(
            f'<p><img src="{_png_data_uri(img_path)}" alt="{heading}"/></p>',
        )
    body_parts.append("</body></html>")
    html_path.write_text("\n".join(body_parts), encoding="utf-8")
    return html_path


def write_compare_report(
    run_dir: Path,
    *,
    peb_path: Path | None,
    compare_pngs: dict[int, Path],
    heatmap_pngs: dict[int, Path] | None = None,
    include_config: bool = True,
    embed_images: bool | None = None,
    write_html: bool = True,
) -> Path:
    """run_report.md (+ run_report.html) with impedance compare figures."""
    if embed_images is None:
        embed_images = os.getenv("LATENT_REPORT_EMBED_IMAGES", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )

    report_path = run_dir / "run_report.md"
    plots_dir = run_dir / REPORT_PLOTS_SUBDIR
    if plots_dir.is_dir():
        shutil.rmtree(plots_dir)
    parts: list[str] = []
    html_sections: list[tuple[str, Path, str | None]] = []

    if include_config:
        from pipelines.latent.generate_run_report import build_report

        parts.append(build_report(run_dir))

    parts.append("")
    has_real = any(p.name == CMP_IMPEDANCE for p in compare_pngs.values())
    title = "generated vs real" if has_real else "generated vs target (no ECADStar Real/ yet)"
    parts.append(f"## Impedance comparison ({title})")
    parts.append("")
    parts.append(
        "> Plots are embedded for Markdown preview (WSL). "
        "If images still do not show, open **`run_report.html`** in this folder."
    )
    parts.append("")
    if peb_path and peb_path.exists():
        try:
            rel_peb = peb_path.relative_to(run_dir)
        except ValueError:
            rel_peb = peb_path
        parts.append(f"PEB: `{rel_peb.as_posix()}`  ")
        parts.append(f"Exported: `{EXPORT_SUBDIR}/`  ")
        parts.append("")

    if not compare_pngs:
        parts.append(
            "No impedance plots produced. Run ECADStar on `latent_run.peb`, set "
            "`LATENT_SOURCE_EMC_DIR` to the folder with PI-1, PI-2, …, then re-run "
            "`latent_run_compare_report.py` for gen-vs-real plots."
        )
        parts.append("")
    else:
        for k_val in sorted(compare_pngs.keys()):
            p = compare_pngs[k_val]
            if not p.is_file():
                continue
            dest = _prepare_report_plot(run_dir, p, f"K{k_val:02d}_impedance.png")
            heading = f"K = {k_val:02d}"
            decap_line = active_decaps_summary(run_dir, k_val)
            parts.append(f"### {heading}")
            parts.append("")
            if decap_line:
                parts.append(decap_line)
                parts.append("")
            parts.append(
                _markdown_image(
                    f"K{k_val:02d} impedance compare", dest, run_dir, embed=embed_images,
                )
            )
            parts.append("")
            note = None
            if decap_line:
                note = decap_line.replace("**", "")
            html_sections.append((heading, dest, note))

    if heatmap_pngs:
        parts.append("## Heatmap comparison (real vs generated)")
        parts.append("")
        for k_val in sorted(heatmap_pngs.keys()):
            p = heatmap_pngs[k_val]
            if not p.is_file():
                continue
            dest = _prepare_report_plot(run_dir, p, f"K{k_val:02d}_heatmap.png")
            heading = f"K = {k_val:02d} heatmap"
            decap_line = active_decaps_summary(run_dir, k_val)
            parts.append(f"### {heading}")
            parts.append("")
            if decap_line:
                parts.append(decap_line)
                parts.append("")
            parts.append(
                _markdown_image(f"K{k_val:02d} heatmap compare", dest, run_dir, embed=embed_images)
            )
            parts.append("")
            note = decap_line.replace("**", "") if decap_line else None
            html_sections.append((heading, dest, note))

    report_path.write_text("\n".join(parts), encoding="utf-8")

    if write_html and html_sections:
        html_path = _write_html_report(
            run_dir,
            title=f"Impedance comparison ({title})",
            sections=html_sections,
        )
        print(f"  HTML report (images embedded) → {html_path.relative_to(run_dir)}")

    return report_path
