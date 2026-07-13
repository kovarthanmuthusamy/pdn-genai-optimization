"""Build Comparison HTML Report.

Run: python scrap/build_comparison_report.py"""
from __future__ import annotations

import base64
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Iterable

from repo_paths import REPO_ROOT as _PROJECT_ROOT, setup_path
setup_path()

from scrap.comparison.compare import HEATMAP_OUT_NAME  # noqa: E402
from scrap.comparison.heatmap_sim_metrics import SIM_METRICS_MD  # noqa: E402

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

# WORKFLOW = "multifreq_heatmap_sweep"  # run_all_k | multifreq_heatmap_sweep
WORKFLOW = "run_all_k"

# =============================================================================

HEATMAP_IMG = HEATMAP_OUT_NAME
IMPEDANCE_IMG = "generated_vs_real_impedance_profile.png"
OCCUPANCY_IMG = "generated_occupancy.png"

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body {
    font-family: Arial, sans-serif;
    background: #f5f5f5;
    color: #222;
}

#header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 100;
    background: #fff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
h1 { padding: 20px 28px 10px; font-size: 3rem; text-align: center; }
.subtitle { text-align: center; font-size: 1.1rem; color: #555; padding-bottom: 12px; }

.tab-bar {
    display: flex;
    justify-content: center;
    gap: 24px;
    padding: 0 36px;
    border-bottom: 3px solid #ccc;
    background: #fff;
}
.tab-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    padding: 18px 48px 14px;
    border: none;
    border-bottom: 5px solid transparent;
    background: none;
    cursor: pointer;
    font-size: 1.5rem;
    font-weight: 700;
    color: #555;
    letter-spacing: 0.03em;
    transition: color .15s, border-color .15s;
}
.tab-btn svg {
    width: 3rem;
    height: 3rem;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.tab-btn:hover { color: #000; }
.tab-btn.active { color: #1a73e8; border-bottom-color: #1a73e8; }

#content {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    overflow: hidden;
}

.tab-panel {
    display: none;
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    overflow-y: auto;
    padding: 28px;
}
.tab-panel.active { display: block; }

.k-section { margin-bottom: 36px; text-align: center; }
.k-section h3 {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 10px;
    padding: 8px 14px;
    background: #e8f0fe;
    border-left: 5px solid #1a73e8;
    border-radius: 3px;
    text-align: center;
}
.k-section img {
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: #fff;
    display: block;
    margin: 0 auto;
}
.missing { color: #c00; font-style: italic; }

.k-links {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin: 8px 0 16px;
    flex-wrap: wrap;
}
.k-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 18px;
    font-size: 1rem;
    font-weight: 600;
    color: #1a73e8;
    background: #e8f0fe;
    border: 1.5px solid #1a73e8;
    border-radius: 20px;
    cursor: pointer;
    border: none;
    transition: background .15s, color .15s;
}
.k-link:hover {
    background: #1a73e8;
    color: #fff;
}
"""

_JS = """
function showTab(idx) {
    document.querySelectorAll('.tab-btn').forEach((b, i) => {
        b.classList.toggle('active', i === idx);
    });
    document.querySelectorAll('.tab-panel').forEach((p, i) => {
        p.classList.toggle('active', i === idx);
    });
}

function goTo(tabIdx, anchorId) {
    showTab(tabIdx);
    requestAnimationFrame(() => {
        const el = document.getElementById(anchorId);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
}

function fitContent() {
    const h = document.getElementById('header').offsetHeight;
    document.getElementById('content').style.top = h + 'px';
}

window.addEventListener('load', fitContent);
window.addEventListener('resize', fitContent);
new ResizeObserver(fitContent).observe(document.getElementById('header'));
"""

_ICON_HEATMAP = (
    '<svg viewBox="0 0 24 24">'
    '<path d="M12 2C8 2 5 6 5 10c0 5.25 7 12 7 12s7-6.75 7-12c0-4-3-8-7-8z"/>'
    '<circle cx="12" cy="10" r="2.5"/>'
    '</svg>'
)
_ICON_IMPEDANCE = (
    '<svg viewBox="0 0 24 24">'
    '<polyline points="2,17 6,11 10,14 14,7 18,10 22,4"/>'
    '</svg>'
)
_ICON_OCCUPANCY = (
    '<svg viewBox="0 0 24 24">'
    '<rect x="3" y="3" width="7" height="7" rx="1"/>'
    '<rect x="14" y="3" width="7" height="7" rx="1"/>'
    '<rect x="3" y="14" width="7" height="7" rx="1"/>'
    '<rect x="14" y="14" width="7" height="7" rx="1"/>'
    '</svg>'
)


def _img_tag(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<p class="missing">Image not found: {path}</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def _copy_report_to_dest(out_path: Path, dest_str: str | None) -> None:
    if not dest_str:
        return
    if re.match(r"^[A-Za-z]:\\", dest_str):
        win = PureWindowsPath(dest_str)
        drive = win.drive.rstrip(":").lower()
        dest_path = Path("/mnt") / drive / Path(*win.parts[1:])
    else:
        dest_path = Path(dest_str)
    try:
        dest_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        print(f"⚠ REPORT_COPY_DEST not writable, skipping copy: {dest_path}")
        return
    if dest_path.is_dir():
        target = dest_path / out_path.name
        shutil.copy2(str(out_path), str(target))
        print(f"Copied to   : {target}")
        print(f"Open from Windows: {dest_str}\\{out_path.name}")
    else:
        print(f"⚠ REPORT_COPY_DEST not found, skipping copy: {dest_path}")


def _build_panel(
    panel_id: str,
    tab_idx: int,
    k_iter: Iterable[int],
    img_name: str,
    base_dir: Path,
    freq_scan: list[tuple[str, str | None]],
    all_tabs: list[tuple[str, str, str, str]],
    *,
    active: bool = False,
) -> str:
    active_cls = " active" if active else ""
    parts = [f'<div class="tab-panel{active_cls}" id="{panel_id}">']
    for freq_label, freq_sub in freq_scan:
        freq_dir = base_dir / freq_sub if freq_sub else base_dir
        for k in k_iter:
            img_path = freq_dir / f"K{k}" / img_name
            safe_tag = freq_sub.replace("_", "-") if freq_sub else "nf"
            sec_id = f"{panel_id}-{safe_tag}-k{k}"
            heading = f"K = {k}" + (f" @ {freq_label}" if freq_label else "")
            parts.append(f'  <div class="k-section" id="{sec_id}">')
            parts.append(f"    <h3>{heading}</h3>")
            if len(all_tabs) > 1:
                links = []
                for i, (tid, label, _img, icon) in enumerate(all_tabs):
                    if i == tab_idx:
                        continue
                    short = label.replace(" Comparison", "")
                    cross_id = f"{tid}-{safe_tag}-k{k}"
                    links.append(
                        f'<button class="k-link" onclick="goTo({i},\'{cross_id}\')">'
                        f"{icon} {short}</button>"
                    )
                parts.append(f'    <div class="k-links">{" ".join(links)}</div>')
            parts.append(f"    {_img_tag(img_path, f'K={k}')}")
            parts.append("  </div>")
    parts.append("</div>")
    return "\n".join(parts)


def _write_markdown_summary(
    *,
    out_md: Path,
    base_dir: Path,
    freq_scan: list[tuple[str, str | None]],
    k_iter: Iterable[int],
    title: str,
    html_name: str,
    img_name: str = HEATMAP_IMG,
    section_title: str = "Heatmap comparisons",
) -> None:
    """Short index of comparison PNGs (open comparison_report.html for full gallery)."""
    lines = [
        f"# {title}",
        "",
        f"Interactive gallery: [`{html_name}`](./{html_name})",
        "",
        f"## {section_title}",
        "",
        "| PI frequency | K | PNG | Status |",
        "|---|---:|---|---|",
    ]
    for freq_label, freq_sub in freq_scan:
        freq_dir = base_dir / freq_sub if freq_sub else base_dir
        for k in k_iter:
            rel = (
                f"{freq_sub}/K{k}/{img_name}"
                if freq_sub
                else f"K{k}/{img_name}"
            )
            img_path = freq_dir / f"K{k}" / img_name
            status = "ok" if img_path.is_file() else "missing"
            fl = freq_label or "—"
            lines.append(f"| {fl} | {k} | `{rel}` | {status} |")
    metrics_md = base_dir / SIM_METRICS_MD
    if metrics_md.is_file():
        lines.extend([
            "",
            "## Simulated-real vs generated metrics",
            "",
            f"Full per-sample error table (ECADStar `.map` vs generated): [`{SIM_METRICS_MD}`](./{SIM_METRICS_MD})",
            "",
        ])
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written: {out_md}")


def render_comparison_report(
    *,
    base_dir: Path,
    output_html: Path,
    output_md: Path | None,
    freq_scan: list[tuple[str, str | None]],
    k_min: int,
    k_max: int,
    tabs: list[tuple[str, str, str, str]],
    title: str,
    subtitle: str = "",
    report_copy_dest: str | None = None,
    k_values: list[int] | None = None,
) -> Path:
    """Build HTML (and optional markdown index) from comparison PNGs under *base_dir*."""
    base_dir = base_dir.resolve()
    out_path = output_html.resolve()
    k_iter = list(k_values) if k_values is not None else list(range(k_min, k_max + 1))

    show_tabs = len(tabs) > 1
    buttons = ""
    if show_tabs:
        buttons = "\n".join(
            f'<button class="tab-btn{" active" if i == 0 else ""}" '
            f'onclick="showTab({i})">{icon}<span>{label}</span></button>'
            for i, (_, label, _, icon) in enumerate(tabs)
        )

    panels = "\n".join(
        _build_panel(tab_id, i, k_iter, img_name, base_dir, freq_scan, tabs, active=(i == 0))
        for i, (tab_id, _, img_name, _icon) in enumerate(tabs)
    )

    sub_html = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    tab_bar = f'<div class="tab-bar">\n{buttons}\n</div>' if show_tabs else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<div id="header">
<h1>{title}</h1>
{sub_html}
{tab_bar}
</div>
<div id="content">
{panels}
</div>
<script>{_JS}</script>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Written: {out_path}")

    if output_md is not None:
        primary_img = tabs[0][2] if tabs else HEATMAP_IMG
        primary_label = tabs[0][1] if tabs else "Heatmap Comparison"
        _write_markdown_summary(
            out_md=output_md.resolve(),
            base_dir=base_dir,
            freq_scan=freq_scan,
            k_iter=k_iter,
            title=title,
            html_name=out_path.name,
            img_name=primary_img,
            section_title=f"{primary_label}s" if not primary_label.endswith("s") else primary_label,
        )

    _copy_report_to_dest(out_path, report_copy_dest)
    if output_md is not None:
        _copy_report_to_dest(output_md.resolve(), report_copy_dest)
    return out_path


def build_run_all_k_report(
    *,
    report_copy_dest: str | None = None,
) -> Path:
    from scrap.generation.run_all_k import (  # noqa: E402
        K_MAX,
        K_MIN,
        OUTPUT_ROOT,
        PEB_COPY_DEST,
        PI_FREQ_MHZ,
    )

    if PI_FREQ_MHZ is None:
        freq_scan: list[tuple[str, str | None]] = [("", None)]
    elif isinstance(PI_FREQ_MHZ, (int, float)):
        mhz = int(PI_FREQ_MHZ)
        freq_scan = [(f"{mhz} MHz", f"freq_{mhz}MHz")]
    else:
        freq_scan = [(f"{int(f)} MHz", f"freq_{int(f)}MHz") for f in PI_FREQ_MHZ]

    base = _PROJECT_ROOT / OUTPUT_ROOT
    dest = report_copy_dest if report_copy_dest is not None else PEB_COPY_DEST
    tabs = [
        ("tab-heatmap", "Heatmap Comparison", HEATMAP_IMG, _ICON_HEATMAP),
        ("tab-impedance", "Impedance Comparison", IMPEDANCE_IMG, _ICON_IMPEDANCE),
        ("tab-occupancy", "Occupancy Comparison", OCCUPANCY_IMG, _ICON_OCCUPANCY),
    ]
    _dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    return render_comparison_report(
        base_dir=base,
        output_html=base / f"comparison_report_{_dt}.html",
        output_md=base / f"comparison_summary_{_dt}.md",
        freq_scan=freq_scan,
        k_min=K_MIN,
        k_max=K_MAX,
        tabs=tabs,
        title="Generated vs Real — Comparison Report",
        report_copy_dest=dest,
    )


def build_multifreq_sweep_report(
    *,
    report_copy_dest: str | None = None,
    run_heatmap: bool = True,
    run_impedance: bool = False,
) -> list[Path]:
    """Separate HTML + markdown indices for the multifreq sweep.

    Heatmap and impedance galleries are written to **independent** HTML files so
    each comparison can be opened / shared on its own. Toggle each with
    ``run_heatmap`` / ``run_impedance`` (driven by the pipeline CONFIG).
    """
    from scrap.generation.run_multifreq_heatmap_sweep import (  # noqa: E402
        OUTPUT_ROOT,
        REPORT_COPY_DEST,
        exported_freq_mhz_list,
        exported_k_values,
    )

    k_values = exported_k_values()
    base = _PROJECT_ROOT / OUTPUT_ROOT
    dest = report_copy_dest if report_copy_dest is not None else REPORT_COPY_DEST
    _dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    subtitle_k = (
        f"K = {k_values[0]}"
        if len(k_values) == 1
        else f"K = {', '.join(str(k) for k in k_values)}"
    )
    if run_impedance and not run_heatmap:
        freq_scan: list[tuple[str, str | None]] = [("", None)]
        subtitle = subtitle_k
    else:
        freq_scan = [(f"{mhz} MHz", f"freq_{mhz}MHz") for mhz in exported_freq_mhz_list()]
        subtitle = f"{subtitle_k} · {len(freq_scan)} PI frequencies"

    # (kind, tab tuple, title, html/md stem)
    report_specs: list[tuple[bool, tuple[str, str, str, str], str, str]] = [
        (
            run_heatmap,
            ("tab-heatmap", "Heatmap Comparison", HEATMAP_IMG, _ICON_HEATMAP),
            "PI-Distribution sweep — Heatmap: Generated vs Real",
            "heatmap",
        ),
        (
            run_impedance,
            ("tab-impedance", "Impedance Comparison", IMPEDANCE_IMG, _ICON_IMPEDANCE),
            "PI-Distribution sweep — Impedance: Generated vs Real",
            "impedance",
        ),
    ]

    outputs: list[Path] = []
    for enabled, tab, title, stem in report_specs:
        if not enabled:
            continue
        outputs.append(
            render_comparison_report(
                base_dir=base,
                output_html=base / f"comparison_report_{stem}_{_dt}.html",
                output_md=base / f"comparison_summary_{stem}_{_dt}.md",
                freq_scan=freq_scan,
                k_min=min(k_values),
                k_max=max(k_values),
                k_values=k_values,
                tabs=[tab],
                title=title,
                subtitle=subtitle,
                report_copy_dest=dest,
            )
        )

    if not outputs:
        print("⚠ No comparison report requested (heatmap & impedance both disabled).")
    return outputs


def build_report(workflow: str = "run_all_k") -> Path | list[Path]:
    if workflow == "multifreq_heatmap_sweep":
        return build_multifreq_sweep_report()
    if workflow == "run_all_k":
        return build_run_all_k_report()
    raise ValueError(f"Unknown workflow: {workflow!r}")


def main() -> None:
    build_report(WORKFLOW)


if __name__ == "__main__":
    main()
