---
title: build_comparison_report
type: code
path: scrap/orchestration/build_comparison_report.py
group: scrap/orchestration
loc: 537
tags: [code, scrap, runnable]
---

# build_comparison_report

> Build Comparison HTML Report.

**Source:** `scrap/orchestration/build_comparison_report.py` · 537 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/orchestration/build_comparison_report.py`

## Purpose

```text
Build Comparison HTML Report.

Run: python scrap/build_comparison_report.py
```

## Constants

| Name | Value |
|------|-------|
| `WORKFLOW` | `'run_all_k'` |
| `HEATMAP_IMG` | `HEATMAP_OUT_NAME` |
| `IMPEDANCE_IMG` | `'generated_vs_real_impedance_profile.png'` |
| `OCCUPANCY_IMG` | `'generated_occupancy.png'` |
| `_CSS` | `'\n* { box-sizing: border-box; margin: 0; padding: 0; }\nhtml, body { height: 100%; overf…` |
| `_JS` | `"\nfunction showTab(idx) {\n document.querySelectorAll('.tab-btn').forEach((b, i) => {\n …` |
| `_ICON_HEATMAP` | `'<svg viewBox="0 0 24 24"><path d="M12 2C8 2 5 6 5 10c0 5.25 7 12 7 12s7-6.75 7-12c0-4-3-…` |
| `_ICON_IMPEDANCE` | `'<svg viewBox="0 0 24 24"><polyline points="2,17 6,11 10,14 14,7 18,10 22,4"/></svg>'` |
| `_ICON_OCCUPANCY` | `'<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="…` |

## Functions

- **`_img_tag(path: Path, alt: str)`**
- **`_copy_report_to_dest(out_path: Path, dest_str: str | None)`**
- **`_build_panel(panel_id: str, tab_idx: int, k_iter: Iterable[int], img_name: str, base_dir: Path, freq_scan: list[tuple[str, str | None]], all_tabs: list[tuple[str, str, str, str]], *, active: bool=False)`**
- **`_write_markdown_summary(*, out_md: Path, base_dir: Path, freq_scan: list[tuple[str, str | None]], k_iter: Iterable[int], title: str, html_name: str, img_name: str=HEATMAP_IMG, section_title: str='Heatmap comparisons')`** — Short index of comparison PNGs (open comparison_report.html for full gallery).
- **`render_comparison_report(*, base_dir: Path, output_html: Path, output_md: Path | None, freq_scan: list[tuple[str, str | None]], k_min: int, k_max: int, tabs: list[tuple[str, str, str, str]], title: str, subtitle: str='', report_copy_dest: str | None=None, k_values: list[int] | None=None)`** — Build HTML (and optional markdown index) from comparison PNGs under *base_dir*.
- **`build_run_all_k_report(*, report_copy_dest: str | None=None)`**
- **`build_multifreq_sweep_report(*, report_copy_dest: str | None=None, run_heatmap: bool=True, run_impedance: bool=False)`** — Separate HTML + markdown indices for the multifreq sweep.
- **`build_report(workflow: str='run_all_k')`**
- **`main()`**

## Imports

- [[compare]]
- [[heatmap_sim_metrics]]
- [[repo_paths]]
- [[run_all_k]]
- [[run_multifreq_heatmap_sweep]]

## Imported by

- [[multifreq_move_and_compare]]

## External dependencies

`base64`, `repo_paths`
