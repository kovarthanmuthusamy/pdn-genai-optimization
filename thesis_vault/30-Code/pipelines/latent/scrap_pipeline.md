---
title: scrap_pipeline
type: code
path: pipelines/latent/scrap_pipeline.py
group: pipelines/latent
loc: 642
tags: [code, pipelines]
---

# scrap_pipeline

> Latent scrap pipeline — shared export, PEB, and impedance-compare helpers.

**Source:** `pipelines/latent/scrap_pipeline.py` · 642 lines

## Purpose

```text
Latent scrap pipeline — shared export, PEB, and impedance-compare helpers.

Run:
    Not run directly — imported by latent_run_export_peb.py and
```

## Constants

| Name | Value |
|------|-------|
| `EXPORT_SUBDIR` | `'exported_samples'` |
| `REPORT_PLOTS_SUBDIR` | `'report_plots'` |
| `PEB_NAME` | `'latent_run.peb'` |
| `CMP_IMPEDANCE` | `'generated_vs_real_impedance_profile.png'` |
| `CMP_GEN_TARGET` | `'generated_vs_target_impedance.png'` |
| `PI_FREQ_MHZ_DEFAULT` | `100` |
| `CMP_HEATMAP` | `'generated_vs_real_heatmap.png'` |
| `CMP_OCCUPANCY` | `'generated_occupancy.png'` |
| `_COMPARE_PLOT_NAMES` | `(CMP_IMPEDANCE, CMP_GEN_TARGET, CMP_HEATMAP, CMP_OCCUPANCY)` |

## Functions

- **`export_root(run_dir: Path)`**
- **`active_decaps_summary(run_dir: Path, k: int)`** — Short line listing decap components turned on for this K result.
- **`collect_existing_compare_pngs(run_dir: Path, k_values: list[int])`** — Find existing comparison PNGs under exported_samples/K{n}/ (for --report-only).
- **`discover_latent_k_dirs(run_dir: Path)`** — Map K → latent solution folder (run_dir/K##/ with best_latent.npy).
- **`peb_pi_layout(peb_path: Path)`** — Return (pis_per_sample, pi_output_kind) from PEB content.
- **`peb_pis_per_sample(peb_path: Path)`** — Backward-compatible: PI count per sample inferred from PEB.
- **`load_pi_freq_mhz(run_dir: Path)`**
- **`export_k_from_latent(latent_k_dir: Path, out_k_dir: Path)`** — Write run_all_k-compatible files (occupancy + impedance log only).
- **`move_pi_from_emc(run_dir: Path, k_values: list[int], *, source_emc_dir: str, peb_path: Path | None=None)`** — Move ECADStar PI-* outputs into exported_samples/K{n}/Real/ (scrap/move_pi_to_real).
- **`export_all(run_dir: Path)`** — Export every K with a latent solution. Returns sorted K list.
- **`clear_exported_real_and_plots(run_dir: Path, k_values: list[int] | None=None)`** — Remove stale ECADStar Real/ data and comparison PNGs under exported_samples/K{n}/.
- **`generate_peb_for_run(run_dir: Path, k_values: list[int], *, pi_freq_mhz: float, powerbus: str='Power_GND', components: str='IC1_Port1')`**
- **`_real_has_impedance(real_dir: Path)`**
- **`_sample_has_gen_heatmap(sample_dir: Path)`**
- **`run_compare_k_like_scrap(k: int, *, repo_root: Path, base_generated_dir: Path, pi_freq_mhz: float)`** — Run scrap/comparison/compare._run_single_k (Target + Real + Generated).
- **`_load_gen_impedance_row(cmp: Any, k_dir: Path, k: int, i: int)`**
- **`run_impedance_compare_k(k: int, *, repo_root: Path, base_generated_dir: Path, pi_freq_mhz: float)`** — Impedance Target + Real + Generated (same as scrap/comparison/compare.py).
- **`run_impedance_gen_vs_target_k(k: int, *, repo_root: Path, base_generated_dir: Path, pi_freq_mhz: float)`** — Plot generated impedance vs target (no Real/ data required).
- **`run_impedance_compare_all(run_dir: Path, repo_root: Path, k_values: list[int], *, pi_freq_mhz: float)`**
- **`_prepare_report_plot(run_dir: Path, src: Path, dest_name: str)`** — Copy plot into run_dir/report_plots/ and return the destination path.
- **`_png_data_uri(path: Path)`**
- **`_markdown_image(alt: str, img_path: Path, run_dir: Path, *, embed: bool)`** — Markdown image line that works in Cursor/VS Code preview (incl. WSL UNC workspaces).
- **`_write_html_report(run_dir: Path, *, title: str, sections: list[tuple[str, Path, str | None]])`** — Standalone HTML report with embedded PNGs (always renders in a browser).
- **`write_compare_report(run_dir: Path, *, peb_path: Path | None, compare_pngs: dict[int, Path], heatmap_pngs: dict[int, Path] | None=None, include_config: bool=True, embed_images: bool | None=None, write_html: bool=True)`** — run_report.md (+ run_report.html) with impedance compare figures.

## Imports

- [[compare]]
- [[generate_peb]]
- [[generate_run_report]]
- [[move_pi_to_real]]

## Imported by

- [[compare_report]]
- [[export_peb]]

## External dependencies

`base64`, `matplotlib`, `numpy`, `scrap`
