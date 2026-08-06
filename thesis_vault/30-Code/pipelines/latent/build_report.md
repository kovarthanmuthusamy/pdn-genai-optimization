---
title: build_report
type: code
path: pipelines/latent/build_report.py
group: pipelines/latent
loc: 628
tags: [code, pipelines, runnable]
---

# build_report

> Build latent opt report — self-contained HTML with embedded impedance/heatmap/occ plots.

**Source:** `pipelines/latent/build_report.py` · 628 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/latent/build_report.py`

## Purpose

```text
Build latent opt report — self-contained HTML with embedded impedance/heatmap/occ plots.

Run:
    python pipelines/latent/build_report.py
```

## Constants

| Name | Value |
|------|-------|
| `FREQUENCY_PATH` | `'configs/Frequency_data_hz.npy'` |
| `TARGET_IMP_PATH` | `'configs/target_impedance.npy'` |
| `MASK_PATH` | `'configs/binary_mask.npy'` |
| `NORM_STATS_PATH` | `'datasets/data_norm/normalization_stats.json'` |
| `HEATMAP_CMAP` | `'jet'` |
| `HEATMAP_LEVELS` | `22` |
| `IMP_YLIM` | `(0.001, 100.0)` |
| `DPI` | `200` |
| `PNG_IMP` | `'best_seed_impedance.png'` |
| `PNG_HM` | `'best_seed_heatmap.png'` |
| `PNG_OCC` | `'best_seed_occupancy.png'` |
| `DEVICE` | `'cuda' if torch.cuda.is_available() else 'cpu'` |
| `DTYPE` | `torch.float32` |
| `_CSS` | `'\n* { box-sizing: border-box; margin: 0; padding: 0; }\nhtml, body { height: 100%; overf…` |
| `_JS` | `"\nfunction showTab(idx) {\n document.querySelectorAll('.tab-btn').forEach((b,i) => b.cla…` |
| `_ICON_IMP` | `'<svg viewBox="0 0 24 24"><polyline points="2,17 6,11 10,14 14,7 18,10 22,4"/></svg>'` |
| `_ICON_HM` | `'<svg viewBox="0 0 24 24"><path d="M12 2C8 2 5 6 5 10c0 5.25 7 12 7 12s7-6.75 7-12c0-4-3-…` |
| `_ICON_OCC` | `'<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="…` |

## Classes

- **`NormStats`**
- **`HmStats`**

## Functions

- **`_project_root()`**
- **`_load_json(path: Path)`**
- **`_load_norm_stats(path: Path)`**
- **`_latest_run_dir(runs_root: Path)`**
- **`_iter_solution_dirs(run_dir: Path)`** — All leaf dirs that contain a feasible_best_latent.npy, sorted by (K, seed).
- **`_parse_k(sol_dir: Path)`**
- **`_best_seed(solutions: list[dict])`** — Return the solution with the lowest peak impedance (max of imp_ohm).
- **`_decode(model, z_t: torch.Tensor, K_tensor: torch.Tensor)`** — Call model.decode() with or without PI_freq depending on signature.
- **`_denorm_imp(imp_norm: torch.Tensor, stats: NormStats)`** — Return blended log-impedance curve, then convert to Ohm.
- **`_denorm_hm(hm_z: torch.Tensor, stats: HmStats)`**
- **`_decode_solution(sol_dir: Path, model, stats: NormStats, hm_stats: HmStats)`**
- **`_plot_impedance(sol: dict, frequency_hz: np.ndarray, target_imp: np.ndarray, out_path: Path)`**
- **`_plot_heatmap(sol: dict, mask: np.ndarray | None, cmap, out_path: Path)`**
- **`_plot_occupancy(sol: dict, out_path: Path)`**
- **`_draw_occ_checkboxes(ax, occ_prob: np.ndarray, *, k_expected: int)`**
- **`_img_tag(path: Path, alt: str)`**
- **`_build_panel(panel_id: str, tab_idx: int, k_img_paths: dict[int, Path], all_tabs: list, active: bool)`**
- **`_build_html(title: str, imp_paths: dict[int, Path], hm_paths: dict[int, Path], occ_paths: dict[int, Path])`**
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`base64`, `importlib`, `inspect`, `matplotlib`, `numpy`, `repo_paths`, `torch`
