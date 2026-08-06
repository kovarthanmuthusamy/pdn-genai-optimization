---
title: compare_report
type: code
path: pipelines/latent/compare_report.py
group: pipelines/latent
loc: 85
tags: [code, pipelines, runnable]
---

# compare_report

> Latent run compare report — step 2: PI move, impedance plots, and report.

**Source:** `pipelines/latent/compare_report.py` · 85 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/latent/compare_report.py`

## Purpose

```text
Latent run compare report — step 2: PI move, impedance plots, and report.

Run: python pipelines/latent/compare_report.py
```

## Constants

| Name | Value |
|------|-------|
| `SOURCE_EMC_DIR` | `os.getenv('LATENT_SOURCE_EMC_DIR', 'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc')` |
| `REPORT_ONLY` | `False` |
| `NO_MOVE` | `False` |

## Functions

- **`_k_list(run_dir: Path)`**
- **`main()`**

## Imports

- [[optimization_loader]]
- [[repo_paths]]
- [[scrap_pipeline]]

## External dependencies

`repo_paths`
