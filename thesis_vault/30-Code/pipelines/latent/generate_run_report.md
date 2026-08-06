---
title: generate_run_report
type: code
path: pipelines/latent/generate_run_report.py
group: pipelines/latent
loc: 296
tags: [code, pipelines, runnable]
---

# generate_run_report

> Run report generator — Markdown summary for a latent optimization run.

**Source:** `pipelines/latent/generate_run_report.py` · 296 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/latent/generate_run_report.py`

## Purpose

```text
Run report generator — Markdown summary for a latent optimization run.

Run:
    python pipelines/latent/generate_run_report.py
```

## Functions

- **`_fmt_list(v: list, max_items: int=10)`**
- **`_fmt_seconds(s: float)`**
- **`_isnan(v: float)`**
- **`build_report(run_folder: Path)`**
- **`write_report(run_folder: Path)`**
- **`main()`**

## Imports

- [[repo_paths]]

## Imported by

- [[scrap_pipeline]]

## External dependencies

`repo_paths`
