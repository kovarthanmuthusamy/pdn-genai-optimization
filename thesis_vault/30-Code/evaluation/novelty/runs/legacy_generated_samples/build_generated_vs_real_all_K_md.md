---
title: build_generated_vs_real_all_K_md
type: code
path: evaluation/novelty/runs/legacy_generated_samples/build_generated_vs_real_all_K_md.py
group: evaluation/novelty/runs/legacy_generated_samples
loc: 105
tags: [code, evaluation]
---

# build_generated_vs_real_all_K_md

> Generate a single Markdown file embedding per-K comparison plots.

**Source:** `evaluation/novelty/runs/legacy_generated_samples/build_generated_vs_real_all_K_md.py` · 105 lines

## Purpose

```text
Generate a single Markdown file embedding per-K comparison plots.

Expected folder layout (relative to this script's directory):
    K1/
      generated_vs_real_heatmap.png
      generated_vs_real_impedance_profile.png
      generated_occupancy.png
    K2/
      ...

It writes `generated_vs_real_all_K.md` next to this script.

Run:
    python scrap/generated_samples/build_generated_vs_real_all_K_md.py
```

## Classes

- **`ImageSpec`**

## Functions

- **`_parse_k_dir_name(name: str)`**
- **`_find_k_dirs(base_dir: Path)`**
- **`_relpath(from_dir: Path, to_path: Path)`**
- **`build_markdown(base_dir: Path)`**
- **`main()`**
