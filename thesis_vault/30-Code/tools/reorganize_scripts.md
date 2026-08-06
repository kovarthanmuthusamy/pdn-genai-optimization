---
title: reorganize_scripts
type: code
path: tools/reorganize_scripts.py
group: tools
loc: 275
tags: [code, tools]
---

# reorganize_scripts

> One-time layout migration: move scripts into pipelines/ + libs/, leave compat shims.

**Source:** `tools/reorganize_scripts.py` · 275 lines

## Constants

| Name | Value |
|------|-------|
| `ROOT` | `Path(__file__).resolve().parents[1]` |
| `IMPORT_REPLACEMENTS` | `[('\\bfrom heatmap import\\b', 'from libs.data_creation.heatmap import'), ('\\bfrom imped…` |
| `SHIM_PKG_MAP` | `{'libs/data_creation/heatmap.py': 'libs.data_creation.heatmap', 'libs/data_creation/imped…` |

## Functions

- **`_module_path(new_rel: str)`**
- **`_shim_content(old_rel: str, new_rel: str)`**
- **`_fix_shim_root_depth(old_rel: str, new_rel: str)`**
- **`make_shim(old_rel: str, new_rel: str)`**
- **`patch_imports(path: Path)`**
- **`main()`**
