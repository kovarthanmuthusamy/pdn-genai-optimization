---
title: rename_folders
type: code
path: rename_folders.py
group: (root)
loc: 99
tags: [code, rename_folders.py, runnable]
---

# rename_folders

> Rename folders PI-19500..PI-38998 to PI-1..PI-19499 (rename only).

**Source:** `rename_folders.py` · 99 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python rename_folders.py`

## Purpose

```text
Rename folders PI-19500..PI-38998 to PI-1..PI-19499 (rename only).

Run: python rename_folders.py

Run: python rename_folders.py
```

## Constants

| Name | Value |
|------|-------|
| `ROOT` | `Path('C:\\Users\\muthusamy\\Desktop\\Raw\\heatmap_450MHz')` |
| `DRY_RUN` | `True` |
| `SOURCE_START` | `19500` |
| `SOURCE_END` | `38998` |
| `PREFIX` | `'PI-'` |
| `FOLDER_PATTERN` | `re.compile(f'^{re.escape(PREFIX)}(\\d+)$')` |

## Functions

- **`collect_renames(root: Path)`** — Build ordered list of (old_path, new_path) for matching folders.
- **`apply_renames(renames: list[tuple[Path, Path]], dry_run: bool)`** — Rename folders via a temporary name to avoid collisions.
- **`main()`**
