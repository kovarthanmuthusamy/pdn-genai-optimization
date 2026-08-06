---
title: peb_copy
type: code
path: scrap/peb_copy.py
group: scrap
loc: 44
tags: [code, scrap]
---

# peb_copy

> Copy PEB Files (WSL-Safe).

**Source:** `scrap/peb_copy.py` · 44 lines

## Purpose

```text
Copy PEB Files (WSL-Safe).

Run: import and call ``copy_peb_to_folder`` (library helper, not a standalone entry script).
```

## Functions

- **`resolve_windows_path(path_str: str)`** — Resolve a Windows path; on WSL use /mnt/<drive>/... when the drive is mounted.
- **`copy_peb_to_folder(peb_file: Path, dest_dir: str | Path, *, mkdir: bool=True)`** — Copy *peb_file* into *dest_dir* (created if ``mkdir``). Returns destination path.

## Imported by

- [[multifreq_move_and_compare]]
- [[run_multifreq_heatmap_sweep]]
