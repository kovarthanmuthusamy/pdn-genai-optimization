# Folder Rename Script — heatmap_450MHz

## Purpose

Renamed **19,499** folders in `C:\Users\muthusamy\Desktop\Raw\heatmap_450MHz` from the range **PI-19500 … PI-38998** to **PI-1 … PI-19499**, preserving order.

Mapping rule:

```
PI-N  →  PI-(N - 19499)    where N is 19500..38998
```

Examples:

| Before     | After   |
|-----------|---------|
| PI-19500  | PI-1    |
| PI-19501  | PI-2    |
| PI-38998  | PI-19499|

## Script

**File:** `rename_folders.py`

**Behavior:**

- Only renames **directories** matching `PI-<number>`.
- Skips files and folders outside the 19500–38998 range.
- Uses a two-step temp rename (`__rename_tmp_XXXXX__`) to avoid name collisions.
- Does not move or modify files inside folders.

## Usage

Dry run (preview only):

```bash
python rename_folders.py --dry-run
```

Run rename (default path is the heatmap folder):

```bash
python rename_folders.py
```

Custom directory:

```bash
python rename_folders.py "D:\path\to\folders"
```

## Execution Result

- **Date:** 2026-06-13
- **Folders renamed:** 19,499
- **Status:** Completed successfully (~29 seconds)

## Verification

After running, the directory contains folders **PI-1** through **PI-19499** (19,499 total). No `PI-19500+` folders remain.
