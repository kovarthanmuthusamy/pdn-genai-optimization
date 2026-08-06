---
title: fix_docstring_paths
type: code
path: tools/fix_docstring_paths.py
group: tools
loc: 69
tags: [code, tools]
---

# fix_docstring_paths

> Fix docstrings corrupted by migrate_repo_paths.py (setup_path lines inside quotes).

**Source:** `tools/fix_docstring_paths.py` · 69 lines

## Constants

| Name | Value |
|------|-------|
| `REPO` | `Path(__file__).resolve().parents[1]` |
| `BAD` | `re.compile('\\nfrom repo_paths import setup_path\\nsetup_path\\(\\)\\n', re.MULTILINE)` |
| `NEEDS_SETUP` | `['pipelines/analysis/check_mask.py', 'pipelines/normalize/apply_stats.py']` |

## Functions

- **`fix_file(path: Path)`**
- **`main()`**
