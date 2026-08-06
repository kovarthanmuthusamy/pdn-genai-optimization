---
title: fix_repo_paths_imports
type: code
path: tools/fix_repo_paths_imports.py
group: tools
loc: 47
tags: [code, tools]
---

# fix_repo_paths_imports

> Fix broken repo_paths import lines from migrate_all_paths.py.

**Source:** `tools/fix_repo_paths_imports.py` · 47 lines

## Constants

| Name | Value |
|------|-------|
| `REPO` | `Path(__file__).resolve().parents[1]` |
| `BROKEN` | `re.compile('from repo_paths import REPO_ROOT, resolve_repo_path as (PROJECT_ROOT\|_PROJECT…` |
| `FIXED` | `'from repo_paths import REPO_ROOT as PROJECT_ROOT, resolve_repo_path, setup_path\n'` |
| `BROKEN2` | `re.compile('from repo_paths import REPO_ROOT, resolve_repo_path as _PROJECT_ROOT, setup_p…` |
| `FIXED2` | `'from repo_paths import REPO_ROOT as _PROJECT_ROOT, resolve_repo_path, setup_path\n'` |
| `DUP_IMPORTS` | `re.compile('import sys\\nfrom pathlib import Path\\n\\nimport sys\\nfrom pathlib import P…` |

## Functions

- **`main()`**
