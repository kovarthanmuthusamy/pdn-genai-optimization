---
title: fix_repo_paths_bootstrap
type: code
path: tools/fix_repo_paths_bootstrap.py
group: tools
loc: 63
tags: [code, tools]
---

# fix_repo_paths_bootstrap

> Insert sys.path bootstrap before ``from repo_paths import`` in entry scripts.

**Source:** `tools/fix_repo_paths_bootstrap.py` · 63 lines

## Constants

| Name | Value |
|------|-------|
| `REPO` | `Path(__file__).resolve().parents[1]` |
| `SKIP` | `{'repo_paths.py', 'gan_paths.py', 'fix_repo_paths_bootstrap.py', 'migrate_repo_paths.py'}` |
| `IMPORT_RE` | `re.compile('^from repo_paths import', re.MULTILINE)` |
| `BOOTSTRAP_MARK` | `'_REPO_BOOT = Path(__file__).resolve().parents['` |

## Functions

- **`_bootstrap_block(depth: int)`**
- **`patch_file(path: Path)`**
- **`main()`**
