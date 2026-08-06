---
title: migrate_repo_paths
type: code
path: tools/migrate_repo_paths.py
group: tools
loc: 158
tags: [code, tools]
---

# migrate_repo_paths

> One-shot migration: replace hard-coded repo-root detection with repo_paths imports.

**Source:** `tools/migrate_repo_paths.py` · 158 lines

## Constants

| Name | Value |
|------|-------|
| `REPO` | `Path(__file__).resolve().parents[1]` |
| `SKIP` | `{'repo_paths.py', 'gan_paths.py', 'migrate_repo_paths.py'}` |
| `_BOOTSTRAP_BLOCK` | `re.compile('# ── Bootstrap project root ─+\\n_PROJECT_ROOT = Path\\(__file__\\)\\.resolve…` |
| `_BOOTSTRAP_BLOCK2` | `re.compile('_PROJECT_ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]\\nif str\…` |
| `_PROJECT_ROOT_BOOT` | `'import sys\nfrom pathlib import Path\n\n_REPO_BOOT = Path(__file__).resolve().parents[2]…` |
| `_PROJECT_ROOT_BLOCK` | `re.compile('PROJECT_ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]\\nif str\\…` |
| `_PROJECT_ROOT_REPL` | `'import sys\nfrom pathlib import Path\n\n_REPO_BOOT = Path(__file__).resolve().parents[2]…` |
| `_SYSPATH_INSERT` | `re.compile('sys\\.path\\.insert\\(0, str\\(Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\…` |
| `_GAN_PATHS_IMPORT` | `re.compile('from gan_paths import', re.MULTILINE)` |
| `_ABS_GAN` | `re.compile('/home/ubuntu/gan')` |
| `_ROOT_LINE` | `re.compile('^ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]\\n', re.MULTILINE)` |
| `_ROOT_REPL` | `'from repo_paths import REPO_ROOT as ROOT, setup_path\nsetup_path()\n'` |
| `_REPO_ROOT_VAR` | `re.compile('^_REPO_ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]\\n', re.MUL…` |
| `_REPO_ROOT_REPL` | `'from repo_paths import REPO_ROOT as _REPO_ROOT, setup_path\nsetup_path()\n'` |
| `_ROOT_VAR` | `re.compile('^_ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]\\n', re.MULTILIN…` |
| `_ROOT_VAR_REPL` | `'from repo_paths import REPO_ROOT as _ROOT, setup_path\nsetup_path()\n'` |
| `_REPO_ROOT_LOWER` | `re.compile('^_repo_root = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]\\n', re.MUL…` |
| `_REPO_ROOT_LOWER_REPL` | `'from repo_paths import REPO_ROOT as _repo_root, setup_path\nsetup_path()\n'` |

## Functions

- **`_ensure_repo_paths_import(text: str)`**
- **`patch_file(path: Path)`**
- **`main()`**
