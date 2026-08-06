---
title: migrate_all_paths
type: code
path: tools/migrate_all_paths.py
group: tools
loc: 246
tags: [code, tools]
---

# migrate_all_paths

> Migrate stale absolute paths to repo-relative / repo_paths usage.

**Source:** `tools/migrate_all_paths.py` · 246 lines

## Purpose

```text
Migrate stale absolute paths to repo-relative / repo_paths usage.

Fixes:
  - experiments/*/config.yaml: /home/ubuntu/gan/... and /home/ubuntu/genai_pdn/...
  - Python: parents[N] repo-root bootstraps → repo_paths bootstrap
  - Python: hard-coded /home/ubuntu/gan and /home/ubuntu/genai_pdn string prefixes
  - Shell: cd /home/ubuntu/gan → cd to repo root via $(dirname ...) or relative note

Run from repo root:
  python tools/migrate_all_paths.py
  python tools/migrate_all_paths.py --dry-run
```

## Constants

| Name | Value |
|------|-------|
| `REPO` | `Path(__file__).resolve().parents[1]` |
| `SKIP_PY` | `{'repo_paths.py', 'gan_paths.py', 'migrate_repo_paths.py', 'migrate_all_paths.py', 'fix_r…` |
| `SKIP_PARTS` | `{'.git', '__pycache__', 'node_modules'}` |
| `STALE_PREFIXES` | `('/home/ubuntu/gan/', '/home/ubuntu/genai_pdn/', '/home/ubuntu/GAN/')` |
| `_BOOTSTRAP_INLINE` | `'import sys\nfrom pathlib import Path\n\n_REPO_BOOT = Path(__file__).resolve().parents[{d…` |
| `_BOOTSTRAP_P3` | `re.compile('^(?:PROJECT_ROOT\|_PROJECT_ROOT) = Path\\(__file__\\)\\.resolve\\(\\)\\.parent…` |
| `_BOOTSTRAP_P2` | `re.compile('^(?:PROJECT_ROOT\|_PROJECT_ROOT) = Path\\(__file__\\)\\.resolve\\(\\)\\.parent…` |
| `_STR_PROJECT_ROOT` | `re.compile('^(PROJECT_ROOT\|_PROJECT_ROOT)\\s*=\\s*["\\\']/home/ubuntu/(?:gan\|genai_pdn)["…` |
| `_STR_PROJECT_ROOT_REPL` | `'from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path\nsetup_path()\n'` |
| `_OLD_REPO_BOOT` | `re.compile('import sys\\nfrom pathlib import Path\\n\\n_REPO_BOOT = Path\\(__file__\\)\\.…` |

## Functions

- **`_bootstrap_repl(m: re.Match[str])`**
- **`_infer_alias_from_context(text: str, pos: int)`**
- **`strip_stale_in_text(text: str)`** — Replace stale absolute prefixes with repo-relative paths in quoted strings.
- **`patch_yaml_config(path: Path, *, dry_run: bool)`**
- **`patch_python(path: Path, *, dry_run: bool)`**
- **`patch_shell(path: Path, *, dry_run: bool)`**
- **`main()`**
