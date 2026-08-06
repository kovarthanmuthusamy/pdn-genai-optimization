---
title: fix_scratch_paths
type: code
path: tools/fix_scratch_paths.py
group: tools
loc: 89
tags: [code, tools]
---

# fix_scratch_paths

> Normalize scratch/*.py to use repo_paths.

**Source:** `tools/fix_scratch_paths.py` · 89 lines

## Constants

| Name | Value |
|------|-------|
| `REPO` | `Path(__file__).resolve().parents[1]` |
| `SCRATCH` | `REPO / 'scratch'` |
| `OLD_NEXT` | `re.compile('sys\\.path\\.insert\\(0, str\\(next\\(p for p in Path\\(__file__\\)\\.resolve…` |
| `NEW_BOOT` | `'from repo_paths import setup_path\n\nsetup_path()\n'` |
| `OLD_ROOT1` | `re.compile('(_ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[1\\]\\n(?:if str\\(_R…` |
| `OLD_ROOT1_ALT` | `re.compile('ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[1\\]\\nsys\\.path\\.ins…` |
| `OLD_PROJECT_NEXT` | `re.compile('_ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\nPROJECT_ROOT = next\\(\\n \\(p f…` |
| `OLD_PROJECT_NEXT2` | `re.compile('PROJECT_ROOT = next\\(\\n \\(p for p in _ROOT\\.parents if \\(p / \\"src_vae\…` |
| `OLD_SMOKE` | `re.compile('_ROOT = Path\\(__file__\\)\\.resolve\\(\\)\\nPROJECT_ROOT = next\\(\\n str\\(…` |

## Functions

- **`patch(text: str)`**
- **`main()`**
