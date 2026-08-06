---
title: add_config_keys_to_docstrings
type: code
path: tools/add_config_keys_to_docstrings.py
group: tools
loc: 162
tags: [code, tools, runnable]
---

# add_config_keys_to_docstrings

> One-off: add Config keys to Agent notes and normalize CONFIGURATION headers.

**Source:** `tools/add_config_keys_to_docstrings.py` · 162 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python tools/add_config_keys_to_docstrings.py`

## Constants

| Name | Value |
|------|-------|
| `ROOT` | `Path(__file__).resolve().parents[1]` |
| `FOLDERS` | `['active_learning_pi', 'Data_Creation', 'datasets', 'experiments/exp043/codes', 'Latent_o…` |
| `CONFIG_HEADER` | `'# CONFIGURATION — edit these before running:'` |
| `CONFIG_BLOCK_RE` | `re.compile('(# ={5,}\\s*\\n# CONFIGURATION[^\\n]*\\n# ={5,}\\s*\\n)(.*?)(# ={5,}\\s*\\n)'…` |
| `ALT_CONFIG_RE` | `re.compile('(# ={5,}\\s*\\n# CONFIGURATION\\s*\\n# ={5,}\\s*\\n)(.*?)(# ={5,}\\s*\\n)', r…` |

## Functions

- **`_extract_keys(block: str)`**
- **`_find_config_keys(text: str)`**
- **`_is_library(text: str, path: Path)`**
- **`_add_config_keys_line(doc: str, keys_line: str)`**
- **`_fix_run_lines(doc: str, rel: str)`**
- **`process_file(path: Path)`**
- **`main()`**
