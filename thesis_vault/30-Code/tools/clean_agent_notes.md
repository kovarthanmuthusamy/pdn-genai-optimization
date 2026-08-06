---
title: clean_agent_notes
type: code
path: tools/clean_agent_notes.py
group: tools
loc: 142
tags: [code, tools, runnable]
---

# clean_agent_notes

> Remove CLI/argparse wording from module docstrings and user-facing messages.

**Source:** `tools/clean_agent_notes.py` · 142 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python tools/clean_agent_notes.py`

## Constants

| Name | Value |
|------|-------|
| `REPO` | `Path(__file__).resolve().parents[1]` |
| `USAGE_BLOCK` | `re.compile('\\n(?:Usage\|Example\|Run\\n---)\\n(?:.*\\n)*?(?=\\n(?:Outputs\|Dependencies\|Age…` |

## Functions

- **`_clean_docstring_usage(text: str)`** — Remove multi-line Usage/Example sections that only show --flags.
- **`process_file(path: Path)`**
- **`main()`**
