---
title: frequency
type: code
path: libs/peb/frequency.py
group: libs/peb
loc: 22
tags: [code, libs]
---

# frequency

> PEB PI-Distribution frequency replacement (shared by change_frequency / regenerate_mhz_pebs).

**Source:** `libs/peb/frequency.py` · 22 lines

## Constants

| Name | Value |
|------|-------|
| `_FREQ_PATTERN` | `re.compile('(<EditPIDistribution Frequency=")[^"]*(")')` |

## Functions

- **`replace_frequency_mhz(content: str, mhz: int | float | str)`** — Replace all PI-Distribution frequency tags; return (new_content, replacement_count).
- **`write_peb_at_mhz(src: Path, dst: Path, mhz: int | float | str)`** — Read *src*, set frequency to *mhz*, write *dst*; return replacement count.

## Imported by

- [[change_frequency]]
- [[regenerate_mhz_pebs]]
