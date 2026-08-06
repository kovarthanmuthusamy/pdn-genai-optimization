---
title: impedance
type: code
path: libs/data_creation/impedance.py
group: libs/data_creation
loc: 105
tags: [code, libs]
---

# impedance

> Read IC1 impedance CSV files and plot log-log impedance profiles.

**Source:** `libs/data_creation/impedance.py` · 105 lines

## Purpose

```text
Read IC1 impedance CSV files and plot log-log impedance profiles.

Run:
    Import from data pipelines: ``from libs.data_creation.impedance import read_impedance_file``.
```

## Constants

| Name | Value |
|------|-------|
| `EXPECTED_IMP_LENGTH` | `231` |

## Functions

- **`_read_csv(filepath, cols=None, **kwargs)`** — Generic CSV reader.
- **`read_impedance_file(filepath)`** — Reads impedance values from .csv file.
- **`visualize_impedance(impedance_file, output_path=None, show=True)`** — Visualize impedance profile from a numpy array or a .npy file.

## Imported by

- [[normalize_labels]]
- [[processing_eval]]
- [[processing_multifreq]]
- [[visualize_sample]]

## External dependencies

`matplotlib`, `numpy`, `pandas`
