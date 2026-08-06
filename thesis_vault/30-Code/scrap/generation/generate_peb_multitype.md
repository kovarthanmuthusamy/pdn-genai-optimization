---
title: generate_peb_multitype
type: code
path: scrap/generation/generate_peb_multitype.py
group: scrap/generation
loc: 186
tags: [code, scrap, uncommitted]
---

# generate_peb_multitype

> Generate ECADStar Batch PEB for multi-type decap layouts.

**Source:** `scrap/generation/generate_peb_multitype.py` · 186 lines
**Git:** uncommitted — not yet tracked

## Purpose

```text
Generate ECADStar Batch PEB for multi-type decap layouts.

Each slot carries an integer *type code* (not just 0/1):

    0 = empty
    1 = type-1 decap
    2 = type-2 decap
    ...

For every component C{i}, four Edit lines are written (matching
``configs/Batch_Decap_example.peb``)::

    <Edit Table="Component" Name="Ci" Column="Name"  Value="true|false"/>
    <Edit Table="Component" Name="Ci" Column="Value" Value="{C in Farads}"/>
    <Edit Table="PI Decap"  Name="Ci" Column="ESR"   Value="{ESR in Ohms}"/>
    <Edit Table="PI Decap"  Name="Ci" Column="ESL"   Value="{ESL in Henries}"/>

Empty slots write ``false`` + ``0`` for Value/ESR/ESL.

The type → (C, ESR, ESL) mapping lives in ``TYPE_CATALOG`` (SI units).
```

## Constants

| Name | Value |
|------|-------|
| `INPUT_PATH` | `'type_codes.npy'` |
| `OUTPUT_PATH` | `'new_multitype.peb'` |
| `POWERBUS` | `'Power_GND'` |
| `FREQ` | `'63e6'` |
| `COMPONENTS` | `'IC1_Port1,IC2_Port2'` |
| `N_COMPONENTS` | `52` |

## Functions

- **`_fmt(v: float)`** — Format a numeric value ECADStar-style (e.g. 1e-07, 0.0089, 0).
- **`catalog_for_type(type_id: int)`**
- **`build_component_edits(type_row: np.ndarray, indent: str='      ')`** — Return the 4-line Edit block for all 52 components from a type-code row.
- **`build_distribution_group(type_row: np.ndarray, freq: str)`**
- **`build_spectrum_group(type_row: np.ndarray, components: str)`**
- **`generate_peb_multitype(type_codes: np.ndarray, output_path: str, powerbus: str='Power_GND', freq: str='63e6', components: str='IC1_Port1,IC2_Port2', per_sample_freqs: list[str] | None=None, include_distribution: bool=True, include_spectrum: bool=True)`** — Write a multi-type Batch PEB.

## Imported by

- [[peb_multitype]]

## External dependencies

`numpy`
