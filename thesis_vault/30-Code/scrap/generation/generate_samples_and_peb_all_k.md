---
title: generate_samples_and_peb_all_k
type: code
path: scrap/generation/generate_samples_and_peb_all_k.py
group: scrap/generation
loc: 104
tags: [code, scrap, runnable]
---

# generate_samples_and_peb_all_k

> Generate samples and combined PEB for all K (legacy wrapper).

**Source:** `scrap/generation/generate_samples_and_peb_all_k.py` · 104 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/generation/generate_samples_and_peb_all_k.py`

## Purpose

```text
Generate samples and combined PEB for all K (legacy wrapper).

Purpose:
    Loop ``K_MIN``..``K_MAX`` calling single-K generation; stack occupancies into one combined PEB.

Run:
    python scrap/generation/generate_samples_and_peb_all_k.py

Agent notes:
    - What: Multi-K wrapper around ``generate_samples_and_peb`` (legacy). Prefer ``run_all_k.py`` for new sweeps.
    - Usage: Set ``K_MIN``, ``K_MAX``, ``OUT_ROOT``; inherits VAE settings from ``gsp`` module CONFIG.
    - Config keys:
        - ``K_MIN``, ``K_MAX`` — decap budget range
        - ``OUT_ROOT``, ``PEB_OUT_FILE`` — per-K folders and combined PEB output
    - Key symbols: ``main``, ``gsp`` (imported generate module)
```

## Constants

| Name | Value |
|------|-------|
| `SCRAP_DIR` | `Path(__file__).resolve().parent` |
| `K_MIN` | `1` |
| `K_MAX` | `52` |
| `OUT_ROOT` | `Path('scrap/generated_samples_v2')` |
| `PEB_OUT_DIR` | `Path(gsp.PEB_PATH) if getattr(gsp, 'PEB_PATH', '') else Path('scrap/PEB')` |
| `PEB_OUT_FILE` | `PEB_OUT_DIR / f'K{K_MIN}_to_K{K_MAX}.peb'` |

## Functions

- **`main()`**

## Imports

- [[generate_samples_and_peb]]
- [[scrap.generation.__init__]]

## External dependencies

`numpy`, `torch`
