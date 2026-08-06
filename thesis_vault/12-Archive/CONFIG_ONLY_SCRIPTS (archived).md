---
title: CONFIG_ONLY_SCRIPTS (archived)
type: archive
source: docs/_archive/CONFIG_ONLY_SCRIPTS.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# CONFIG-only scripts

Runnable entry scripts use a **CONFIG block** near the top of the file (after imports). Edit constants there, then run:

```bash
python path/to/script.py
```

## Workflow

1. Open the script and find `# CONFIGURATION — edit these before running`.
2. Set paths, flags, and numeric parameters in that block.
3. Run with `python` only — no extra arguments.

## Module docstrings (Agent notes)

Each script documents:

| Section | Meaning |
|---------|---------|
| **Purpose** | What problem the script solves |
| **Run** | Exact ``python path/script.py`` command |
| **Agent notes → What** | One-line definition |
| **Agent notes → Usage** | Edit CONFIG block, then run |
| **Agent notes → Config keys** | Each constant and what it controls |
| **Agent notes → Key symbols** | Main functions/classes to import or read |

Library modules (under ``libs/``, model classes) say **Import only** in Run and omit Config keys.

## Main entry points

| Script | Important CONFIG keys |
|--------|------------------------|
| `pipelines/active_learning/run.py` | `COMMAND`, `CONFIG_PATH`, `ITERATION`, `SKIP_SIMULATE`, `SKIP_INGEST`, `PROPOSE_ONLY` |
| `pipelines/latent/optimize.py` | `K_LIST`, `NUM_STEPS`, `LR`, `USE_SURROGATE`, checkpoint paths |
| `pipelines/data/processing_multifreq.py` | `OUTPUT_ROOT`, `DATA_ROOT`, `APPEND`, MHz filters |
| `pipelines/normalize/multifreq.py` | `DATA_DIR`, `OUTPUT_DIR`, `APPEND` |
| `pipelines/dataset/extract_subset.py` | `SRC`, `REF`, `DST`, `EXECUTE`, `OVERWRITE`, `RESUME` |
| `experiments/exp043/codes/evaluate_vae.py` | `CHECKPOINT`, `TEST_MHZ_LIST`, `SKIP_OFF_ANCHOR`, `MAX_VAL_BATCHES` |
| `evaluation/novelty/scripts/run_vae_novelty_test.py` | `K_VALUE`, `NUM_SAMPLES`, `CHECKPOINT`, `DATASET_ROOT` |

Active learning also reads `active_learning_pi/config/default.json` when `CONFIG_PATH = None` in `run.py`.

## Troubleshooting

- **Wrong entry path** — use `pipelines/active_learning/run.py`, not removed shims under legacy folders.
- **`pipeline.py` message** — library module; run `pipelines/active_learning/run.py` instead.
- **Dry-run messages** — set boolean flags like `EXECUTE=True` in CONFIG, not terminal flags.
