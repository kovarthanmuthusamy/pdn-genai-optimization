---
title: model_to_config
type: code
path: src_vae/others/model_to_config.py
group: src_vae/others
loc: 256
tags: [code, src_vae, runnable]
---

# model_to_config

> Convert PyTorch model checkpoints to YAML experiment configs.

**Source:** `src_vae/others/model_to_config.py` · 256 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python src_vae/others/model_to_config.py`

## Purpose

```text
Convert PyTorch model checkpoints to YAML experiment configs.

Run: ``python src_vae/others/model_to_config.py`` (edit ``INPUT_FILE``/``OUTPUT_FILE`` at top)
```

## Constants

| Name | Value |
|------|-------|
| `INPUT_FILE` | `'path/to/model.pt'` |
| `OUTPUT_FILE` | `'config.yaml'` |
| `EPOCH` | `None` |
| `BATCH_SIZE` | `None` |
| `LEARNING_RATE` | `None` |
| `NUM_EPOCHS` | `None` |
| `DESCRIPTION` | `None` |

## Functions

- **`extract_model_architecture(model: torch.nn.Module)`** — Automatically extract architecture information from a PyTorch model
- **`load_checkpoint(file_path: str)`** — Load model or checkpoint from file
- **`model_to_config_generic(input_path: str, output_path: str, additional_config: Optional[Dict[str, Any]]=None)`** — Generic function to convert model/checkpoint to config YAML
- **`print_config_summary(config: Dict[str, Any])`** — Print a formatted summary of the configuration
- **`main()`** — Convert model/checkpoint to YAML configuration using top-level parameters

## External dependencies

`torch`, `yaml`
