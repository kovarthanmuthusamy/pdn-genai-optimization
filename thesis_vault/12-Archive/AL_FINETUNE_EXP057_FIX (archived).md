---
title: AL_FINETUNE_EXP057_FIX (archived)
type: archive
source: docs/_archive/AL_FINETUNE_EXP057_FIX.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# AL Fine-Tune exp057 — Runtime Fix (iter 2)

### 📝 Summary of Changes

- **`train_vae_simple.py` — `_apply_yaml_exp055`**: Now loads `VAE_CONFIG_PATH` (runtime AL finetune yaml) after `config.yaml`, so `num_epochs=1050`, `al_overlay_data_dir`, and other finetune overrides apply.
- **`train_vae_simple.py` — resume banner**: RESUMING message reads target `num_epochs` from `VAE_CONFIG_PATH` instead of the stale value stored in the checkpoint (was showing `1001–1000`).
- **`train_core.py`**: Skip `last_model.pt` save when no validation metrics exist (guards against empty training loops).
- **`dataloader_base.py` — `multifreq_collate_fn`**: Only collate `sample_idx` when **all** batch rows have it, fixing mixed batches of base train + AL overlay (`KeyError: 'sample_idx'`).

### 🚀 Implementation Details

**Root cause of `KeyError: 'modality_stats'`**

1. AL fine-tune sets `VAE_CONFIG_PATH` → `config_al_finetune.runtime.yaml` (`num_epochs=1050`).
2. `train_vae_simple._apply_yaml_exp055` only loaded `config.yaml` (`num_epochs=1000`), ignoring the runtime override.
3. Training loop `range(1000, 1000)` ran zero epochs → `val={}` → crash in `build_latent_stats(val)`.

**Root cause of `KeyError: 'sample_idx'`**

- Train loader uses `ConcatDataset([_IndexedSubset(base), overlay_ds])` with `WeightedRandomSampler`.
- Base rows include `sample_idx`; AL overlay rows do not.
- Collate checked only `batch[0]` and failed when the first row had `sample_idx` but a later overlay row did not.

**Verification**

- Config load: `epochs 1050`, `overlay datasets/data_multifreq_al_overlay_exp057`
- Data log: `al_overlay(w=40.0) al_overlay=24`
- Resume banner: `training epochs 1001–1050`
- Training progressed: `Ep 1001/1050`, `Ep 1002/1050`, `Ep 1003/1050` (~87 s/epoch)

### 🛠️ Verification & Execution Results

```bash
python pipelines/active_learning/finetune_exp057.py
# or: COMMAND = "finetune" in pipelines/active_learning/run.py
```

- **Status**: Fine-tune started successfully after fixes; 50 epochs (1001→1050) running in background.
- **Log**: `/tmp/finetune_exp057_run.log` (or terminal output from `finetune_exp057.py`).
- **Expected duration**: ~70–90 minutes for 50 epochs at ~87 s/epoch.
