# Evaluation

This folder is the canonical home for evaluation code and run artifacts.

## Novelty

- Code: `evaluation/novelty/scripts/`
- Run artifacts / reports: `evaluation/novelty/runs/`
- Backwards-compatible wrappers: `scripts/*.py` dispatch to the new locations.
- Legacy artifacts previously under `scrap/generated_samples/` were moved to `evaluation/novelty/runs/legacy_generated_samples/` (and `scrap/generated_samples` is kept as a symlink for compatibility).

## VAE (reconstruction + latent diagnostics)

Edit the config block at the top of `evaluation/vae/run_vae_eval.py`, then run:

```bash
python evaluation/vae/run_vae_eval.py
```

Outputs (default `evaluation/vae/`):

- `vae_eval_report.md` (compact markdown report with plots)
- `plots/*.png`
- `vae_eval_metrics.json` (aggregated metrics)
- `vae_eval_per_sample.csv` (per-sample metrics)

### Held-out full test (non-training combinations)

If you created a held-out dataset via `Data_Creation/Data_processing_eval.py`, run:

```bash
python evaluation/vae/run_vae_eval_heldout.py \
  --checkpoint experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt \
  --dataset-root datasets/data_eval_norm
```

This assumes you have already normalized the held-out dataset into a VAE-compatible dataset root
(e.g. `datasets/data_eval_norm`). If your normalized dataset is in a different location, pass
`--dataset-root`.
