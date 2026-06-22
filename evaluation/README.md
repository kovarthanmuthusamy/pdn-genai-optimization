# Evaluation

Evaluation code and run artifacts for VAE quality and novelty checks.

## Novelty (memorization / nearest-neighbour)

- Scripts: `evaluation/novelty/scripts/`
- Outputs: `evaluation/novelty/runs/`

Edit the CONFIG block at the top of each script, then run:

```bash
python evaluation/novelty/scripts/run_vae_novelty_test.py
python evaluation/novelty/scripts/run_vae_novelty_sweep.py
python evaluation/novelty/scripts/vae_novelty_report.py
python evaluation/novelty/scripts/summarize_novelty_sweep.py
```

Legacy sample folders may appear under `evaluation/novelty/runs/legacy_generated_samples/`.

## VAE reconstruction & latent diagnostics

Edit CONFIG in `evaluation/vae/run_vae_eval.py`, then:

```bash
python evaluation/vae/run_vae_eval.py
```

Default outputs under `evaluation/vae/`:

- `vae_eval_report.md`
- `plots/*.png`
- `vae_eval_metrics.json`
- `vae_eval_per_sample.csv`

### Held-out full test

After building a held-out dataset with `pipelines/data/processing_eval.py` and normalizing it, edit CONFIG in `run_vae_eval_heldout.py`:

```bash
python evaluation/vae/run_vae_eval_heldout.py
```

Set `CHECKPOINT`, `DATASET_ROOT`, and `OUT_DIR` in the CONFIG block.
