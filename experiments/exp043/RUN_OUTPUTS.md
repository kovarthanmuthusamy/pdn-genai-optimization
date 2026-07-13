## exp043: Preserve outputs with per-run directories

### What changed

`experiments/exp043/codes/train_vae_simple.py` no longer deletes existing run outputs when starting a **fresh** training run.

Instead, every fresh start creates a new run directory:

- `experiments/exp043/runs/run_<UTC timestamp>/`
  - `checkpoints/`
  - `logs/`
  - `metrics/` (including `metrics/plots/`)

Resume runs continue to write under the chosen `experiment_dir` (i.e. the run directory you resume from).

### Why

- Keeps a complete history of training runs for comparison/debugging
- Avoids accidental loss of checkpoints and metrics

### How to use

- **Fresh run**: just start training as usual. The script will print the new run path.
- **Resume**: set `resume_checkpoint` (or point `experiment_dir` to a specific run directory) and restart training.

