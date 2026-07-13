# Active-learning Option B — exp057 full loop

## One command (recommended)

Edit `pipelines/active_learning/run.py` CONFIG block, then:

```bash
python pipelines/active_learning/run.py
```

Default: `COMMAND = "full"`.

Runs **all 7 steps in sequence**:

| Step | Action |
|------|--------|
| 1 | Generate 96 candidates (random layout × MHz) |
| 2 | MC infer → uncertainty score |
| 3 | Select **worst 16** → build PEB |
| 4 | ECADStar simulate (16 PI jobs) |
| 5 | Ingest + normalize + evaluate |
| 6 | Build overlay dataset (cumulative) |
| 7 | Fine-tune exp057: **resume `last_model.pt` → +50 epochs** |

## Config (`active_learning_pi/config/exp057.json`)

| Key | Value | Notes |
|-----|-------|-------|
| `simulate_batch_size` | **16** | ECAD layouts per cycle |
| `num_candidates` | **96** | Pool to score before selection |
| `finetune.checkpoint_path` | `last_model.pt` | Always resume latest |
| `finetune.extra_epochs` | **50** | e.g. epoch 1000 → 1050 |

Runtime writes `config_al_finetune.runtime.yaml` with correct `num_epochs` from checkpoint.

## Other commands

Edit `COMMAND` in `pipelines/active_learning/run.py`:

| COMMAND | Effect |
|---------|--------|
| `full` | All 7 steps (default) |
| `cycle` | Steps 1–6, no fine-tune |
| `finetune` | Overlay + train only |
| Set `PROPOSE_ONLY = True` | Generate + infer only (no ECAD) |

Fine-tune standalone: `python pipelines/active_learning/finetune_exp057.py`

## Resume behaviour

- Main training `config.yaml`: `resume_checkpoint: last_model.pt` (epoch 1000)
- Each AL fine-tune: reads epoch from `last_model.pt`, trains to **epoch + extra_epochs**
- After fine-tune completes, `last_model.pt` updates → next `full` run continues from new epoch
