# active_learning_pi/

Active-learning library and configuration for the PI multifreq VAE loop.

| Path | Role |
|------|------|
| [`al/`](al/) | Library modules (pipeline, inference, ingest, normalize, acquisition, …) |
| [`config/`](config/) | Default JSON config (`default.json`) |
| [`../pipelines/active_learning/run.py`](../pipelines/active_learning/run.py) | **Entry script** — edit CONFIG, then run |

## Run

```bash
python pipelines/active_learning/run.py
```

Set `COMMAND` in the CONFIG block at the top of `run.py` (`full`, `cycle`, `finetune`, …). See `docs/AL_OPTION_B_FINETUNE_EXP057.md`.

Do not run `al/pipeline.py` directly; it delegates to `run.py`.
