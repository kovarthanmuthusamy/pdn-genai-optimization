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

Set `COMMAND` in the CONFIG block (`cycle`, `generate`, `infer`, `simulate`, …). Use `CONFIG_PATH = None` to load `active_learning_pi/config/default.json`.

Do not run `al/pipeline.py` directly; it delegates to `run.py`.
