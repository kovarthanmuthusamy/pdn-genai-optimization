# active_learning_pi/

Active-learning library and configuration for the PI multifreq VAE loop.

| Path | Role |
|------|------|
| [`al/`](al/) | Library modules (pipeline, inference, ingest, normalize, acquisition, …) |
| [`config/`](config/) | JSON configs (`exp059_gp_error.json` = primary; `exp059_random.json` = control) |
| [`../pipelines/active_learning/run.py`](../pipelines/active_learning/run.py) | **Entry script** — edit CONFIG, then run |
| [`al/decision_report.py`](al/decision_report.py) | Collects cycle numericals → `DECISION_REPORT.md` (PASS/FAIL/UNCERTAIN) |

## Run

```bash
python pipelines/active_learning/run.py
```

Set `COMMAND` in the CONFIG block at the top of `run.py` (`full`, `cycle`, `finetune`, `evaluate-decision`, …). See `docs/active-learning.md`.

For residual-GP acquisition on **exp059**, point `CONFIG_PATH` at `active_learning_pi/config/exp059_gp_error.json` (default in `run.py`). Random control: `exp059_random.json`. Validate first with:

```bash
python active_learning_pi/al/validate_acquisition_ab.py
```

Every full cycle writes numerical evals and a decision report under `runs/<run_name>/DECISION_REPORT.md`. Regenerate with `COMMAND=evaluate-decision`.

See `docs/gp-error-surrogate.md`.

Do not run `al/pipeline.py` directly; it delegates to `run.py`.
