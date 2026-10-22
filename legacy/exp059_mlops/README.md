# legacy/exp059_mlops — the first MLOps stack (exp059 model)

These files were the MLOps setup built in August 2026 around the earlier `exp059_capacity_freq` model:

| File | What it was |
|---|---|
| `Dockerfile` | CUDA 11.8 training image for exp059 |
| `Dockerfile.ci` | CPU smoke-test image |
| `ci.yml` | the GitHub Actions workflow (lint, tests, Docker build) |
| `serving/` | FastAPI `/predict` service for the exp059 surrogate |
| `requirements-serving.txt` | its dependencies |
| `tests/` | the seven exp059 tests (config schema, loss CSVs, normalization round trip, straight-through top-K) |

They were moved here on 2026-10-22 when the stack was rebuilt for the current model (`thesis_code/`
and the design-assistant web app): see the root `README.md`, `Dockerfile`, `.github/workflows/ci.yml`
and `tests/ci/`. The files are kept unchanged for reference; they still point at `experiments/exp059_*`
and are not run by the new CI.
