# Tier-1 MLOps Build Guide

**Project:** `genai_pdn` (PCB PDN generative surrogate — **exp059** track)  
**Audience:** you (or a future you), collaborators, and anyone evaluating whether this research repo is production-aware  
**Status:** Tier 1 is implemented in-repo. Tier 2 ideas are listed at the end but **not** built yet.

---

## How to read this document

This guide has three layers. You can stop after any of them:

1. **Big picture** — What “MLOps” means here, and why a thesis codebase needs it.
2. **The six modules** — Each tool explained as a concept first, then as concrete files and commands in *this* repo.
3. **Day-to-day workflows** — Copy-paste recipes for train, test, serve, and CI.

If a section feels abstract, skip to **“In this repo”** under that module. That block is always concrete.

---

## Table of contents

1. [What problem are we solving?](#1-what-problem-are-we-solving)
2. [What “Tier 1” means](#2-what-tier-1-means)
3. [Mental model: the ML lifecycle](#3-mental-model-the-ml-lifecycle)
4. [Module map (cheat sheet)](#4-module-map-cheat-sheet)
5. [Module 1 — Config validation (Pydantic)](#5-module-1--config-validation-pydantic)
6. [Module 2 — Automated tests (pytest)](#6-module-2--automated-tests-pytest)
7. [Module 3 — Reproducible environments (Docker)](#7-module-3--reproducible-environments-docker)
8. [Module 4 — Experiment tracking (MLflow)](#8-module-4--experiment-tracking-mlflow)
9. [Module 5 — Model serving (FastAPI)](#9-module-5--model-serving-fastapi)
10. [Module 6 — Continuous integration (GitHub Actions)](#10-module-6--continuous-integration-github-actions)
11. [How the six modules work together](#11-how-the-six-modules-work-together)
12. [Common workflows](#12-common-workflows)
13. [Portfolio / hiring framing](#13-portfolio--hiring-framing)
14. [What Tier 1 deliberately does *not* include](#14-what-tier-1-deliberately-does-not-include)
15. [Tier 2 roadmap (not built)](#15-tier-2-roadmap-not-built)
16. [File index](#16-file-index)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What problem are we solving?

A research repo naturally optimizes for **ideas**:

- try a new loss
- bump `latent_dim`
- run another AL cycle
- look at CSVs and plots

That is correct for discovery. It is incomplete for **trust**.

Without MLOps habits, common failure modes look like this:

| What happens | Why it hurts |
|---|---|
| A typo in `config.yaml` is silently ignored | You “train for days” on the wrong setup |
| A gradient trick breaks, but only under STE | Latent optimization looks fine until peak metrics collapse |
| A colleague cannot re-run your setup | CUDA / Python / package drift → “works on my machine” |
| Metrics live only in scattered CSVs | Hard to compare runs; easy to lose which config produced which plot |
| The surrogate lives only in a notebook | No clean way for another tool (or person) to query it |
| Nobody runs tests before merge | Broken math ships, then burns a GPU night |

**MLOps** here does not mean “Kubernetes and feature stores.”  
It means: *make the research loop hard to fool yourself with.*

For this project that loop is:

> **config → train (exp059) → evaluate (physical metrics / off-anchor) → maybe fine-tune (AL) → serve predictions**

Tier 1 adds guardrails around that loop.

---

## 2. What “Tier 1” means

Think of maturity levels:

| Tier | Intent | Examples |
|------|--------|----------|
| **0 — Research scripts** | Get a result once | ad-hoc YAML, manual plots, no CI |
| **1 — Industrial basics** *(this guide)* | Make results *repeatable, checkable, shareable* | typed configs, unit tests, Docker, MLflow, FastAPI, CI |
| **2 — Team / long-lived systems** | Scale collaboration & data | DVC, Hydra, orchestration, drift monitors |

Tier 1 is the minimum stack that:

- catches many bugs *before* a long train
- makes environments comparable
- turns the surrogate into something another process can call
- shows (to you and to employers) that you understand the full lifecycle, not only model code

It is **not** a claim that surrogate accuracy improved. Accuracy claims still need the project’s numerical protocol (PASS/FAIL metrics, decision ledger, equal-budget checks). Tier 1 is about *process quality*, not physical MAE.

---

## 3. Mental model: the ML lifecycle

Ignore tools for a moment. Every serious ML project cycles through these stages:

```text
  ┌─────────────┐
  │  Configure  │  What are we about to run?
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   Train     │  Fit the model (expensive)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Evaluate   │  Did it get better on the right metric?
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   Decide    │  Keep / discard / fine-tune / ship
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │    Serve    │  Let other systems use the model
  └─────────────┘
```

Around that cycle sit two *meta* concerns:

- **Environment** — same code + same deps → comparable runs  
- **Automation** — do not rely on humans remembering to run tests

Tier 1 maps onto that cycle like this:

```text
Configure ──► Module 1 (Pydantic) validates the config
Train     ──► Module 3 (Docker) can host the same env
              Module 4 (MLflow) records the run
Evaluate  ──► Module 2 (pytest) protects load-bearing math
              (plus your existing CSV / off-anchor metrics)
Decide    ──► Module 4 makes runs comparable in a UI
Serve     ──► Module 5 (FastAPI) exposes /predict
Always    ──► Module 6 (CI) re-checks on every PR
```

---

## 4. Module map (cheat sheet)

| # | Module | Concept in one sentence | Primary files |
|---|--------|-------------------------|---------------|
| 1 | **Pydantic config** | Unknown / mistyped config keys fail loudly | `experiments/exp059_capacity_freq/codes/config_schema.py` |
| 2 | **pytest** | Small automated checks for load-bearing math | `tests/` |
| 3 | **Docker** | Pin OS + Python + packages into an image | `Dockerfile`, `pyproject.toml`, `.dockerignore` |
| 4 | **MLflow** | Structured run history (params + metrics) | hooks in `train_core.py`, optional local `mlflow server` |
| 5 | **FastAPI** | HTTP API around exp059 inference | `serving/` |
| 6 | **GitHub Actions** | Lint / test / docker smoke on every PR | `.github/workflows/ci.yml` |

---

## 5. Module 1 — Config validation (Pydantic)

### Concept (plain language)

A config file is a contract: “this run will use *these* settings.”

In many research repos, configs are plain dicts / YAML. If you mistype:

```json
"bacth_size": 64
```

nothing complains. Training uses the default `batch_size` instead. You debug the *model* for a week.

**Validation** means: parse the config into a typed object, and reject anything that does not match the schema.

**Pydantic** is a Python library that builds that schema from class fields (`latent_dim: int`, `batch_size: int`, …). When you construct `TrainConfig(**data)`, it:

- checks types (string `"96"` can coerce to int; `"abc"` cannot)
- applies defaults for missing keys
- can reject or allow unknown keys (policy choice)

### Why this matters for exp059

exp059 has a large training surface: latent sizes, loss weights, curriculum fractions, AL overlay paths, DDP knobs, off-anchor eval MHz, etc. The schema currently exposes **183 typed fields** on `TrainConfig`.

Silent defaults here are expensive because one train can take many GPU-hours.

### In this repo

| Item | Detail |
|------|--------|
| Schema | `experiments/exp059_capacity_freq/codes/config_schema.py` |
| Model | `TrainConfig` (Pydantic v2 `BaseModel`) |
| Alias | `Config = TrainConfig` (training code uses `Config`) |
| Helpers | curriculum epoch mapping, YAML/JSON loaders, `vae_model_kwargs()` |
| Policy | `extra="allow"` so *runtime* hooks (`_norm_stats`, callbacks) can attach without being schema fields |

Important nuance:

- **Declared train knobs** should live on `TrainConfig` so YAML typos are caught when you validate.
- **Runtime-only objects** (tensors, callables) are intentionally allowed as extras; they are not meant to be in `config.yaml`.

Curriculum note (easy footgun): mapping `*_frac → *_epoch` is done by **explicit helpers** (`apply_curriculum_epochs`), not by a recursive `model_validator` that fights `validate_assignment=True`.

### Minimal example

```python
from experiments.exp059_capacity_freq.codes.config_schema import TrainConfig, apply_curriculum_epochs

# Config files in this project are JSON objects (often named *.yaml) with optional # comments.
cfg = TrainConfig(latent_dim=128, batch_size=96)
apply_curriculum_epochs(cfg)
print(cfg.beta_end_epoch)  # derived from num_epochs * beta_end_frac

# Type errors fail immediately:
# TrainConfig(batch_size="nope")  → ValidationError
```

Load the real experiment file:

```python
from experiments.exp059_capacity_freq.codes.config_schema import load_yaml_config, train_config_from_yaml

raw = load_yaml_config()                 # experiments/exp059_capacity_freq/config.yaml
cfg = train_config_from_yaml()           # validated + curriculum applied (when using that helper)
print(cfg.latent_dim, cfg.keep_last_n_checkpoints)
```

### How training uses it

Training entrypoint:

```bash
python -m experiments.exp059_capacity_freq.codes.train_vae_simple
```

`train_core` / `train_vae_simple` merge `config.yaml` (and optional `VAE_CONFIG_PATH` overlays for AL fine-tunes) into a `Config` instance. The point of the schema is: **invalid overlays fail at the start**, not after epoch 200.

### What this module does *not* do

- It does not prove the chosen hyperparameters are *good* — only that they are *well-formed*.
- It does not replace physical evaluation (`off_anchor_eval.csv`, decision reports).

---

## 6. Module 2 — Automated tests (pytest)

### Concept (plain language)

A **unit test** is a tiny program that checks one claim:

> “If I call function F with input X, I get property Y.”

**pytest** finds files named `test_*.py`, runs functions named `test_*`, and reports failures.

Good research tests are not “test the whole VAE.” They protect **load-bearing invariants** — things that are easy to break and expensive to notice late.

### What we chose to test (and why)

| Test file | Claim under test | Why it is load-bearing |
|-----------|------------------|------------------------|
| `tests/test_config_schema.py` | Valid configs construct; bad types raise; disk config loads | Prevents silent config drift |
| `tests/test_ste_topk.py` | STE top‑K is discrete in forward, identity in backward | Latent occupancy optimization depends on correct STE gradients |
| `tests/test_normalize_roundtrip.py` | Normalize → denormalize recovers the signal (gmax / z-score) | Wrong norm = wrong physical Ω interpretation |

STE in one paragraph:

- Forward: force a hard top‑K occupancy pattern (discrete CAD-like mask).
- Backward: pretend the function was identity so gradients can flow to continuous scores.
- If backward is wrong, “optimization” updates the wrong direction even when losses look finite.

### In this repo

```text
tests/
  conftest.py                 # repo path bootstrap (setup_path)
  test_config_schema.py
  test_ste_topk.py
  test_normalize_roundtrip.py
```

pytest is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
```

### How to run

```bash
# from repo root, with your venv active
pytest tests/ -v
pytest tests/test_config_schema.py -v
pytest tests/ -k "ste_topk" -v
pytest tests/ --collect-only
```

### How to think about expanding tests later

Prefer tests that are:

1. **Fast** (seconds, not GPU hours)
2. **Deterministic**
3. **Falsifiable** (clear pass/fail)
4. **Close to a known past bug**

Examples that fit this project’s numerical culture:

- Spearman / acquisition rank checks already have dedicated scripts under `active_learning_pi/` — those are *evaluation* tools; unit tests should stay lighter.
- Do not put “AL beats random” claims into pytest without the equal-budget protocol artifacts.

---

## 7. Module 3 — Reproducible environments (Docker)

### Concept (plain language)

Your code is only half the experiment. The other half is:

- Python version
- PyTorch build
- CUDA / cuDNN
- every library version

**Docker** packages an operating system userland + your installed dependencies into an **image**. Running a container from that image is closer to: “boot the same machine every time.”

Analogy:

- **venv** pins Python packages on *one* OS.
- **Docker image** pins packages *and* the OS layer those packages expect.

### Two files that matter

1. **`pyproject.toml`** — declares what Python packages the project needs (source of truth for installs).
2. **`Dockerfile`** — recipe that starts from a CUDA base image, installs Python, installs the project, copies the repo.

`.dockerignore` keeps the image small and buildable by excluding huge artifacts (datasets, checkpoints, plots, `.git`, etc.). Those are mounted at runtime instead of being baked into the image.

### In this repo

**Base image:** `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`  
**Default entrypoint:** train exp059

```dockerfile
ENTRYPOINT ["python", "-m", "experiments.exp059_capacity_freq.codes.train_vae_simple"]
```

**Dependency highlights** (`pyproject.toml`):

- Core: `torch==2.7.1`, `numpy==2.1.2`, `gpytorch==1.15.2`, …
- Tier-1 tools: `pydantic`, `pytest`, `mlflow`, `fastapi`, `uvicorn`

### How to use

```bash
# Build (first time is slow; layer caching helps later)
docker build -t genai-pdn:exp059 .

# Train with GPU; mount data + checkpoints (do not copy 10–80GB into the image)
docker run --gpus all \
  -v "$PWD/datasets:/app/datasets" \
  -v "$PWD/experiments/exp059_capacity_freq/checkpoints:/app/experiments/exp059_capacity_freq/checkpoints" \
  genai-pdn:exp059

# Shell into the same environment
docker run --gpus all -it --entrypoint /bin/bash genai-pdn:exp059
```

### What Docker guarantees — and what it doesn’t

| Guarantees (approximately) | Does not guarantee |
|---|---|
| Same package versions | Same *random* GPU nondeterminism unless you also pin seeds / deterministic algorithms |
| Same Python | Bit-identical floating point across all GPU architectures |
| Same training entrypoint | That your mounted dataset/checkpoint paths are correct |

Docker is about **environment parity**, not magical bitwise reproducibility on every accelerator.

---

## 8. Module 4 — Experiment tracking (MLflow)

### Concept (plain language)

While training, you already write CSVs under `metrics/` (`loss.csv`, `off_anchor_eval.csv`, …). That is good and remains the project’s detailed source for many analyses.

**Experiment tracking** adds a second, queryable layer:

- **Parameters** — what config produced the run
- **Metrics** — curves over steps/epochs
- **UI** — compare runs without opening ten folders

**MLflow** is a common open-source tracker. A “run” is one training session. You can host a local server that reads a SQLite file and shows a browser UI.

### In this repo

Training hooks live in `experiments/exp059_capacity_freq/codes/train_core.py`:

- `mlflow.start_run(...)` near train start
- `mlflow.log_params(c.model_dump())` for config snapshot
- `mlflow.log_metrics({...}, step=epoch)` for train/val losses
- `mlflow.end_run()` at the end

Active-learning fine-tunes can thread a cycle id via env:

- `active_learning_pi/al/finetune_run.py` sets `MLFLOW_CYCLE_ID` (existing value or a new UUID)

That lets you group “this AL cycle’s fine-tune” even when training is launched as a subprocess.

### How to use

```bash
# Terminal A — UI
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
# open http://127.0.0.1:5000

# Terminal B — train as usual
python -m experiments.exp059_capacity_freq.codes.train_vae_simple
```

### How to use MLflow *with* this project’s evidence rules

MLflow is for **navigation and comparison**, not for declaring thesis claims by itself.

For claims like “uncertainty ranks errors” or “GP beats random,” continue to use:

- `acquisition_rank_quality.json` / Spearman / lift
- equal-budget A/B scripts
- `DECISION_REPORT.md` / decision ledger PASS|FAIL|UNCERTAIN

You can log those JSON metrics into MLflow later; Tier 1 currently focuses on train loss curves + params.

### Metrics still on disk (do not abandon these)

Canonical exp059 layout after cleanup:

```text
experiments/exp059_capacity_freq/metrics/
  loss.csv
  epoch_timing.csv
  latent_stats.csv
  off_anchor_eval.csv
  heatmap_peak_split.csv
  impedance_split.csv
  timing.json
  plots/
```

MLflow complements this; it does not replace off-anchor physical evaluation.

---

## 9. Module 5 — Model serving (FastAPI)

### Concept (plain language)

Training produces a checkpoint. Inference code can load that checkpoint in Python. **Serving** means wrapping inference behind a stable interface — usually HTTP — so:

- another service can call it
- a UI can call it
- you can smoke-test the model without opening notebooks

**FastAPI** is a Python web framework with automatic request validation (also via Pydantic) and generated OpenAPI docs.

### Domain mapping (this project)

The surrogate answers a physical question roughly like:

> Given a decap **occupancy** pattern (52 slots), budget **K**, and an inspection **frequency**, predict the **impedance spectrum** and a **spatial heatmap**.

So the API is not “generic ML predict.” It is a **PDN surrogate predict**.

### In this repo

```text
serving/
  app.py              # FastAPI app, /health + /predict
  schemas.py          # PredictRequest / PredictResponse
  Dockerfile          # serving-oriented image
  test_inference.py
  README.md
```

**Request (`PredictRequest`):**

- `occupancy`: length‑52 floats (binary mask)
- `K`: integer 1–52 (budget; should match sum of occupancy)
- `frequency_mhz`: 1–600

**Response (`PredictResponse`):**

- `spectrum`: length‑231 (log Ω along frequency bins)
- `heatmap`: 64×64 physical Ω
- `metadata`: echo / diagnostics

The server loads `VAEInference` from exp059 once (startup or first request), using:

```text
VAE_CHECKPOINT_PATH   # optional override
default → experiments/exp059_capacity_freq/checkpoints/last_model.pt
```

### How to run

```bash
# Dev
python -m serving.app
# → http://127.0.0.1:8000
# docs: http://127.0.0.1:8000/docs

# Production-style
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 1

# Docker (serving image)
docker build -t genai-pdn-serve -f serving/Dockerfile .
docker run --gpus all -p 8000:8000 \
  -e VAE_CHECKPOINT_PATH=/models/last_model.pt \
  -v "$PWD/experiments/exp059_capacity_freq/checkpoints:/models" \
  genai-pdn-serve
```

Example request:

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "occupancy": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    "K": 26,
    "frequency_mhz": 200.0
  }'
```

### Design choices to notice

- **Same inference stack as research code** (`inference_vae.VAEInference`) — avoids a second, divergent “prod model path.”
- **Validation at the edge** — bad shapes / ranges become HTTP 422 instead of CUDA crashes.
- **Model loaded once** — request latency should not include checkpoint load every time.
- **Health endpoint** — orchestration / monitoring can probe `/health` without running a full predict.

---

## 10. Module 6 — Continuous integration (GitHub Actions)

### Concept (plain language)

**CI** means: when code changes, a robot runs a checklist in a clean environment.

Humans forget. Robots do not (unless you disable the workflow).

GitHub Actions watches events (`push`, `pull_request`) and runs jobs defined in YAML under `.github/workflows/`.

### In this repo

Workflow file: `.github/workflows/ci.yml`

Triggered on:

- push to `main`
- pull request targeting `main`

Three jobs:

| Job | What it does | Failure meaning |
|-----|--------------|-----------------|
| **lint** | `ruff check` on exp059 codes, `serving/`, `tests/` | Style / smell (currently `continue-on-error: true`) |
| **test** | install minimal deps + `pytest tests/ -v` | Load-bearing tests failed — **should block merge** |
| **docker-build** | `docker build` + `python --version` smoke | Image recipe broken |

### Local equivalents (run before pushing)

```bash
ruff check experiments/exp059_capacity_freq/codes serving/ tests/
pytest tests/ -v --tb=short
docker build -t genai-pdn:ci .
```

### How to use in a PR workflow

1. Create a branch and make changes.
2. Push and open a PR into `main`.
3. Wait for checks on the PR page.
4. If **test** fails, fix locally with `pytest`, push again.
5. Merge only when required checks are green.

---

## 11. How the six modules work together

End-to-end story for one feature change:

```text
You edit a loss weight in config.yaml
        │
        ▼
Pydantic (Module 1) rejects typos when config is loaded
        │
        ▼
pytest (Module 2) still passes STE / norm / schema tests
        │
        ▼
You train (local venv or Docker Module 3)
        │
        ├─ metrics/*.csv written as always
        └─ MLflow (Module 4) stores params + loss curves
        │
        ▼
You push a PR → GitHub Actions (Module 6) re-runs lint/tests/docker smoke
        │
        ▼
Optional: ship checkpoint behind FastAPI (Module 5) for /predict
```

ASCII “architecture”:

```text
                 ┌──────────────────────┐
                 │   GitHub Actions CI  │
                 │  lint · pytest · dock│
                 └──────────▲───────────┘
                            │ on PR
┌──────────┐   validate  ┌──┴────────┐   track   ┌─────────┐
│config.yml├────────────►│ train_core│──────────►│ MLflow  │
└──────────┘             │ + exp059  │           └─────────┘
                         └──┬────────┘
                            │ checkpoint
                            ▼
                     ┌──────────────┐
                     │ FastAPI      │
                     │ /predict     │
                     └──────────────┘

Docker image can wrap train and/or serve so envs match.
```

---

## 12. Common workflows

### A. Safe local train (research default)

```bash
cd /home/ubuntu/genai_pdn
source .venv/bin/activate   # or your venv

pytest tests/ -q
python -m experiments.exp059_capacity_freq.codes.train_vae_simple
# metrics → experiments/exp059_capacity_freq/metrics/
```

### B. Train inside Docker

```bash
docker build -t genai-pdn:exp059 .
docker run --gpus all \
  -v "$PWD/datasets:/app/datasets" \
  -v "$PWD/experiments/exp059_capacity_freq:/app/experiments/exp059_capacity_freq" \
  genai-pdn:exp059
```

### C. Compare runs in MLflow

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
# train one or more configs, then compare params/metrics in the UI
```

### D. Serve a checkpoint

```bash
export VAE_CHECKPOINT_PATH=experiments/exp059_capacity_freq/checkpoints/best_off_anchor_model.pt
python -m serving.app
# open http://127.0.0.1:8000/docs
```

### E. Scaffold a new experiment folder (layout only)

After the folder cleanup work, new experiments should use:

```bash
python experiments/folder_structure.py exp061_my_idea --from exp059_capacity_freq --no-interactive
```

That creates the canonical tree (`codes/`, `metrics/plots/`, `logs/`, `evals/sweeps/`, configs, `run_train_ddp.sh`) instead of the old `visuals/` + empty CSV layout.

---

## 13. Portfolio / hiring framing

If you describe this stack to someone quickly:

> “I took a research VAE/AL codebase and added Tier‑1 MLOps: typed configs, unit tests on STE/normalization, Dockerized train env, MLflow run tracking, a FastAPI inference service, and GitHub Actions CI.”

What that demonstrates:

| Skill signal | Evidence in repo |
|--------------|------------------|
| Config discipline | `TrainConfig` / validation errors instead of silent YAML typos |
| Testing judgment | tests target STE gradients & norm round-trips, not vanity coverage |
| Reproducibility mindset | Dockerfile + pinned deps + `.dockerignore` |
| Observability | MLflow hooks beside existing CSV metrics |
| Productization | `/predict` with schemas and health checks |
| Engineering hygiene | CI on PR |

Keep research honesty intact: **do not** imply that MLOps alone improved PDN physics metrics. Point to `metrics/off_anchor_eval.csv` and decision reports for those claims.

---

## 14. What Tier 1 deliberately does *not* include

To keep scope honest:

- No full data versioning of multi‑GB datasets
- No multi-user auth / rate limiting on the API
- No model registry promotion gates (“staging → prod”) beyond checkpoint files
- No cluster autoscaling
- No claim that CI covers GPU training (CI tests are CPU-light by design)
- Lint is currently advisory (`continue-on-error: true`) — tighten when ready

These omissions are intentional for Tier 1.

---

## 15. Tier 2 roadmap (not built)

| Idea | Problem it addresses |
|------|----------------------|
| **DVC** (or similar) | Version large datasets/checkpoints without stuffing git |
| **Hydra** | Compose config groups + clean CLI overrides |
| **Prefect / Dagster** | Orchestrate AL cycles as typed pipelines with retries |
| **Evidently / custom monitors** | Watch residual / acquisition health over time |
| **Model registry + aliases** | `prod`, `candidate` pointers instead of ad-hoc paths |
| **Stricter CI** | GPU smoke job; fail on ruff; cache Docker layers |

Build these when collaboration or data churn demands them — not before.

---

## 16. File index

| Path | Role |
|------|------|
| `experiments/exp059_capacity_freq/codes/config_schema.py` | Pydantic `TrainConfig` + helpers |
| `experiments/exp059_capacity_freq/codes/train_core.py` | Training loop + MLflow hooks |
| `experiments/exp059_capacity_freq/codes/train_vae_simple.py` | Canonical train entry |
| `experiments/exp059_capacity_freq/codes/inference_vae.py` | Research + serving inference engine |
| `tests/` | pytest suite |
| `pyproject.toml` | Dependencies + pytest config |
| `Dockerfile` | Train-oriented image |
| `.dockerignore` | Keep build context lean |
| `serving/` | FastAPI service |
| `.github/workflows/ci.yml` | CI pipeline |
| `experiments/folder_structure.py` | Scaffold new experiments in the cleaned layout |
| `active_learning_pi/al/finetune_run.py` | AL fine-tune launcher (`MLFLOW_CYCLE_ID`) |

---

## 17. Troubleshooting

### Config validation error on train start

- Read the Pydantic error: wrong type vs unknown field vs missing required value.
- Remember curriculum helpers must be applied intentionally after merges.
- AL overlays via `VAE_CONFIG_PATH` must still be schema-compatible for typed fields.

### pytest cannot import project modules

- Run from repo root.
- Ensure `tests/conftest.py` / `repo_paths.setup_path()` can see the repo.
- Use the project venv (`.venv`) that has `torch`, `pydantic`, etc.

### Docker build is huge or fails copying data

- Confirm `.dockerignore` excludes `datasets/` and checkpoints.
- Mount those directories with `-v` at runtime.

### MLflow UI shows no runs

- Confirm training process can write the tracking URI (default local `./mlruns` unless configured).
- Start `mlflow server` against the same backend store you are writing to.

### `/predict` returns 500 / model not loaded

- Check `VAE_CHECKPOINT_PATH` points to an existing `.pt`.
- Hit `/health` — `model_loaded` should become true after a successful load.
- Confirm occupancy length is 52 and `K` matches the number of ones.

### CI green locally but red on GitHub

- CI installs a **minimal** dependency set for tests; if your new test needs an unlisted package, add it to the workflow install step or to `pyproject.toml` and the CI install list.
- Docker job needs Dockerfile + context to stay valid even if you never train in CI.

---

## Closing

Tier 1 does not replace scientific rigor. It supports it.

- **Science** asks: did the surrogate / acquisition / AL loop improve the right physical quantities?  
- **MLOps Tier 1** asks: can we trust that the run we *think* we launched is the run we actually launched — and can someone else re-run, check, and call the model?

For this codebase, keep both questions visible:

- process → this document  
- physics / ranking claims → numerical artifacts and decision reports under the exp059 / AL run trees
