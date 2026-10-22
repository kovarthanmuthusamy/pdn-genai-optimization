# Developer shortcuts. Python environment: see README (Quick start).
PY ?= python

.PHONY: help install test test-full lint serve mlflow docker docker-test

help:            ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:         ## runtime + dev + tracking dependencies (torch from the matching PyTorch index first)
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e ".[dev,tracking]"

test:            ## data-free suite (CPU, no weights) - what CI runs
	$(PY) -m pytest tests/ci -q

test-full:       ## everything: equivalence tests (dataset) + web-app tests (weights, GPU)
	$(PY) -m pytest tests/ci tests/thesis_code tests/assistant_web -q

lint:            ## ruff, rules in pyproject.toml
	ruff check thesis_code assistant_web tests/ci tools/mlflow_log_run.py

serve:           ## the design-assistant web app on http://localhost:8050
	$(PY) -m assistant_web.server

mlflow:          ## log the thesis training run to mlflow.db and open the MLflow UI
	$(PY) tools/mlflow_log_run.py && mlflow ui --backend-store-uri sqlite:///mlflow.db

docker:          ## build the serving image (CUDA 11.8)
	docker build -t pdn-assistant .

docker-test:     ## build the CPU test image and run the data-free suite in it
	docker build --target test --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t pdn-assistant:test . && docker run --rm pdn-assistant:test
