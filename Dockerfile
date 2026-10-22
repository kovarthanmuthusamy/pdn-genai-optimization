# syntax=docker/dockerfile:1
# Images of the current model (thesis_code/ + assistant_web/). The weights and the dataset are mounted at run
# time, never baked in. Targets:
#   serve (default)  the design-assistant web app + REST API on port 8050
#   train            phase-1 training of the surrogate (thesis_code.s07_training.train)
#   test             the data-free test suite (what CI runs)
# Build:  docker build -t pdn-assistant .                                   (CUDA 11.8 wheels)
#         docker build -t pdn-assistant:cpu --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .
ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim AS base
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.7.1 --index-url "${TORCH_INDEX_URL}" && pip install -r requirements.txt
COPY thesis_code/ thesis_code/
COPY assistant_web/ assistant_web/
RUN useradd --create-home --uid 1000 app \
    && mkdir -p assistant_web/cache thesis_code/runs \
    && chown -R app:app assistant_web/cache thesis_code/runs

FROM base AS test
RUN pip install pytest==9.1.1 httpx==0.28.1
COPY pyproject.toml ./
COPY tests/ci/ tests/ci/
USER app
CMD ["python", "-m", "pytest", "tests/ci", "-q", "-p", "no:cacheprovider"]

FROM base AS train
USER app
# Mount the dataset at /app/datasets/pdn_k1-10_balanced_v2 (settings.DATA_DIR); outputs go to THESIS_CODE_OUTPUT_DIR.
ENV THESIS_CODE_OUTPUT_DIR=/app/thesis_code/runs
ENTRYPOINT ["python", "-m", "thesis_code.s07_training.train"]

FROM base AS serve
USER app
# Mount the weights at /app/experiments/exp096_k1k10_balanced30k_ep700/checkpoints (settings.WEIGHTS_DIR).
ENV ASSIST_WEB_HOST=0.0.0.0 ASSIST_WEB_PORT=8050 ASSIST_WEB_GPUS=0,1
EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/api/health', timeout=4)"
CMD ["python", "-m", "assistant_web.server"]
