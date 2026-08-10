FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# Copy only pyproject.toml (for layer caching)
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Copy repo (excluding large artifacts via .dockerignore)
COPY . .

# Default entrypoint: training on a single GPU
ENTRYPOINT ["python", "-m", "experiments.exp059_capacity_freq.codes.train_vae_simple"]
