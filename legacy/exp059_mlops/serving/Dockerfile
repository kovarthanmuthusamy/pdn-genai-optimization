# FastAPI inference server for exp059 surrogate
# Usage:
#   docker build -t genai-pdn-serve .
#   docker run --rm -p 8000:8000 \
#       -v /path/to/checkpoints:/app/checkpoints:ro \
#       -e CUDA_VISIBLE_DEVICES=0 \
#       genai-pdn-serve

FROM pytorch/pytorch:2.7.1-cuda12.1-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement specs (minimal for serving only)
COPY requirements-serving.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy repo code
COPY repo_paths.py .
COPY experiments/exp059_capacity_freq ./experiments/exp059_capacity_freq
COPY src_vae ./src_vae
COPY libs ./libs
COPY configs ./configs
COPY serving ./serving

# Model checkpoint (mount at runtime or copy here)
RUN mkdir -p experiments/exp059_capacity_freq/checkpoints

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# Expose API port
EXPOSE 8000

# Start FastAPI server
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "serving.app"]
