# exp059 Surrogate Inference Service

FastAPI inference server for the exp059 surrogate model. Takes occupancy placement + inspection frequency → returns predicted impedance spectrum and heatmap.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements-serving.txt
```

### 2. Start the server

```bash
# Development mode
python -m serving.app

# Production mode (via uvicorn)
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 4
```

The server will load the model checkpoint on startup from:
```
experiments/exp059_capacity_freq/checkpoints/last_model.pt
```

Or specify via environment variable:
```bash
export VAE_CHECKPOINT_PATH=/path/to/checkpoint.pt
python -m serving.app
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict with 26 decaps at 200 MHz
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "occupancy": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    "K": 26,
    "frequency_mhz": 200.0
  }'
```

### 4. View interactive docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### `GET /health`
Health check. Returns model load status and device info.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda:0"
}
```

### `POST /predict`
Predict impedance spectrum and heatmap.

**Request body:**
```json
{
  "occupancy": [list of 52 floats, exactly K ones],
  "K": 1-52,
  "frequency_mhz": 1-600
}
```

**Response:**
```json
{
  "spectrum": [list of 231 floats in log Ω],
  "heatmap": [[64x64 grid of physical Ω]],
  "metadata": {
    "occupancy_sum": 26,
    "K": 26,
    "frequency_mhz": 200.0,
    "device": "cuda:0"
  }
}
```

## Docker Deployment

### Build image

```bash
docker build -t genai-pdn-serve .
```

### Run container

```bash
# CPU only
docker run -p 8000:8000 genai-pdn-serve

# With GPU (requires nvidia-docker)
docker run --gpus all -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=0 \
  genai-pdn-serve

# With custom checkpoint
docker run -p 8000:8000 \
  -v /path/to/checkpoints:/app/checkpoints:ro \
  -e VAE_CHECKPOINT_PATH=/app/checkpoints/custom.pt \
  genai-pdn-serve
```

## Integration with Active Learning

The service implements the same inference chain as `active_learning_pi/al/inference_pool.py`:

1. **Encode**: occupancy [52] + K → latent z via `encode_occupancy_latent()`
2. **Decode**: latent z + frequency → spectrum [231] + heatmap [64,64] via `decode()`
3. **Denormalize**: model outputs → physical units (log Ω, Ω)

This matches the AL scoring pipeline exactly, ensuring predictions are compatible with fine-tuning loops.

## Model Info

- **Architecture**: MultiInputVAEPoeFreq (Product-of-Experts VAE)
- **Latent dim**: 128 (includes 40-dim heatmap-private, 32-dim frequency-conditioned)
- **Occupancy dim**: 52 (PCB decap slots)
- **Spectrum dim**: 231 (log-scale 1-600 MHz)
- **Heatmap dim**: 64×64 (physical board)
- **Frequency range**: 1-600 MHz (training: discrete anchors; inference: any value)
- **Checkpoint**: `experiments/exp059_capacity_freq/checkpoints/last_model.pt`

## Performance

- **Latency**: ~50-100ms per prediction (GPU: 10-20ms)
- **Memory**: ~2GB model + ~1GB batch buffer
- **Throughput**: ~10-50 req/s per GPU (batch=1; scales with batching)

## Troubleshooting

### Model fails to load
```
FileNotFoundError: Checkpoint not found: experiments/exp059_capacity_freq/checkpoints/last_model.pt
```
**Fix**: Ensure checkpoint exists at that path, or set `VAE_CHECKPOINT_PATH`.

### CUDA errors on inference
```
RuntimeError: CUDA out of memory
```
**Fix**: Run with smaller batches or use CPU:
```bash
CUDA_VISIBLE_DEVICES="" python -m serving.app
```

### Occupancy validation fails
```
ValidationError: "Occupancy sum 25 does not match K 26"
```
**Fix**: Ensure occupancy vector has exactly K ones.

## Development

### Run tests
```bash
pytest serving/tests/
```

### Profile inference
```bash
python -c "
from serving.app import get_model
import torch
import time

model = get_model()
occ = torch.ones(1, 52, device=model.device) / 52 * 26
K = torch.tensor([26], device=model.device)
freq = torch.tensor([0.5], device=model.device)  # normalized

t0 = time.time()
for _ in range(100):
    with torch.no_grad():
        z = model.model.encode_occupancy_latent(occ, K, freq)
        model.model.decode(z, K, freq, occupancy=occ)
print(f'100 predictions: {time.time() - t0:.2f}s')
"
```

## See Also

- **Training**: `experiments/exp059_capacity_freq/codes/train_vae_simple.py`
- **Inference engine**: `experiments/exp059_capacity_freq/codes/inference_vae.py`
- **AL integration**: `active_learning_pi/al/inference_pool.py`
- **Model architecture**: `experiments/exp059_capacity_freq/codes/vae_poe_freq.py`
