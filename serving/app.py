"""FastAPI inference server for exp059 surrogate.

Serves occupancy + K + frequency → spectrum + heatmap predictions.

Usage:
    python -m serving.app                           # dev server
    uvicorn serving.app:app --host 0.0.0.0 --port 8000  # production
"""
import os
import sys
import torch
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Bootstrap repo imports before any relative imports
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_paths import setup_path, resolve_repo_path
setup_path()

from experiments.exp059_capacity_freq.codes.inference_vae import VAEInference
from src_vae.others.pi_freq_utils import pi_freq_norm_for_model
from serving.schemas import PredictRequest, PredictResponse


# Global model instance (loaded once on startup)
_model_instance: Optional[VAEInference] = None


def load_model(checkpoint_path: Optional[str] = None) -> VAEInference:
    """Load trained model checkpoint."""
    global _model_instance

    if checkpoint_path is None:
        checkpoint_path = os.environ.get(
            "VAE_CHECKPOINT_PATH",
            "experiments/exp059_capacity_freq/checkpoints/last_model.pt"
        )

    checkpoint_path = resolve_repo_path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model_instance = VAEInference(
        checkpoint_path=str(checkpoint_path),
        device=device
    )
    print(f"✅ Model loaded from {checkpoint_path} on device {device}")
    return _model_instance


def get_model() -> VAEInference:
    """Get or load the model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = load_model()
    return _model_instance


# Create FastAPI app
app = FastAPI(
    title="GenAI PDN Surrogate",
    description="Occupancy → impedance spectrum + heatmap inference via exp059 VAE",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on server startup."""
    try:
        load_model()
    except Exception as e:
        print(f"⚠ Warning: Model failed to load on startup: {e}")
        print("  Model will be loaded on first prediction request.")


@app.get("/health")
async def health():
    """Health check endpoint."""
    model_loaded = _model_instance is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(_model_instance.device) if model_loaded else "not_loaded"
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict impedance spectrum + heatmap from occupancy.

    Given a decap placement (occupancy vector) and inspection frequency,
    returns the predicted impedance spectrum and physical heatmap.

    Args:
        occupancy: Binary vector [52], exactly K ones
        K: Decap budget (sum of occupancy)
        frequency_mhz: Frequency in MHz (1-600)

    Returns:
        spectrum: Log impedance [231] (1-600 MHz)
        heatmap: Physical impedance Ω [64, 64]
        metadata: Request metadata
    """
    try:
        engine = get_model()

        # Validate occupancy
        occ_list = request.occupancy
        occ_sum = sum(occ_list)
        if abs(occ_sum - request.K) > 0.01:
            raise ValueError(
                f"Occupancy sum {occ_sum} does not match K {request.K}"
            )

        # Convert to tensors
        device = engine.device
        occ_tensor = torch.tensor(
            [occ_list],  # Batch size 1
            dtype=torch.float32,
            device=device
        )  # [1, 52]

        K_tensor = torch.tensor(
            [request.K],
            dtype=torch.long,
            device=device
        )  # [1]

        # Frequency normalization
        PI_freq_tensor = pi_freq_norm_for_model(
            request.frequency_mhz,
            batch_size=1,
            unit="mhz",
            device=device,
            dtype=torch.float32
        )  # [1]

        # Encode occupancy to latent (deterministic, no sampling)
        with torch.no_grad():
            z = engine.model.encode_occupancy_latent(
                occ_tensor,
                K_tensor,
                PI_freq_tensor,
                sample=False  # Use mean, not sample
            )

            # Decode to outputs
            hm_z, _, imp_norm = engine.model.decode(
                z,
                K_tensor,
                PI_freq_tensor,
                occupancy=occ_tensor
            )

            # Denormalize to physical units
            spectrum_log = engine._denorm_impedance(imp_norm)  # [1, 1, 231]
            heatmap_phys = engine.denorm_heatmap_physical(
                hm_z,
                mhz=request.frequency_mhz
            )  # [1, 1, 64, 64]

        # Extract numpy arrays and flatten to lists
        spectrum = spectrum_log[0, 0, :].cpu().numpy().tolist()  # [231]
        heatmap_np = heatmap_phys[0, 0, :, :].cpu().numpy()  # [64, 64]
        heatmap = heatmap_np.tolist()

        return PredictResponse(
            spectrum=spectrum,
            heatmap=heatmap,
            metadata={
                "occupancy_sum": int(occ_sum),
                "K": request.K,
                "frequency_mhz": request.frequency_mhz,
                "device": str(device)
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e):
            raise HTTPException(
                status_code=503,
                detail=f"GPU error: {str(e)}"
            )
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(e).__name__}: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level=os.environ.get("LOG_LEVEL", "info")
    )
