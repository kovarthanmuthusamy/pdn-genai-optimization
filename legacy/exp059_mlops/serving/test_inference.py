"""Quick test of the inference service endpoints.

Usage:
    python serving/test_inference.py

Requires:
    - FastAPI and uvicorn installed
    - Model checkpoint available
    - Server running on localhost:8000
"""
import asyncio
import json
import sys
from pathlib import Path

# Bootstrap repo
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)


async def test_health():
    """Test health endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            print(f"✓ Health check: {response.status_code}")
            print(f"  {json.dumps(response.json(), indent=2)}")
            return response.status_code == 200
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False


async def test_predict():
    """Test prediction endpoint."""
    # 26 decaps at positions 26-51
    occupancy = [0.0] * 26 + [1.0] * 26

    request_data = {
        "occupancy": occupancy,
        "K": 26,
        "frequency_mhz": 200.0
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/predict",
                json=request_data,
                timeout=30.0
            )
            print(f"✓ Prediction: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"  Spectrum shape: {len(result['spectrum'])}")
                print(f"  Heatmap shape: {len(result['heatmap'])}x{len(result['heatmap'][0])}")
                print(f"  Spectrum range: [{min(result['spectrum']):.4f}, {max(result['spectrum']):.4f}]")

                hm_flat = [v for row in result['heatmap'] for v in row]
                print(f"  Heatmap range: [{min(hm_flat):.4f}, {max(hm_flat):.4f}] Ω")
                print(f"  Metadata: {result['metadata']}")
                return True
            else:
                print(f"  Error: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            return False


async def main():
    """Run all tests."""
    print("Testing exp059 inference service...\n")

    print("1. Health check")
    health_ok = await test_health()
    print()

    if health_ok:
        print("2. Prediction endpoint")
        pred_ok = await test_predict()
        print()

        if pred_ok:
            print("✓ All tests passed!")
            return 0
        else:
            print("✗ Prediction test failed")
            return 1
    else:
        print("✗ Server not responding. Is it running?")
        print("  Start server with: python -m serving.app")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
