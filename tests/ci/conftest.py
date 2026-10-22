"""Data-free test suite: runs on a CPU-only machine with no dataset, weights or eCADSTAR (GitHub Actions).

The full suites need the trained weights and GPUs and run on the workstation:
    pytest tests/thesis_code tests/assistant_web
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
