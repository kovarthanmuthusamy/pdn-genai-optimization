#!/usr/bin/env python3
"""Entry point: python active_learning_pi/run_pipeline.py iteration"""

from __future__ import annotations

import sys
from pathlib import Path

_GAN_ROOT = Path(__file__).resolve().parents[1]
if str(_GAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GAN_ROOT))

from active_learning_pi.al.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
