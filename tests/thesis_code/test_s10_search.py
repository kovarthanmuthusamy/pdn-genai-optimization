import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.thesis_code.helpers import EXP, EXP_NAME, REPO, assert_close, assert_same_json

COMMON = {"MASTER_SEED": "20261008", "SWEEP_SEED_BASE": "90000", "NUM_STEPS": "40", "NUM_SEEDS": "8", "K_MAX": "3"}
ENV_ORIG = {"LATENT_OPT_EXPERIMENT": EXP_NAME, "LATENT_OPT_TARGET_PATH": "configs/target_impedance_0.6.npy",
            "LATENT_OPT_RECHECK_HEAD": "H1_spec_hm200", "LATENT_OPT_CLAIM_MODE": "both",
            **{f"LATENT_OPT_{k}": v for k, v in COMMON.items()}}
ENV_NEW = {"SEARCH_TARGET_PATH": str(REPO / "thesis_code/common/inputs/target_impedance_0.6.npy"),
           **{f"SEARCH_{k}": v for k, v in COMMON.items()}}


def _run(cmd, env):
    p = subprocess.run(cmd, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **env},
                       capture_output=True, text=True, timeout=3600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return Path(re.search(r"Run directory: (\S+)", p.stdout).group(1))


def test_search_matches_original(out_dir):
    """Same target, seeds and step count on the CPU: same proposals and claims at every budget."""
    a = _run([sys.executable, "pipelines/latent/optimize.py"], ENV_ORIG)
    try:
        b = _run([sys.executable, "-m", "thesis_code.s10_search.search"], {**ENV_NEW, "THESIS_CODE_OUTPUT_DIR": str(out_dir)})
        assert str(b).startswith(str(out_dir))
        kdirs = sorted(p.name for p in a.iterdir() if p.is_dir())
        assert kdirs == ["K02", "K03"] and kdirs == sorted(p.name for p in b.iterdir() if p.is_dir())
        for kdir in kdirs:
            names = sorted(f.name for f in (a / kdir).iterdir())
            assert names == sorted(f.name for f in (b / kdir).iterdir()), kdir
            for name in names:
                f, g = a / kdir / name, b / kdir / name
                if f.suffix == ".npy":
                    assert_close(np.load(f), np.load(g), f"{kdir}/{name}")
                elif f.suffix == ".json":
                    assert_same_json(json.loads(f.read_text()), json.loads(g.read_text()), f"{kdir}/{name}")
        assert (a / "candidate_seeds.txt").read_text() == (b / "candidate_seeds.txt").read_text()
    finally:
        assert a.parent == REPO / "data/latent_runs" / EXP_NAME and a.name.isdigit(), a
        shutil.rmtree(a)                       # the original's own new run folder, made by this test


def test_latent_stats_match_saved(out_dir):
    """Per-K mean and covariance of the training latents equal the saved file (built on the GPU:
    arrays compared to 1e-6)."""
    from thesis_code.s10_search import latent_stats
    latent_stats.main()
    new = json.loads((out_dir / "s10_search/latent_cov_per_k.json").read_text())
    old = json.loads((EXP / "checkpoints/latent_cov_per_k.json").read_text())
    assert sorted(new) == sorted(old)
    for k in old:
        assert new[k]["n"] == old[k]["n"], k
        assert_close(np.array(new[k]["mu"]), np.array(old[k]["mu"]), f"K={k} mu")
        assert_close(np.array(new[k]["cov"]), np.array(old[k]["cov"]), f"K={k} cov")
