"""Run the latent search of Chapter 4: the main runs, their repeats and the variants.

What: starts s10_search.search once per (run label, target spectrum) and records where each run
    landed in ch4_search_runs.json, so that check_search_ecad can simulate the proposals. Only the
    SEARCH_* environment settings of the search are set; its defaults are the thesis configuration
    (K = 2..10, 64 starting points from a pool of 1,000 draws, 1,200 steps).
    Run labels:
      s0, s1, s2   the search as described in Chapter 3, three runs that differ in the starting points
                   (master seed) and in the pool of 1,000 draws (sweep seed);
      nograd       as s0 without gradient steps: the 64 best draws of the pool are decoded and judged
                   as they are (one evaluation, no update). Baseline: what the optimization adds;
      noste        as s0 with the straight-through gradient of the read-out switched off: the decoded
                   placement is a constant for the gradient, which then reaches the latent vector only
                   through the spectrum decoder's direct input;
      n8, n16, n32 as s0 with 8, 16 and 32 starting points (s0 has 64).
    A run counts as finished when ch4_search_runs.json of this output folder points into
    OUTPUT_DIR; runs listed there from the saved results are run again. One run per GPU at a time.
Usage: python -m thesis_code.s12_evaluation.run_search     (CONFIG-only; GPU; needs the head's margin
       file and latent_cov_per_k.json in WEIGHTS_DIR)
Outputs: OUTPUT_DIR/s10_search/run_<n>/ per run; ch4_search_runs.json; logs/ch4_search_<label>_<target>.log
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from thesis_code.common import settings
from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
LABELS = ["s0", "s1", "s2", "nograd", "noste", "n8", "n16", "n32"]
TARGETS = ("0.8", "0.6", "0.5")
SEEDS = {"s0": (20261008, 90000), "s1": (20261009, 100000), "s2": (20261010, 110000)}   # master seed, sweep seed
VARIANTS = {"nograd": {"SEARCH_NUM_STEPS": "1"}, "noste": {"NO_STE": "1"},
            "n8": {"SEARCH_NUM_SEEDS": "8"}, "n16": {"SEARCH_NUM_SEEDS": "16"}, "n32": {"SEARCH_NUM_SEEDS": "32"}}
GPUS = [0, 1]
RUNS_FILE = "ch4_search_runs.json"
# =============================================================================

_NO_STE = """
import thesis_code.s10_search.search as s
s._ste_topk = lambda occ_prob, K: s._topk_occ(occ_prob, K)      # hard read-out, no gradient to the decoder output
print("read-out without the straight-through gradient", flush=True)
s.main()
"""


def _start(label: str, target: str, gpu: int) -> tuple[subprocess.Popen, Path]:
    master, sweep = SEEDS.get(label, SEEDS["s0"])
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu),
           "THESIS_CODE_OUTPUT_DIR": str(settings.OUTPUT_DIR),
           "SEARCH_MASTER_SEED": str(master), "SEARCH_SWEEP_SEED_BASE": str(sweep),
           "SEARCH_TARGET_PATH": str(c.TARGET_PATHS[target])}
    extra = dict(VARIANTS.get(label, {}))
    no_ste = extra.pop("NO_STE", None)
    env.update(extra)
    log = c.ev_out("logs") / f"ch4_search_{label}_{target}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-c", _NO_STE] if no_ste else [sys.executable, "-m", "thesis_code.s10_search.search"]
    return subprocess.Popen(cmd, cwd=settings.REPO, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT), log


def main() -> None:
    runs_in = c.ev_in(RUNS_FILE)
    runs = json.loads(runs_in.read_text()) if runs_in.is_file() else {}
    here = Path(settings.OUTPUT_DIR).resolve()
    finished = {k for k, v in runs.items() if c.search_run_dir(v["run_dir"]).resolve().is_relative_to(here)}
    todo = [(l, t) for l in LABELS for t in TARGETS if f"{l}/{t}" not in finished]
    print(f"{len(todo)} searches to run, {len(finished)} done", flush=True)
    active: dict[int, tuple] = {}
    while todo or active:
        for gpu in GPUS:
            if gpu not in active and todo:
                l, t = todo.pop(0)
                p, log = _start(l, t, gpu)
                active[gpu] = (l, t, p, log, time.time())
                print(f"[{time.strftime('%H:%M:%S')}] {l}/{t} started on GPU {gpu}", flush=True)
        time.sleep(5)
        for gpu in list(active):
            l, t, p, log, t0 = active[gpu]
            if p.poll() is None:
                continue
            del active[gpu]
            m = re.search(r"Run directory: (\S+)", log.read_text(errors="replace"))
            if p.returncode != 0 or not m:
                print(f"[{time.strftime('%H:%M:%S')}] {l}/{t} FAILED (exit {p.returncode}); see {log}", flush=True)
                continue
            runs[f"{l}/{t}"] = {"run_dir": m.group(1), "seconds": round(time.time() - t0, 1)}
            c.save_json(c.ev_out(RUNS_FILE), runs)
            print(f"[{time.strftime('%H:%M:%S')}] {l}/{t} finished in {time.time() - t0:.0f}s -> {m.group(1)}", flush=True)
    print("all searches finished", flush=True)


if __name__ == "__main__":
    main()
