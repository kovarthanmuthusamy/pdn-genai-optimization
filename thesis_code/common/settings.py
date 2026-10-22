"""Where the code finds its external inputs and where it writes.

What: the only file with absolute locations. The dataset, the trained weights and the eCADSTAR
    project are not part of this folder; edit the paths below to point at them.
Config keys: DATA_DIR, WEIGHTS_DIR, CHECKPOINT, HEAD, INDEPENDENT_DIR, SAVED_RESULTS_DIR,
    SIM_STORE_DIR, PLAN_DIR, RAW_SIM_ROOT, OUTPUT_DIR, ECADSTAR_RUN, ENGINEER_EXE, ERF_PATH, EMC_DIR.
"""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]                 # thesis_code/
REPO = PACKAGE.parent

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
DATA_DIR = REPO / "datasets/pdn_k1-10_balanced_v2"            # raw/, normalized/, test_pool/, split lists, indexes
WEIGHTS_DIR = REPO / "experiments/exp096_k1k10_balanced30k_ep700/checkpoints"
CHECKPOINT = "last_model.pt"                                  # the surrogate after epoch 700
HEAD = "H1_spec_hm200"                                        # placement-only encoder, WEIGHTS_DIR/vae_route_finetune/
INDEPENDENT_DIR = REPO / "experiments/exp093_k1k10_balanced/unseen_sweep_rand10"   # the 480 independent placements
SAVED_RESULTS_DIR = REPO / "experiments/exp096_k1k10_balanced30k_ep700/evals"      # results quoted in Chapter 4 (read only)
SIM_STORE_DIR = REPO / "experiments/exp096_k1k10_balanced30k_ep700"                # eCADSTAR results of Chapter 4 (read only)
PLAN_DIR = REPO / "data/heatmaps"                             # inputs of the sampling plan (read only)
RAW_SIM_ROOT = Path("/mnt/c/Users/muthusamy/Desktop/PDN_raw_data")   # raw eCADSTAR outputs (Raw_balance30k_new, ...)
OUTPUT_DIR = Path(os.getenv("THESIS_CODE_OUTPUT_DIR", PACKAGE / "runs"))   # every output of this folder

ECADSTAR_RUN = False                                          # True only on the eCADSTAR host when no other job runs
ENGINEER_EXE = r"C:\Program Files\eCADSTAR\eCADSTAR 2023.0\Analysis\bin\engineer.exe"
ERF_PATH = r"C:\Users\muthusamy\Desktop\design\H-shape.emc\H-shape.erf"
EMC_DIR = r"C:\Users\muthusamy\Desktop\design\H-shape.emc"
# =============================================================================
