"""Generate VAE samples for every K and produce a combined ECADStar .peb file.

Edit the CONFIGURATION block, then run:
    python scrap/run_all_k.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ============================================================
# CONFIGURATION
# ============================================================
EXPERIMENT_DIR  = "experiments/exp041"          # relative to project root
# Multifreq PI heatmap sweep (1–600 MHz): use run_multifreq_heatmap_sweep.py instead.
CHECKPOINT_PATH = f"{EXPERIMENT_DIR}/checkpoints/last_model.pt"  # path to VAE checkpoint; relative to project root
OUTPUT_ROOT     = f"{EXPERIMENT_DIR}/generated_samples"                # one sub-folder per K is created here
PEB_OUT_FILE    = f"{OUTPUT_ROOT}/K1_to_K52.peb"                          # combined PEB saved alongside generated samples

K_MIN        = 10
K_MAX        = 11
NUM_SAMPLES  = 1
SHARED_TEMP  = 1.5   # higher = more diversity; 1.0 = raw posterior stats

# Optional PI frequency conditioning.
# None  → model uses its default / checkpoint-embedded frequency condition.
# int   → single frequency in MHz, applied to all K values.
# list  → one output folder per frequency; generates samples at each frequency.
#          e.g. [10, 100, 200, 500]
PI_FREQ_MHZ: int | list[int] | None = [10, 100, 200]

# Optional: path to latent-stats .npz; leave empty to use checkpoint-embedded stats
LATENT_STATS_PATH = ""

MODEL_LATENT_DIM  = 32   # hint only – checkpoint may override via embedded config

# PEB settings
POWERBUS   = "Power_GND"
FREQ       = "63e6"
COMPONENTS = "IC1_Port1"

# Optional: copy the generated PEB to a local or WSL-accessible Windows path.
# Set to None to skip.  Windows paths are auto-converted to /mnt/<drive>/... for WSL.
PEB_COPY_DEST: str | None = r"C:\Users\muthusamy\Desktop\design"

FORCE_CPU = False
# ============================================================


# ── Bootstrap: make project root importable ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Dynamic import of VAEInference from the chosen experiment ─
_exp_module = EXPERIMENT_DIR.replace("/", ".").replace("\\", ".") + ".codes.inference_vae"
VAEInference = importlib.import_module(_exp_module).VAEInference

from scrap.generation.generate_peb import generate_peb  # noqa: E402


# ── Core generation logic (inline, no dependency on generate_samples_and_peb.py) ─

def _mhz_to_norm(mhz: int) -> float:
    """Convert MHz integer to the log10-normalised [0,1] value used by the model."""
    return (np.log10(mhz * 1e6) - 5.0) / 4.0


def _generate_save(
    engine: Any,  # VAEInference – loaded dynamically via importlib
    *,
    num_samples: int,
    out_dir: Path,
    K: int,
    shared_temp: float,
    pi_freq: float | None = None,
) -> np.ndarray:
    """Generate `num_samples` samples with exactly K decaps and save to *out_dir*.

    Returns the binary occupancy array of shape (num_samples, 52).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        inference_kwargs: dict[str, Any] = dict(
            num_samples=num_samples,
            device=engine.device,
            K=K,
            latent_stats=engine.latent_stats,
            per_K_latent_stats=engine.per_K_latent_stats,
            shared_temp=shared_temp,
        )
        if pi_freq is not None:
            inference_kwargs["PI_freq"] = pi_freq
        heatmap_zscore, occ_prob, impedance_norm = engine.model.inference(**inference_kwargs)

        occ_bin = torch.zeros_like(occ_prob)
        if K > 0:
            topk_idx = occ_prob.topk(min(K, occ_prob.shape[-1]), dim=-1).indices
            occ_bin.scatter_(-1, topk_idx, 1.0)

        impedance_log = engine._denorm_impedance(impedance_norm)
        if isinstance(impedance_log, tuple):
            impedance_log = impedance_log[0]
        heatmap_physical = (
            torch.exp(heatmap_zscore * engine.hm_log_std + engine.hm_log_mean) - 1.0
        ).clamp(min=0.0)

    heatmap_zscore_np = heatmap_zscore.cpu().numpy()
    heatmap_phys_np   = heatmap_physical.cpu().numpy()
    occ_np            = occ_bin.cpu().numpy().astype(np.int8)
    imp_np            = impedance_log.cpu().numpy()

    np.save(out_dir / "occupancy.npy", occ_np)

    for i in range(num_samples):
        sample_dir = out_dir / f"data_sample_{i}"
        sample_dir.mkdir(exist_ok=True)
        np.save(sample_dir / "heatmap_zscore.npy",   heatmap_zscore_np[i])
        np.save(sample_dir / "heatmap_physical.npy", heatmap_phys_np[i])
        np.save(sample_dir / "occupancy_map.npy",    occ_np[i])
        np.save(sample_dir / "impedance_profile.npy", imp_np[i])

    return occ_np


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    if not (0 <= K_MIN <= K_MAX <= 52):
        raise SystemExit("Expected 0 <= K_MIN <= K_MAX <= 52")

    os.chdir(PROJECT_ROOT)

    device = torch.device(
        "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    engine = VAEInference(
        checkpoint_path=CHECKPOINT_PATH,
        latent_dim=MODEL_LATENT_DIM,
        device=device,
    )
    engine.load_latent_stats(LATENT_STATS_PATH or None)

    out_root  = Path(OUTPUT_ROOT)
    peb_file  = Path(PEB_OUT_FILE)
    out_root.mkdir(parents=True, exist_ok=True)
    peb_file.parent.mkdir(parents=True, exist_ok=True)

    all_occupancies: list[np.ndarray] = []
    all_peb_freqs:   list[str]        = []  # per-sample freq string; empty = single-freq mode

    # When PI_FREQ_MHZ is None → single-frequency mode (no subfolders, PEB uses FREQ="63e6").
    # When PI_FREQ_MHZ is set  → multi-frequency mode (per-freq subfolders, PEB embeds each freq).
    multi_freq_mode = PI_FREQ_MHZ is not None

    # Build list of (freq_norm, folder_tag, peb_freq_str) triples to iterate over
    if not multi_freq_mode:
        freq_runs: list[tuple[float | None, str, str]] = [(None, "", FREQ)]
    elif isinstance(PI_FREQ_MHZ, (int, float)):
        mhz = int(PI_FREQ_MHZ)
        freq_runs = [(_mhz_to_norm(mhz), f"freq_{mhz}MHz", f"{mhz}e6")]
    else:
        freq_runs = [
            (_mhz_to_norm(int(f)), f"freq_{int(f)}MHz", f"{int(f)}e6")
            for f in PI_FREQ_MHZ # type: ignore
        ]

    for freq_norm, freq_tag, peb_freq_str in freq_runs:
        freq_label = f" @ {freq_tag}" if freq_tag else ""
        for k in range(K_MIN, K_MAX + 1):
            out_dir = out_root / freq_tag / f"K{k}" if freq_tag else out_root / f"K{k}"
            print(f"\n=== K={k}{freq_label} → {out_dir} ===")
            occ = _generate_save(
                engine,
                num_samples=NUM_SAMPLES,
                out_dir=out_dir,
                K=k,
                shared_temp=SHARED_TEMP,
                pi_freq=freq_norm,
            )
            all_occupancies.append(occ)
            if multi_freq_mode:
                all_peb_freqs.extend([peb_freq_str] * len(occ))

    occupancy_all = (
        np.concatenate(all_occupancies, axis=0)
        if all_occupancies
        else np.zeros((0, 52), dtype=np.int8)
    )

    peb_file.parent.mkdir(parents=True, exist_ok=True)
    generate_peb(
        occupancy=occupancy_all,
        output_path=str(peb_file),
        powerbus=POWERBUS,
        freq=FREQ,                                        # used in single-freq mode
        components=COMPONENTS,
        per_sample_freqs=all_peb_freqs if multi_freq_mode else None,  # None = single-freq
    )

    print("\n✓ Done")
    print(f"  Samples under : {out_root}")
    if multi_freq_mode:
        print(f"  Combined PEB  : {peb_file}  ({len(occupancy_all)} configs, {len(set(all_peb_freqs))} freq(s))")
    else:
        print(f"  Combined PEB  : {peb_file}  ({len(occupancy_all)} configs, freq={FREQ})")

    # ── Optional copy to Windows/local destination ───────────────────────────
    if PEB_COPY_DEST:
        import re as _re
        import shutil as _shutil
        dest_str = PEB_COPY_DEST
        # Convert Windows path to WSL mount if needed
        if _re.match(r"^[A-Za-z]:\\\\", dest_str) or _re.match(r"^[A-Za-z]:\\", dest_str):
            from pathlib import PureWindowsPath
            win = PureWindowsPath(dest_str)
            drive = win.drive.rstrip(":").lower()
            dest_path = Path("/mnt") / drive / Path(*win.parts[1:])
        else:
            dest_path = Path(dest_str)
        if dest_path.exists() and dest_path.is_dir():
            target = dest_path / peb_file.name
            _shutil.copy2(str(peb_file), str(target))
            print(f"  Copied PEB    : {target}")
        else:
            print(f"  ⚠ PEB_COPY_DEST not found, skipping copy: {dest_path}")


if __name__ == "__main__":
    main()
