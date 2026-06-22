import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FREQ_PATH = "configs/Frequency_data_hz.npy"
target_imp_path = "configs/target_impedance.npy"

def read_real_impedance_csv(csv_path):
    """Reads real impedance from CAD CSV (handles header and column detection)."""
    # Find the line number where the actual data starts (look for '(Hz),(Ohm)')
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith('(Hz),(Ohm)'):
            data_start = idx + 1
            break
    else:
        raise ValueError("Could not find data header in CSV.")
    df = pd.read_csv(csv_path, skiprows=data_start, names=["Hz", "Ohm"])
    df = df.dropna()
    return df["Hz"].values, df["Ohm"].values

def read_generated_impedance_npy(npy_path):
    """Reads VAE-generated impedance from .npy file."""
    return np.load(npy_path).flatten()

def plot_impedance(frequency, real_imp_freq, real_imp,target_imp, gen_imp, out_path=None, show=True):
    plt.figure(figsize=(10, 6))
    plt.loglog(real_imp_freq, target_imp, "r--", linewidth=2, label="Real (CAD)")
    plt.loglog(real_imp_freq, real_imp, "g--", linewidth=2, label="Real (CAD)")
    plt.loglog(frequency, gen_imp, color="#1565C0", linewidth=2, label="VAE Generated")
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Impedance (Ohm)", fontsize=12)
    plt.title("Impedance Comparison", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend(fontsize=11)
    plt.ylim(1e-3, 1e2)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {out_path}")
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    npy_path = "data/latent_runs/latent_opt_checkpoint_epoch_400_20260420_120119/K10/batch/seed000/best_impedance_ohm.npy"  # <-- set this
    csv_path = "Real_Imp_seed001.csv"  # <-- set this
    out_path = "impedance_comparison.png"

    frequency = np.load(FREQ_PATH)
    real_imp_freq, real_imp = read_real_impedance_csv(csv_path)
    gen_imp = read_generated_impedance_npy(npy_path)
    target_imp = read_generated_impedance_npy(target_imp_path)

    plot_impedance(frequency, real_imp_freq, real_imp,target_imp= target_imp,gen_imp=gen_imp, out_path=out_path, show=True)
