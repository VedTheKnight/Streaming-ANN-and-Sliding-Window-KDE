import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Plot memory vs N for StreamingANN")
parser.add_argument("--csv", type=str, required=True, help="Path to results CSV from memory variation experiment")
parser.add_argument("--dim", type=int, required=True, help="Dimensionality of dataset (for linear memory reference)")
parser.add_argument("--outdir", type=str, default="plots", help="Directory to save plots")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# --- Load CSV ---
df = pd.read_csv(args.csv)

# Color list convention
color_list = ['b','g','r','c','m','y','orange']

# Get unique epsilons and etas
epsilons = sorted(df["epsilon"].unique())
etas = sorted(df["eta"].unique())

# Dataset dimension
d = args.dim

for eps in epsilons:
    subset = df[df["epsilon"] == eps]

    plt.figure(figsize=(10, 6))
    for k_, eta in enumerate(etas):
        eta_subset = subset[subset["eta"] == eta].sort_values("N")
        if not eta_subset.empty:
            plt.plot(
                eta_subset["N"].to_numpy(),
                eta_subset["memory_MB"].to_numpy(),
                marker='*', mec='black', linestyle='-',
                color=color_list[k_ % len(color_list)],
                lw=1.75,
                label=f"eta={eta}"
            )

    # Add theoretical linear memory curve in black
    N_vals = np.sort(subset["N"].unique())
    mem_theory = (N_vals * d * 4) / (1024*1024)
    plt.plot(
        N_vals, mem_theory,
        linestyle='--', color='black', lw=2.0, label="Linear Sketch"
    )

    plt.xlabel("N")
    plt.ylabel("Sketch Size (MB)")
    plt.title(f"Sketch Size vs N (epsilon={eps})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    out_file = os.path.join(args.outdir, f"memory_vs_N_eps{eps}.pdf")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")
