import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("results_mem_variation.csv")

# Color list convention
color_list = ['b','g','r','c','m','y','orange']

# Get unique epsilons and etas
epsilons = sorted(df["epsilon"].unique())
etas = sorted(df["eta"].unique())

# Assume dataset dimension 'd' (set manually or compute elsewhere)
d = 128  # <-- update this depending on your dataset

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

    # Save figure instead of showing
    plt.savefig(f"memory_vs_N_eps{eps}.pdf")
    plt.close()
