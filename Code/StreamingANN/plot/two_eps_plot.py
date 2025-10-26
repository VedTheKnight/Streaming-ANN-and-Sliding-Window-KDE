import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse

# --- Arguments ---
parser = argparse.ArgumentParser(description="Plot StreamingANN vs JL for two chosen epsilons")
parser.add_argument("--jl_csv", type=str, required=True, help="Path to JL results CSV")
parser.add_argument("--sann_csv", type=str, required=True, help="Path to StreamingANN results CSV")
parser.add_argument("--dim", type=int, required=True, help="Dimensionality of the dataset (for memory normalization)")
parser.add_argument("--n_points", type=int, required=True, help="Number of points in the dataset (for memory normalization)")
parser.add_argument("--epsilons", type=str, default="0.5,0.9", help="Two epsilons to plot, comma-separated")
parser.add_argument("--outdir", type=str, default="plots", help="Directory to save plots")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# --- Config ---
color_list = ['b', 'g']
chosen_epsilons = {"recall": [float(e) for e in args.epsilons.split(",")],
                   "cr_ann_accuracy": [float(e) for e in args.epsilons.split(",")]}
raw_MB = (args.n_points * args.dim * 4) / (1024 * 1024)  # baseline memory

# --- Load CSVs ---
jl_df = pd.read_csv(args.jl_csv)
streaming_df = pd.read_csv(args.sann_csv)

def plot_two_eps(metric_name, ylabel, filename_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x_ticks = np.arange(0, 3.1, 0.1)
    x_max = 3

    for ax, eps in zip(axes, chosen_epsilons[metric_name]):
        # JL
        subset = jl_df[jl_df['c'] == (1 + eps)]
        if not subset.empty and metric_name in subset.columns:
            n_row = (subset['memory_MB'] / raw_MB).to_numpy()
            vals = subset[metric_name].to_numpy()
            idx = np.argsort(n_row)
            ax.plot(n_row[idx], vals[idx], marker='*', mec='black',
                    linestyle='-', color=color_list[0], lw=1.75, label="JL")
        # StreamingANN
        subset = streaming_df[streaming_df['epsilon'] == eps]
        if not subset.empty and metric_name in subset.columns:
            n_row = (subset['memory_MB'] / raw_MB).to_numpy()
            vals = subset[metric_name].to_numpy()
            idx = np.argsort(n_row)
            ax.plot(n_row[idx], vals[idx], marker='*', mec='black',
                    linestyle='-', color=color_list[1], lw=1.75, label="StreamingANN")
        # Formatting
        ax.set_xlabel("Compression (sketch size / uncompressed size)", fontsize=12)
        ax.set_title(f"{ylabel} vs Compression (ε={eps})", fontsize=13, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xlim(0, x_max)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    axes[0].set_ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{filename_prefix}_vs_CR.pdf"), dpi=300)
    plt.close()
    print(f"Saved {filename_prefix}_vs_CR.pdf")

# --- Plot ---
plot_two_eps("recall", "Approximate Recall@50", "Recall")
plot_two_eps("cr_ann_accuracy", "ANN Accuracy", "Accuracy")
