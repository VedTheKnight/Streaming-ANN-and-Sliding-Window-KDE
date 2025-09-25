import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- Config ---
color_list = ['b', 'g']  # JL, StreamingANN
chosen_epsilons = {"recall":[0.5, 1], "cr_ann_accuracy":[0.5, 1]}    # <--- pick the 2 epsilons you want
raw_MB = (50000 * 784 * 4) / (1024 * 1024)  # baseline memory
outdir = "plots"
os.makedirs(outdir, exist_ok=True)

# --- Load CSVs ---
jl_df = pd.read_csv("jl_results.csv")
streaming_df = pd.read_csv("streaming_ann_results.csv")  # StreamingANN results

def plot_two_eps(metric_name, ylabel, filename_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # --- Fixed x-ticks and limits ---
    x_ticks = np.arange(0, 3.1, 0.1)  # ticks from 0 to 3 with 0.1 step
    x_max = 3

    for ax, eps in zip(axes, chosen_epsilons[metric_name]):
        # --- JL curve ---
        subset = jl_df[jl_df['c'] == (1 + eps)]
        if not subset.empty and metric_name in subset.columns:
            n_row = (subset['memory_MB'] / raw_MB).to_numpy()
            vals = subset[metric_name].to_numpy()
            idx = np.argsort(n_row)
            ax.plot(n_row[idx], vals[idx], marker='*', mec='black',
                    linestyle='-', color=color_list[0], lw=1.75, label="JL")

        # --- StreamingANN curve ---
        subset = streaming_df[streaming_df['epsilon'] == eps]
        if not subset.empty and metric_name in subset.columns:
            n_row = (subset['memory_MB'] / raw_MB).to_numpy()
            vals = subset[metric_name].to_numpy()
            idx = np.argsort(n_row)
            ax.plot(n_row[idx], vals[idx], marker='*', mec='black',
                    linestyle='-', color=color_list[1], lw=1.75, label="StreamingANN")

        # --- Formatting ---
        ax.set_xlabel("Compression (sketch size / uncompressed size)", fontsize=12)
        ax.set_title(f"{ylabel} vs Compression (ε={eps})", fontsize=13, weight="bold")
        ax.grid(True, alpha =0.3)
        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xlim(0, x_max)
        # --- Rotate x-tick labels ---
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    axes[0].set_ylabel(ylabel, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{filename_prefix}_vs_CR_two_eps_fm.pdf"), dpi=300)
    plt.close()
    print(f"Saved {filename_prefix}_vs_CR_two_eps_fm.pdf")




# --- Example usage ---
plot_two_eps("recall", "Approximate Recall@50", "Recall")
plot_two_eps("cr_ann_accuracy", "ANN Accuracy", "Accuracy")
