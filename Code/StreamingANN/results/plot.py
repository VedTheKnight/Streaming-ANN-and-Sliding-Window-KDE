import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- Config ---
color_list = ['b', 'g']  # JL, StreamingANN
epsilon_vals = [0.5, 0.6, 0.7, 0.8, 0.9]
raw_MB = (50000 * 384 * 32) / (1024 * 1024)  # baseline memory
outdir = "plots"
os.makedirs(outdir, exist_ok=True)

# --- Load CSVs ---
jl_df = pd.read_csv("jl_results.csv")
streaming_df = pd.read_csv("streaming_ann_results.csv")  # StreamingANN results

def plot_metric(metric_name, ylabel, filename_prefix):
    for eps in epsilon_vals:
        plt.figure(figsize=(8,6))
        
        # --- JL plot ---
        subset = jl_df[jl_df['c'] == (1 + eps)]
        if not subset.empty and metric_name in subset.columns:
            n_row = (subset['memory_MB'] / raw_MB).to_numpy()  # CR
            vals = subset[metric_name].to_numpy()
            idx = np.argsort(n_row)  # Sort for smooth lines
            plt.plot(n_row[idx], vals[idx], marker='*', mec='black', linestyle='-', 
                     color=color_list[0], lw=1.75, label=f"JL")
        
        # --- StreamingANN plot ---
        subset = streaming_df[streaming_df['epsilon'] == eps]
        if not subset.empty and metric_name in subset.columns:
            n_row = (subset['memory_MB'] / raw_MB).to_numpy()
            vals = subset[metric_name].to_numpy()
            idx = np.argsort(n_row)
            plt.plot(n_row[idx], vals[idx], marker='*', mec='black', linestyle='-', 
                     color=color_list[1], lw=1.75, label=f"StreamingANN")
        
        # --- Formatting ---
        plt.xlabel("Compression Rate (CR)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Compression Rate (epsilon={eps})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # --- Save figure ---
        plt.savefig(os.path.join(outdir, f"{filename_prefix}_vs_CR_epsilon_{eps}.png"), dpi=300)
        plt.close()

# --- Generate Recall and Accuracy plots ---
plot_metric("recall", "Recall", "Recall")
plot_metric("cr_ann_accuracy", "ANN Accuracy", "Accuracy")

print("All Recall and Accuracy figures saved successfully.")
