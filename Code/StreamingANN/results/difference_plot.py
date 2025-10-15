import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Load results
jl_df = pd.read_csv("jl_results.csv")
streaming_df = pd.read_csv("streaming_ann_results.csv")

# Parameters
epsilon_vals = sorted(streaming_df['epsilon'].unique())
raw_MB = (50000 * 784 * 4) / (1024 * 1024)

# Fixed CR grid (common for all eps)
CR_grid = np.linspace(0.03, 0.35, 300)

# Fixed color list (same convention as memory_vs_N)
color_list = ['b','g','r','c','m','y','orange']

# ---------------------------
# BLOCK 1: Recall difference
# ---------------------------
plt.figure(figsize=(10, 6))
recall_medians = []

for i, eps in enumerate(epsilon_vals):
    jl_subset = jl_df[np.isclose(jl_df['c'], 1 + eps)]
    sa_subset = streaming_df[streaming_df['epsilon'] == eps]
    if jl_subset.empty or sa_subset.empty:
        continue

    jl_cr = (jl_subset['memory_MB'] / raw_MB).to_numpy()
    jl_rec = jl_subset['recall'].to_numpy()
    sa_cr = (sa_subset['memory_MB'] / raw_MB).to_numpy()
    sa_rec = sa_subset['recall'].to_numpy()

    jl_cr, jl_rec = jl_cr[1:], jl_rec[1:]
    sa_cr, sa_rec = sa_cr[1:], sa_rec[1:]

    jl_idx = np.argsort(jl_cr)
    sa_idx = np.argsort(sa_cr)

    jl_interp = np.interp(CR_grid, jl_cr[jl_idx], jl_rec[jl_idx], left=np.nan, right=np.nan)
    sa_interp = np.interp(CR_grid, sa_cr[sa_idx], sa_rec[sa_idx], left=np.nan, right=np.nan)

    mask = ~np.isnan(jl_interp) & ~np.isnan(sa_interp)
    if mask.sum() < 5:
        continue

    diff = (sa_interp[mask] - jl_interp[mask])
    median_val = np.median(diff)
    recall_medians.append((eps, median_val))

    window_length = min(21, len(diff))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 3:
        diff_smooth = savgol_filter(diff, window_length=window_length, polyorder=3)
    else:
        diff_smooth = diff

    plt.plot(
        CR_grid[mask], diff_smooth,
        lw=1.75, marker='*', mec='black', linestyle='-',
        color=color_list[i % len(color_list)], label=f"ε={eps}"
    )
    plt.hlines(median_val, CR_grid[mask].min(), CR_grid[mask].max(),
               colors=color_list[i % len(color_list)], linestyles="dashed", linewidth=1.5)

plt.axhline(0, color='black', lw=1.5, linestyle='--')
plt.xlabel("Compression (sketch size / uncompressed size)")
plt.ylabel("(Recall StreamingANN - Recall JL) × 100")
plt.title("Recall Difference: StreamingANN vs JL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("recall_difference_fm.pdf", dpi=300)
plt.close()

# ----------------------------------
# BLOCK 2: c,r-ANN accuracy difference
# ----------------------------------
plt.figure(figsize=(10, 6))
accuracy_medians = []

for i, eps in enumerate(epsilon_vals):
    jl_subset = jl_df[np.isclose(jl_df['c'], 1 + eps)]
    sa_subset = streaming_df[streaming_df['epsilon'] == eps]
    if jl_subset.empty or sa_subset.empty:
        continue

    jl_cr = (jl_subset['memory_MB'] / raw_MB).to_numpy()
    jl_acc = jl_subset['cr_ann_accuracy'].to_numpy()
    sa_cr = sa_subset['memory_MB'].to_numpy() / raw_MB
    sa_acc = sa_subset['cr_ann_accuracy'].to_numpy()

    jl_cr, jl_acc = jl_cr[1:], jl_acc[1:]
    sa_cr, sa_acc = sa_cr[1:], sa_acc[1:]

    jl_idx = np.argsort(jl_cr)
    sa_idx = np.argsort(sa_cr)

    jl_interp = np.interp(CR_grid, jl_cr[jl_idx], jl_acc[jl_idx], left=np.nan, right=np.nan)
    sa_interp = np.interp(CR_grid, sa_cr[sa_idx], sa_acc[sa_idx], left=np.nan, right=np.nan)

    mask = ~np.isnan(jl_interp) & ~np.isnan(sa_interp)
    if mask.sum() < 5:
        continue

    diff = (sa_interp[mask] - jl_interp[mask])
    median_val = np.median(diff)
    accuracy_medians.append((eps, median_val))

    window_length = min(21, len(diff))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 3:
        diff_smooth = savgol_filter(diff, window_length=window_length, polyorder=3)
    else:
        diff_smooth = diff

    plt.plot(
        CR_grid[mask], diff_smooth,
        lw=1.75, marker='*', mec='black', linestyle='-',
        color=color_list[i % len(color_list)], label=f"ε={eps}"
    )
    plt.hlines(median_val, CR_grid[mask].min(), CR_grid[mask].max(),
               colors=color_list[i % len(color_list)], linestyles="dashed", linewidth=1.5)

plt.axhline(0, color='black', lw=1.5, linestyle='--')
plt.xlabel("Compression (sketch size / uncompressed size)")
plt.ylabel("(Accuracy StreamingANN - JL) x 100")
plt.title("c,r-ANN Accuracy Difference: StreamingANN vs JL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_difference_fm.pdf", dpi=300)
plt.close()

# ----------------------------------
# BLOCK 3: Combined summary (median diff vs ε)
# ----------------------------------
recall_eps, recall_vals = zip(*recall_medians)
acc_eps, acc_vals = zip(*accuracy_medians)

plt.figure(figsize=(10, 6))
plt.plot(recall_eps, recall_vals, marker='o', mec='black', lw=1.75,
         color='tab:blue', label="Approximate Recall@50")
plt.plot(acc_eps, acc_vals, marker='s', mec='black', lw=1.75,
         color='tab:red', label="(c,r)-ANN Accuracy")
plt.axhline(0, color='black', linestyle='--')

plt.xlabel("ε")
plt.ylabel("Median Difference")
plt.title("Median Difference vs ε (StreamingANN vs JL)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("summary_difference_jl.pdf")
plt.close()

print("Plots saved: recall_difference_fm.pdf, accuracy_difference_fm.pdf, summary_difference_jl.pdf")
