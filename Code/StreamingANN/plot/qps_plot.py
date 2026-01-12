import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Config ---
CSV = "qps_results.csv"
out_dir = "plots_compare"
os.makedirs(out_dir, exist_ok=True)

dataset_labels = {
    'data_csvs/fashion_mnist.csv': 'Fashion-MNIST',
    'data_csvs/sift-1m.csv': 'SIFT1M',
    'data_csvs/synthetic_points.csv': 'Syn-32'
}

# Plotting style
plt.rcParams.update({
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "font.size": 10
})

# Colors / markers
JL_color = "tab:blue"
ANN_color = "tab:orange"
JL_marker = "o"
ANN_marker = "^"

# --- Load data ---
df = pd.read_csv(CSV)
df["dataset_label"] = df["dataset"].map(dataset_labels).fillna(df["dataset"])

# Ensure numeric types
for col in ["QPS", "recall", "eta", "k"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

datasets = ["Fashion-MNIST", "SIFT1M", "Syn-32"]

# --- Figure setup ---
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), sharey=True)

for row_idx, ds in enumerate(datasets):
    ds_df = df[df["dataset_label"] == ds]
    qps_vals = ds_df["QPS"].dropna()

    # Consistent QPS range per dataset
    if not qps_vals.empty:
        qps_min = max(0.0, qps_vals.min() * 0.9)
        qps_max = qps_vals.max() * 1.1
        if np.isclose(qps_min, qps_max):
            qps_min = 0.0
            qps_max += 1.0
    else:
        qps_min, qps_max = 0.0, 1.0

    # --- JL subplot (left) ---
    ax_jl = axes[row_idx, 0]
    ax_jl_right = ax_jl.twinx()

    jl_df = df[(df["method"] == "JL") & (df["dataset_label"] == ds)].dropna(subset=["k", "recall", "QPS"])
    if not jl_df.empty:
        jl_grp = jl_df.groupby("k").agg({"recall": "mean", "QPS": "mean"}).reset_index().sort_values("k")

        # Convert to numpy arrays to avoid the Multi-dimensional indexing issue
        k_vals = jl_grp["k"].to_numpy()
        recall_vals = jl_grp["recall"].to_numpy()
        qps_vals = jl_grp["QPS"].to_numpy()

        ax_jl.scatter(jl_df["k"], jl_df["recall"], s=30, facecolors='none', edgecolors=JL_color, alpha=0.3)
        ax_jl.plot(k_vals, recall_vals, marker=JL_marker, color=JL_color, lw=1.6,
                   label="JL recall", mec='k', markersize=5)
        ax_jl_right.plot(k_vals, qps_vals, linestyle="--", marker='s', color='black',
                         alpha=0.9, lw=1.0, label="QPS", markersize=5)
        ax_jl_right.set_ylim(qps_min, qps_max)

    ax_jl.set_xlabel("k (JL projection dim)")
    if row_idx == 1:
        ax_jl.set_ylabel("Recall", fontsize=11)
    ax_jl.set_title(f"{ds} — JL", fontsize=11)
    ax_jl.set_ylim(-0.02, 1.02)

    ax_jl_right.set_ylabel("QPS", color='black')
    ax_jl_right.tick_params(axis="y", labelcolor='black')
    ax_jl_right.set_ylim(qps_min, qps_max)

    if not jl_df.empty:
        h1, l1 = ax_jl.get_legend_handles_labels()
        h2, l2 = ax_jl_right.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        ax_jl.legend(handles, labels, loc="lower left", fontsize=8)

    # --- ANN subplot (right) ---
    ax_ann = axes[row_idx, 1]
    ax_ann_right = ax_ann.twinx()

    ann_df = df[(df["method"] == "ANN") & (df["dataset_label"] == ds)].dropna(subset=["eta", "recall", "QPS"])
    if not ann_df.empty:
        ann_grp = ann_df.groupby("eta").agg({"recall": "mean", "QPS": "mean"}).reset_index().sort_values("eta")

        eta_vals = ann_grp["eta"].to_numpy()
        recall_vals = ann_grp["recall"].to_numpy()
        qps_vals = ann_grp["QPS"].to_numpy()

        ax_ann.scatter(ann_df["eta"], ann_df["recall"], s=30, facecolors='none', edgecolors=ANN_color, alpha=0.3)
        ax_ann.plot(eta_vals, recall_vals, marker=ANN_marker, color=ANN_color, lw=1.6,
                    label="ANN recall", mec='k', markersize=5)
        ax_ann_right.plot(eta_vals, qps_vals, linestyle="--", marker='*', color='black',
                          alpha=0.9, lw=1.0, label="QPS", markersize=5)
        ax_ann_right.set_ylim(qps_min, qps_max)

    ax_ann.set_xlabel("η (ANN sampling rate)")
    ax_ann.set_title(f"{ds} — ANN", fontsize=11)
    ax_ann.set_ylim(-0.02, 1.02)

    ax_ann_right.set_ylabel("QPS", color='black')
    ax_ann_right.tick_params(axis="y", labelcolor='black')
    ax_ann_right.set_ylim(qps_min, qps_max)

    if not ann_df.empty:
        h1, l1 = ax_ann.get_legend_handles_labels()
        h2, l2 = ax_ann_right.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        ax_ann.legend(handles, labels, loc="lower left", fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out_path = os.path.join(out_dir, "qps.pdf")
plt.savefig(out_path, dpi=100, bbox_inches="tight")
print(f"Saved figure to {out_path}")
plt.show()
