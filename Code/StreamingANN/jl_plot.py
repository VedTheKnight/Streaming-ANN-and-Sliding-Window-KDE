#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_jl_results(csv_file, N, d, outdir="plots"):
    # Load CSV
    df = pd.read_csv(csv_file)

    # Compute raw dataset size in MB (assuming float32)
    raw_MB = (N * d * 32) / (1024 * 1024)

    # Compute compression ratio = memory used / raw size
    df["compression_ratio"] = df["memory_MB"] / raw_MB

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Group by c
    for c_val, group in df.groupby("c"):
        group = group.sort_values("compression_ratio")
        ratio = group["compression_ratio"].to_numpy()
        recall = group["recall"].to_numpy()
        cr_acc = group["cr_ann_accuracy"].to_numpy()

        # --- Plot Recall ---
        plt.figure(figsize=(6, 4))
        plt.plot(ratio, recall, marker="o", color="blue", label="Recall")
        plt.xlabel("Compression Ratio")
        plt.ylabel("Recall")
        plt.title(f"JL: Recall vs Compression for c={c_val}")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        outfile = os.path.join(outdir, f"jl_recall_c_{c_val}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved Recall plot: {outfile}")

        # --- Plot (c,r)-NN Accuracy ---
        plt.figure(figsize=(6, 4))
        plt.plot(ratio, cr_acc, marker="x", color="red", label="(c,r)-NN Accuracy")
        plt.xlabel("Compression Ratio")
        plt.ylabel("(c,r)-NN Accuracy")
        plt.title(f"JL: (c,r)-NN Accuracy vs Compression for c={c_val}")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        outfile = os.path.join(outdir, f"jl_crnn_acc_c_{c_val}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved (c,r)-NN Accuracy plot: {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot JL Compression vs Accuracy")
    parser.add_argument("--csv", type=str, required=True, help="Path to JL results CSV")
    parser.add_argument("--N", type=int, required=True, help="Number of points")
    parser.add_argument("--d", type=int, required=True, help="Dimension of points")
    parser.add_argument("--outdir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    plot_jl_results(args.csv, args.N, args.d, args.outdir)
