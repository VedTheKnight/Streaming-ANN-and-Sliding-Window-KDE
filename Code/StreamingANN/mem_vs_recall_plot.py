import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_recall_vs_compression_ratio(csv_file, N, d, outdir="plots"):
    # Load CSV
    df = pd.read_csv(csv_file)

    # Round memory to 2 decimals
    df["memory_MB"] = df["memory_MB"].round(2)

    # Compute raw dataset size in MB (assuming float32)
    raw_MB = (N * d * 32) / (1024 * 1024)

    # Compute compression ratio = memory used / raw size
    df["compression_ratio"] = df["memory_MB"] / raw_MB

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Group by epsilon
    for eps, group in df.groupby("epsilon"):
        plt.figure(figsize=(6, 4))

        # Sort by compression ratio for nicer plots
        group = group.sort_values("compression_ratio")

        ratio = group["compression_ratio"].to_numpy()
        recall = group["recall"].to_numpy()

        plt.plot(ratio, recall, marker="o", label=f"eps={eps}")

        plt.xlabel("Compression Ratio")
        plt.ylabel("Recall")
        plt.title(f"Recall vs Compression Ratio for eps={eps}")
        plt.grid(True)
        plt.legend()

        outfile = os.path.join(outdir, f"recall_vs_compression_ratio_eps_{eps}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Saved {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Recall vs Compression Ratio from results.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to results.csv file")
    parser.add_argument("--N", type=int, required=True, help="Number of points")
    parser.add_argument("--d", type=int, required=True, help="Dimension of points")
    parser.add_argument("--outdir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    plot_recall_vs_compression_ratio(args.csv, args.N, args.d, args.outdir)
