import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_metrics_vs_compression_ratio(csv_file, N, d, outdir="plots"):
    # Load CSV
    df = pd.read_csv(csv_file)

    # Round memory to 2 decimals
    df["memory_MB"] = df["memory_MB"].round(2)

    # Compute raw dataset size in MB (assuming float32)
    raw_MB = (N * d * 32) / (1024 * 1024)  # bits -> MB
    df["compression_ratio"] = df["memory_MB"] / raw_MB

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # -------- Plot per (r, eps) --------
    for (r_val, eps_val), df_group in df.groupby(["r", "epsilon"]):
        group_outdir = os.path.join(outdir, f"r_{r_val}_eps_{eps_val}")
        os.makedirs(group_outdir, exist_ok=True)

        # Sort by compression ratio
        df_group = df_group.sort_values("compression_ratio")

        # ---- Continuous Recall vs Compression Ratio ----
        plt.figure(figsize=(6, 4))
        plt.plot(
            df_group["compression_ratio"].to_numpy(),
            df_group["recall"].to_numpy(),
            marker="o",
            linestyle="-",  # make it continuous
            color="blue",
            label="Recall"
        )
        plt.xlabel("Compression Ratio")
        plt.ylabel("Recall")
        plt.title(f"Recall vs Compression Ratio (r={r_val}, eps={eps_val})")
        plt.grid(True)
        plt.legend()
        outfile = os.path.join(group_outdir, f"recall_vs_compression_ratio_r{r_val}_eps_{eps_val}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {outfile}")

        # ---- Continuous (c,r)-ANN Accuracy vs Compression Ratio ----
        plt.figure(figsize=(6, 4))
        plt.plot(
            df_group["compression_ratio"].to_numpy(),
            df_group["cr_ann_acc"].to_numpy(),
            marker="s",
            linestyle="-",
            color="green",
            label="(c,r)-ANN Accuracy"
        )
        plt.xlabel("Compression Ratio")
        plt.ylabel("(c,r)-ANN Accuracy")
        plt.title(f"(c,r)-ANN Accuracy vs Compression Ratio (r={r_val}, eps={eps_val})")
        plt.grid(True)
        plt.legend()
        outfile = os.path.join(group_outdir, f"crann_vs_compression_ratio_r{r_val}_eps_{eps_val}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Recall & (c,r)-ANN Accuracy vs Compression Ratio from results.csv")
    parser.add_argument("--csv", type=str, required=True, help="Path to results.csv file")
    parser.add_argument("--N", type=int, required=True, help="Number of points")
    parser.add_argument("--d", type=int, required=True, help="Dimension of points")
    parser.add_argument("--outdir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    plot_metrics_vs_compression_ratio(args.csv, args.N, args.d, args.outdir)
