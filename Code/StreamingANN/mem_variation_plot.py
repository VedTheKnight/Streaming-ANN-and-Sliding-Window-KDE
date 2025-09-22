#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Results CSV file")
    parser.add_argument("--out", type=str, default="mem_vs_N.png", help="Output figure filename")
    args = parser.parse_args()

    # Load results
    df = pd.read_csv(args.csv)

    # Ensure correct types
    df["N"] = df["N"].astype(int)
    df["epsilon"] = df["epsilon"].astype(float)
    df["eta"] = df["eta"].astype(float)
    df["memory_MB"] = df["memory_MB"].astype(float)

    # Pick only eta=0.5 and eta=0.3
    etas = [0.5, 0.3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, eta in zip(axes, etas):
        subset = df[df["eta"] == eta]
        for eps in sorted(subset["epsilon"].unique()):
            df_eps = subset[subset["epsilon"] == eps]
            df_eps = df_eps.sort_values("N")
            ax.plot(
                df_eps["N"],
                df_eps["memory_MB"],
                marker="o",
                label=f"$\\epsilon={eps}$"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (dataset size)")
        ax.set_title(f"$\\eta={eta}$")
        ax.grid(True, which="both", ls="--", lw=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Memory (MB)")
        ax.legend()

    plt.suptitle("Memory vs N for different $\\epsilon$, fixed $\\eta$")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[INFO] Plot saved to {args.out}")

if __name__ == "__main__":
    main()
