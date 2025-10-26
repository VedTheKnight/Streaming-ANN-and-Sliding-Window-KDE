import numpy as np
from scipy.special import gamma
import argparse
import matplotlib.pyplot as plt
from scipy.stats import poisson
import pandas as pd

def ball_volume(d, r):
    """Volume of a d-dimensional ball of radius r."""
    return (np.pi ** (d / 2) / gamma(d / 2 + 1)) * (r ** d)

def generate_points(d, R, m=None, r=1.0, target_total=1e5, seed=None):
    """
    Generate points from a homogeneous Poisson Point Process (PPP) in [0,R]^d.
    For high dimensions (d>20), fall back to generating approximately `target_total`
    uniform random points for numerical stability.
    """
    rng = np.random.default_rng(seed)

    if d > 20:
        N = int(target_total)
        print(f"[high-dim] Directly generating {N} uniform points in {d}D cube (R={R}).")
        points = rng.uniform(0, R, size=(N, d))
        # Provide dummy m, lam for consistency
        Vdr = ball_volume(d, r)
        lam = N / (R ** d)
        m = lam * Vdr
        return points, m, lam

    # Normal PPP logic
    Vdr = ball_volume(d, r)
    if m is None:
        lam = target_total / (R ** d)
        m = lam * Vdr
        print(f"[auto] Chose m = {m:.3f} to yield about {int(target_total)} points on average.")
    else:
        lam = m / Vdr

    expected_total = lam * (R ** d)
    N = rng.poisson(expected_total)
    print(f"Expected total points: {expected_total:.0f}, actually generated: {N}")

    points = rng.uniform(0, R, size=(N, d))
    return points, m, lam


def verify_distribution(points, d, R, m, r, num_samples=500, seed=None):
    """Empirically verify that counts in random balls follow Poisson(m)."""
    if m is None:
        print("[Warning] Skipping verification since m=None (high-dimensional case).")
        return

    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(num_samples):
        center = rng.uniform(r, R - r, size=d)
        dists = np.linalg.norm(points - center, axis=1)
        counts.append(np.sum(dists <= r))
    counts = np.array(counts)

    plt.hist(counts, bins=np.arange(counts.min(), counts.max()+2)-0.5,
             density=True, alpha=0.6, label="Empirical")
    k = np.arange(counts.min(), counts.max()+1)
    plt.plot(k, poisson.pmf(k, m), 'ro-', label=f"Poisson({m:.2f})")
    plt.xlabel("Number of points in ball")
    plt.ylabel("Probability")
    plt.title(f"Verification: Counts in ball of radius {r}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--R", type=float, default=50.0)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=None)
    parser.add_argument("--target_total", type=float, default=1e5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out", type=str, default="points.csv")
    args = parser.parse_args()

    pts, m, lam = generate_points(args.d, args.R, args.m, args.r, args.target_total, args.seed)
    print(f"Generated {len(pts)} points in {args.d}D cube of side {args.R}")
    
    pd.DataFrame(pts, columns=[f"x{i+1}" for i in range(args.d)]).to_csv(args.out, index=False)
    print(f"Saved points to {args.out}")

    if args.verify:
        verify_distribution(pts, args.d, args.R, m, args.r, seed=args.seed)
