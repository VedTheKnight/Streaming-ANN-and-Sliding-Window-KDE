import numpy as np
from scipy.special import gamma
import argparse
import matplotlib.pyplot as plt
from scipy.stats import poisson

def ball_volume(d, r):
    """Volume of a d-dimensional ball of radius r."""
    return (np.pi ** (d / 2) / gamma(d / 2 + 1)) * (r ** d)

def generate_points(d, R, m, r, seed=None):
    """
    Generate points from a homogeneous Poisson Point Process (PPP) in [0,R]^d
    such that every ball of radius r contains Poisson(m) points in expectation.
    """
    rng = np.random.default_rng(seed)
    
    # intensity lambda = m / volume(ball of radius r)
    Vdr = ball_volume(d, r)
    lam = m / Vdr
    
    # total number of points is Poisson(lambda * volume of cube)
    expected_total = lam * (R ** d)
    N = rng.poisson(expected_total)
    
    # sample N points uniformly in [0,R]^d
    points = rng.uniform(0, R, size=(N, d))
    return points

def verify_distribution(points, d, R, m, r, num_samples=500, seed=None):
    """
    Empirically verify that the number of points inside a ball of radius r 
    follows Poisson(m).
    """
    rng = np.random.default_rng(seed)
    counts = []
    
    for _ in range(num_samples):
        center = rng.uniform(0, R, size=d)
        dists = np.linalg.norm(points - center, axis=1)
        counts.append(np.sum(dists <= r))
    
    counts = np.array(counts)
    
    # Plot histogram vs Poisson(m)
    plt.hist(counts, bins=np.arange(counts.min(), counts.max()+2)-0.5, 
             density=True, alpha=0.6, label="Empirical")
    
    k = np.arange(counts.min(), counts.max()+1)
    plt.plot(k, poisson.pmf(k, m), 'ro-', label=f"Poisson({m})")
    
    plt.xlabel("Number of points in ball")
    plt.ylabel("Probability")
    plt.title(f"Verification: Counts in ball of radius {r}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="(optional) fixed number of points, overrides Poisson sampling")
    parser.add_argument("--d", type=int, required=True, help="dimension of the points")
    parser.add_argument("--R", type=float, required=True, help="side length of cube")
    parser.add_argument("--m", type=float, required=True, help="mean of Poisson distribution per ball of radius r")
    parser.add_argument("--r", type=float, required=True, help="radius of the ball")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--verify", action="store_true", help="run verification routine")
    args = parser.parse_args()
    
    pts = generate_points(args.d, args.R, args.m, args.r, seed=args.seed)
    
    if args.n is not None and args.n < len(pts):
        pts = pts[:args.n]
    
    print(f"Generated {len(pts)} points in {args.d}D cube of side {args.R}")
    np.savetxt("points.txt", pts)
    
    if args.verify:
        verify_distribution(pts, args.d, args.R, args.m, args.r, seed=args.seed)
