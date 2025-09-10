import random
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from collections import defaultdict

class StreamingANN:
    def __init__(self, d, n_estimate = None, eta=0.0, r = 1.0, epsilon=0.0, w=None, bits_per_hash=32):
        """
        n_estimate: upper bound on stream size
        d: dimension of data points
        eta: sampling parameter
        r: inner radius for the LSH data structure 
        epsilon: approximation factor (so c = 1+epsilon)
        w: bucket width for p-stable LSH (if None, we will tune it)
        bits_per_hash: number of bits allocated per hash in packed key
        """
    
        self.epsilon = epsilon
        self.c = 1 + epsilon
        self.r = r
        self.eta = eta
        self.n = n_estimate
        self.d = d

        # Define the LSH parameters (p1, p2, rho, w) given epsilon
        if w is None:
            self.w, self.p1, self.p2, self.rho = self._get_optimal_w(self.c, self.r)
        else:
            self.w = w
            self.p1, self.p2, self.rho = self._get_gaussian_lsh_probs(self.w, self.c, self.r)

        self.bits_per_hash = bits_per_hash
        
        
        # Parameters 
        self.k = math.ceil(math.log(self.n, 1/self.p2))
        self.L = int(self.n ** self.rho / self.p1)
        
        # Initialize L hash tables
        self.hash_tables = [defaultdict(list) for _ in range(self.L)]
        
        # Build L hash families, each with k hash functions (a,b)
        self.hash_functions = []
        for _ in range(self.L):
            g = []
            for _ in range(self.k):
                a = np.random.normal(0, 1, self.d)   # Gaussian vector
                b = np.random.uniform(0, self.w)     # Uniform offset
                g.append((a, b))
            self.hash_functions.append(g)
        
        self.stream_count = 0
        self.dropped_points = 0
        self.points = []

        print(f"[Init] Inputs : eta={self.eta:.2f}, epsilon={self.epsilon:0.2f}, r={self.r:0.2f}")
        print(f"[Init] LSH Parameters : w={self.w:.3f}, p1={self.p1:.4f}, p2={self.p2:.4f}, rho={self.rho:.4f}")
        print(f"[Init] Data Structure Parameters : k={self.k}, L={self.L}")
    
    def _get_gaussian_lsh_probs(self, w, c, r):
        """
        Compute (p1, p2, rho) for L2 (Gaussian) p-stable LSH given c using the closed form given in Datar-Indyk
        w : Bucket width
        c : Approximation factor (1+epsilon) (>1).
        Return (p1, p2, rho)
        """
        # p1 corresponds to distance = r
        t1 = w/r
        p1 = 1 - 2*norm.cdf(-t1) - (2.0/(math.sqrt(2*math.pi)*t1)) * (1 - math.exp(-0.5*t1*t1))

        # p2 corresponds to distance = cr
        t2 = w / (c*r)  
        p2 = 1 - 2*norm.cdf(-t2) - (2.0/(math.sqrt(2*math.pi)*t2)) * (1 - math.exp(-0.5*t2*t2))

        rho = math.log(1.0/p1) / math.log(1.0/p2)
        return p1, p2, rho
    
    def _get_optimal_w(self, c, r):
        """
        Use continuous optimization to find w that minimizes rho.
        Returns (w*, p1, p2, rho).
        """
        def objective(w):
            if w <= 0:  # avoid invalid widths
                return float("inf")
            _, _, rho = self._get_gaussian_lsh_probs(w, c, r)
            return rho
        
        res = minimize_scalar(objective, bounds=(1e-3, 10), method='bounded')
        
        w_opt = res.x
        p1, p2, rho = self._get_gaussian_lsh_probs(w_opt, c, r)
        return w_opt, p1, p2, rho

    def _h(self, v, a, b):
        """Single p-stable hash h_{a,b}(v)."""
        return int(np.floor((np.dot(a, v) + b) / self.w))
    
    def _make_key(self, point, hash_funcs):
        """
        Pack the outputs of k p-stable hashes into a single integer.
        Normalize negative hash values into unsigned space so they fit in bits_per_hash.
        Drop the point if any hash exceeds bits_per_hash capacity.
        """
        key = 0
        max_val = (1 << self.bits_per_hash) - 1
        offset = max_val // 2   # center around 0

        for (a, b) in hash_funcs:
            val = self._h(point, a, b)
            val_shifted = val + offset
            if val_shifted < 0 or val_shifted > max_val:
                return None
            key = (key << self.bits_per_hash) | val_shifted
        return key
    
    def _should_sample(self):
        """Bernoulli sampling with probability i^(-eta) or n^(-eta)"""
        self.stream_count += 1
        # prob = self.stream_count ** (-self.eta)
        prob = self.n ** (-self.eta) 
        return random.random() < prob
    
    def insert(self, point):
        """Insert sampled point into LSH tables."""
        if not self._should_sample():
            self.dropped_points += 1
            return
        
        pid = len(self.points)   # new point ID
        self.points.append(point)

        for j in range(self.L):
            key = self._make_key(point, self.hash_functions[j])
            if key is None:
                self.dropped_points += 1
                return
            self.hash_tables[j][key].append(pid)
    
    def query(self, q):
        """Find an approximate near neighbor for q using L2 distance."""
        candidates = set()
        for j in range(self.L):
            key = self._make_key(q, self.hash_functions[j])
            if key is None:
                continue
            candidates.update(self.hash_tables[j].get(key, []))
            if len(candidates) >= 3 * self.L:
                break

        cr = self.c * self.r
        for pid in candidates:
            p = self.points[pid]
            # Euclidean distance check
            if np.linalg.norm(p - q) <= cr:
                return p
        return None

