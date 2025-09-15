import random
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from collections import defaultdict
import psutil
import os
import gc

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
                a = np.random.normal(0, 1, self.d).astype(np.float32)   # Gaussian vector
                b = np.float32(np.random.uniform(0, self.w))             # Uniform offset
                g.append((a, b))
            self.hash_functions.append(g)
        
        self.stream_count = 0
        self.dropped_points = 0
        self.points = []  # store points as float32 for memory efficiency

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
        self.points.append(np.array(point, dtype=np.float32))  # float32 to save memory

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

    def query_topk(self, q, K=10, max_candidates=None):
        """
        Return top-K candidate IDs and distances for query q.
        """
        candidates = set()
        for j in range(self.L):
            key = self._make_key(q, self.hash_functions[j])
            if key is None:
                continue
            candidates.update(self.hash_tables[j].get(key, []))
            if max_candidates is None:
                cap = 3 * self.L
            else:
                cap = max_candidates
            if len(candidates) >= cap:
                break

        # compute exact distances for candidate set
        cand_list = []
        for pid in candidates:
            p = self.points[pid]
            dist = float(np.linalg.norm(p - q))
            cand_list.append((pid, dist))

        cand_list.sort(key=lambda x: x[1])
        return cand_list[:K]


# -----------------------------
# Function to track memory vs points
# -----------------------------
def build_ann_memory_track(files, lengths, idx, r, epsilon, K, eta, queries, interval=100):
    """
    Stream points into ANN and track memory vs points stored.
    Every 'interval' inserts, log number of points stored, memory usage, and avg insert time.
    """
    import time
    process = psutil.Process(os.getpid())
    d = queries.shape[1]
    ann = StreamingANN(d=d, n_estimate=len(idx), eta=eta, epsilon=epsilon, r=r)

    # Generator to yield points
    def sample_points(files, lengths, indices):
        import numpy as np
        start = 0
        idx_set = set(indices)
        for f, L in zip(files, lengths):
            arr = np.load(f, mmap_mode="r")
            for i in range(L):
                gid = start + i
                if gid in idx_set:
                    yield arr[i]
            start += L

    gen = sample_points(files, lengths, idx)

    points_logged = []
    memory_logged = []
    insert_times = []
    fetch_time = 0
    insert_time = 0

    total_points = len(idx)
    mem0 = process.memory_info().rss
    last_t = time.time()

    for i in range(total_points):
        t0 = time.time()
        p = next(gen)
        t1 = time.time()
        fetch_time += (t1 - t0)

        t2 = time.time()
        ann.insert(p)
        t3 = time.time()
        insert_time += (t3 - t2)

        if (i + 1) % interval == 0 or (i + 1) == total_points:
            points_logged.append(len(ann.points))
            mem_now = process.memory_info().rss
            memory_logged.append((mem_now - mem0)/1024/1024)  # in MB
            avg_insert = insert_time / interval * 1e3
            insert_times.append(avg_insert)
            fetch_time = 0
            insert_time = 0
            last_t = time.time()

    # Interpolate slope and estimate total memory
    import numpy as np
    points_logged = np.array(points_logged)
    memory_logged = np.array(memory_logged)
    slope, _ = np.polyfit(points_logged, memory_logged, 1)
    estimated_total_memory_MB = slope * total_points

    # Return estimated memory and epsilon recall
    del ann
    gc.collect()
    return estimated_total_memory_MB, points_logged, memory_logged, insert_times


# -------------------------
# Config
# -------------------------
# files = ["data/encodings_combined.npy"]  # list of npy files
# epsilon = 0.2
# r = 1.0
# K = 10
# eta = 0.0
# n_points_to_insert = 1000
# n_queries = 100

# # Count points in files
# def count_points(files):
#     total = 0
#     lengths = []
#     for f in files:
#         arr = np.load(f, mmap_mode="r")
#         lengths.append(arr.shape[0])
#         total += arr.shape[0]
#     return total, lengths

# total, lengths = count_points(files)

# # Sample indices
# rng = np.random.default_rng(0)
# idx = rng.choice(total, size=n_points_to_insert, replace=False)
# remaining = list(set(range(total)) - set(idx))
# q_idx = rng.choice(remaining, size=n_queries, replace=False)

# # Sample queries
# def sample_points(files, lengths, indices):
#     start = 0
#     idx_set = set(indices)
#     for f, L in zip(files, lengths):
#         arr = np.load(f, mmap_mode="r")
#         for i in range(L):
#             gid = start + i
#             if gid in idx_set:
#                 yield arr[i]
#         start += L

# queries = np.array([p for p in sample_points(files, lengths, q_idx)])

# # -------------------------
# # Run ANN with memory tracking
# # -------------------------
# mem_estimate_MB, points_logged, memory_logged, insert_times = build_ann_memory_track(
#     files, lengths, idx, r, epsilon, K, eta, queries, interval=100
# )

# print(f"Estimated total memory for {n_points_to_insert} points: {mem_estimate_MB:.2f} MB")
