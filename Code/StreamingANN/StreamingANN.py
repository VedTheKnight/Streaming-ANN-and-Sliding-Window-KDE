import random
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from collections import defaultdict
import psutil
import os
import gc
from array import array
from typing import Optional


class StreamingANN:
    def __init__(self, d, dataset, n_estimate=None, eta=0.0, r=1.0, epsilon=0.0, w=None,
                 bits_per_hash=32, mode: str = "speed"):
        """
        n_estimate: upper bound on stream size
        d: dimension of data points
        dataset: accessor supporting __getitem__(idx) (e.g. np.memmap or custom accessor)
        eta: sampling parameter
        r: inner radius for the LSH data structure 
        epsilon: approximation factor (so c = 1+epsilon)
        w: bucket width for p-stable LSH (if None, we will tune it)
        bits_per_hash: number of bits allocated per hash in packed key
        mode: "speed" (faster, slightly more memory) or "memory" (smaller memory, slower)
        """
        self.epsilon = epsilon
        self.c = 1 + epsilon
        self.r = r
        self.eta = eta
        self.n = n_estimate
        self.d = d
        self.dataset = dataset
        self.mode = mode

        # Define the LSH parameters (p1, p2, rho, w) given epsilon
        if w is None:
            self.w, self.p1, self.p2, self.rho = self._get_optimal_w(self.c, self.r)
        else:
            self.w = w
            self.p1, self.p2, self.rho = self._get_gaussian_lsh_probs(self.w, self.c, self.r)

        self.bits_per_hash = bits_per_hash
        self.inv_w = 1.0 / self.w  # precompute 1/w
        self.stream_count = 0
        self.dropped_points = 0
        self.num_stored_points = 0

        # Parameters 
        p2_safe = max(self.p2, 1e-12)
        self.k = max(1, math.ceil(math.log(max(1, (self.n if self.n else 1)), 1.0/p2_safe)))
        try:
            self.L = max(1, int((self.n ** self.rho) / max(self.p1, 1e-12)))
        except Exception:
            self.L = 1

        # Initialize L hash tables
        self.hash_tables = [defaultdict(lambda: array('I')) for _ in range(self.L)]

        # Build L hash families, each with k hash functions (a,b)
        self.hash_functions = []
        for _ in range(self.L):
            g = []
            for _ in range(self.k):
                if self.mode == "speed":
                    a = np.random.normal(0, 1, self.d).astype(np.float32)
                else:
                    a = np.random.normal(0, 1, self.d).astype(np.float16)
                b = np.float32(np.random.uniform(0, self.w))
                g.append((a, b))
            a_stack = np.vstack([aa for aa, _ in g])
            b_vec = np.array([bb for _, bb in g], dtype=np.float32)
            b_scaled = b_vec * self.inv_w  # precompute division
            self.hash_functions.append((a_stack, b_vec, b_scaled))

        # Precompute whether packed key fits in 64 bits
        self._total_key_bits = int(self.k * self.bits_per_hash)
        self._use_uint64_key = (self._total_key_bits <= 64)

        print(f"[Init] Inputs : eta={self.eta:.2f}, epsilon={self.epsilon:0.2f}, r={self.r:0.2f}")
        print(f"[Init] LSH Parameters : w={self.w:.3f}, p1={self.p1:.4f}, p2={self.p2:.4f}, rho={self.rho:.4f}")
        print(f"[Init] Data Structure Parameters : k={self.k}, L={self.L}, key_bits={self._total_key_bits}, mode={self.mode}")

    def _h_vectorized(self, point: np.ndarray, a_stack: np.ndarray, b_scaled: np.ndarray) -> np.ndarray:
        """Vectorized computation of k hashes for a table"""
        proj_bins = a_stack.dot(point.astype(np.float32)) * self.inv_w + b_scaled
        return proj_bins.astype(np.int64)

    def _pack_key_from_hashes(self, hashes: np.ndarray):
        """Pack k integer hash values into a single key"""
        max_val = (1 << self.bits_per_hash) - 1
        offset = max_val // 2

        if self._use_uint64_key:
            key = np.uint64(0)
            for v in hashes:
                v_shift = int(v) + offset
                if v_shift < 0 or v_shift > max_val:
                    return None
                key = (key << np.uint64(self.bits_per_hash)) | np.uint64(v_shift)
            return key
        else:
            key = 0
            for v in hashes:
                v_shift = int(v) + offset
                if v_shift < 0 or v_shift > max_val:
                    return None
                key = (key << self.bits_per_hash) | v_shift
            return key

    def _should_sample(self):
        self.stream_count += 1
        prob = self.n ** (-self.eta) 
        return random.random() < prob

    def insert(self, point_id):
        if not self._should_sample():
            self.dropped_points += 1
            return

        point = self.dataset[point_id].astype(np.float32)

        inserted_ok = True
        for j, (a_stack, b_vec, b_scaled) in enumerate(self.hash_functions):
            # vectorized dot
            proj_bins = a_stack.dot(point) * self.inv_w + b_scaled
            hashes = proj_bins.astype(np.int64)

            key = self._pack_key_from_hashes(hashes)
            if key is None:
                inserted_ok = False
                break

            self.hash_tables[j][key].append(np.uint32(point_id))

        if inserted_ok:
            self.num_stored_points += 1
        else:
            self.dropped_points += 1
        del point

    def query(self, q):
        q32 = q.astype(np.float32, copy=False)
        candidate_ids = []
        for j in range(self.L):
            a_stack, b_vec, b_scaled = self.hash_functions[j]
            hashes = self._h_vectorized(q32, a_stack, b_scaled)
            if hashes is None:
                continue
            key = self._pack_key_from_hashes(hashes)
            if key is None:
                continue
            bucket = self.hash_tables[j].get(key)
            if bucket:
                candidate_ids.extend(bucket)
            if len(candidate_ids) >= 3 * self.L * 10:
                break
        if not candidate_ids:
            return None

        uniq = np.unique(np.array(candidate_ids, dtype=np.uint32))
        cr = self.c * self.r
        if uniq.size > 64:
            mats = np.vstack([self.dataset[int(pid)].astype(np.float32) for pid in uniq])
            dists = np.linalg.norm(mats - q32, axis=1)
            idx = np.where(dists <= cr)[0]
            if idx.size > 0:
                return mats[idx[0]].copy()
            del mats, dists
        else:
            for pid in uniq:
                p = self.dataset[int(pid)]
                if np.linalg.norm(p.astype(np.float32, copy=False) - q32) <= cr:
                    return np.array(p, copy=True)
                del p
        return None

    def query_topk(self, q, K=10, max_candidates=None):
        q32 = q.astype(np.float32, copy=False)
        candidate_ids = []
        for j in range(self.L):
            a_stack, b_vec, b_scaled = self.hash_functions[j]
            hashes = self._h_vectorized(q32, a_stack, b_scaled)
            if hashes is None:
                continue
            key = self._pack_key_from_hashes(hashes)
            if key is None:
                continue
            bucket = self.hash_tables[j].get(key)
            if bucket:
                candidate_ids.extend(bucket)
            if max_candidates is None:
                cap = 3 * self.L
            else:
                cap = max_candidates
            if len(candidate_ids) >= cap:
                break

        if not candidate_ids:
            return []

        uniq = np.unique(np.array(candidate_ids, dtype=np.uint32))
        cand_list = []
        if uniq.size > 256:
            mats = np.vstack([self.dataset[int(pid)].astype(np.float32) for pid in uniq])
            dists = np.linalg.norm(mats - q32, axis=1)
            cand_list = [(int(pid), float(dist)) for pid, dist in zip(uniq, dists)]
            del mats, dists
        else:
            for pid in uniq:
                p = self.dataset[int(pid)]
                dist = float(np.linalg.norm(p.astype(np.float32, copy=False) - q32))
                cand_list.append((int(pid), dist))
                del p

        cand_list.sort(key=lambda x: x[1])
        return cand_list[:K]

    def memory_breakdown(self):
        """Memory in MB (updated with precomputed b_scaled)"""
        breakdown = {}

        hash_bytes = 0
        for a_stack, b_vec, b_scaled in self.hash_functions:
            hash_bytes += a_stack.nbytes + b_vec.nbytes + b_scaled.nbytes
        breakdown["hash_functions_MB"] = hash_bytes / (1024**2)

        bucket_bytes = 0
        n_ids = 0
        n_buckets = 0
        for table in self.hash_tables:
            for bucket in table.values():
                bucket_bytes += len(bucket) * 4
                n_ids += len(bucket)
                n_buckets += 1
        breakdown["bucket_ids_MB"] = bucket_bytes / (1024**2)
        breakdown["num_ids"] = n_ids
        breakdown["num_buckets"] = n_buckets

        dict_overhead = sum(len(table) * (72 + 16) for table in self.hash_tables)
        breakdown["hash_tables_overhead_MB"] = dict_overhead / (1024**2)
        breakdown["total_MB"] = breakdown["hash_functions_MB"] + breakdown["bucket_ids_MB"] + breakdown["hash_tables_overhead_MB"]

        return breakdown

    def _get_gaussian_lsh_probs(self, w, c, r):
        """Compute (p1, p2, rho) for L2 (Gaussian) p-stable LSH"""
        t1 = w / r
        p1 = 1 - 2*norm.cdf(-t1) - (2.0/(math.sqrt(2*math.pi)*t1)) * (1 - math.exp(-0.5*t1*t1))

        t2 = w / (c*r)
        p2 = 1 - 2*norm.cdf(-t2) - (2.0/(math.sqrt(2*math.pi)*t2)) * (1 - math.exp(-0.5*t2*t2))

        p1 = min(max(p1, 1e-12), 1-1e-12)
        p2 = min(max(p2, 1e-12), 1-1e-12)

        rho = math.log(1.0/p1) / math.log(1.0/p2)
        return p1, p2, rho

    def _get_optimal_w(self, c, r):
        """Use continuous optimization to find w that minimizes rho"""
        def objective(w):
            if w <= 0:
                return float("inf")
            _, _, rho = self._get_gaussian_lsh_probs(w, c, r)
            return rho

        res = minimize_scalar(objective, bounds=(1e-3, 10), method='bounded')
        w_opt = res.x
        p1, p2, rho = self._get_gaussian_lsh_probs(w_opt, c, r)
        return w_opt, p1, p2, rho
