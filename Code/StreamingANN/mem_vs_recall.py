import argparse, gc, sys, time, os
import numpy as np
from StreamingANN import StreamingANN
import pandas as pd

def count_points(files):
    total = 0
    lengths = []
    for f in files:
        arr = np.load(f, mmap_mode="r")
        lengths.append(arr.shape[0])
        total += arr.shape[0]
    return total, lengths

class DatasetAccessor:
    """Lightweight accessor across multiple shards with mmap."""
    def __init__(self, files):
        self.files = [np.load(f, mmap_mode="r") for f in files]
        self.offsets = np.cumsum([0] + [arr.shape[0] for arr in self.files])

    def __getitem__(self, idx):
        shard = np.searchsorted(self.offsets, idx, side="right") - 1
        local_idx = idx - self.offsets[shard]
        return self.files[shard][local_idx]

def sample_ids(total, indices):
    """Yield IDs instead of vectors."""
    for gid in indices:
        yield gid

def ground_truth_topk(dataset, queries, K):
    res, kth_dists = [], []
    dataset = np.array(dataset)
    for q in queries:
        dists = np.linalg.norm(dataset - q, axis=1)
        topk = np.argsort(dists)[:K]
        res.append(topk)
        kth_dists.append(dists[topk[-1]])
    return res, np.array(kth_dists)

def build_ann(dataset, idx, r, epsilon, K, eta, queries, log_interval=100):
    d = queries.shape[1]
    ann = StreamingANN(d=d, n_estimate=len(idx), eta=eta, epsilon=epsilon, r=r, dataset=dataset)
    total_points = len(idx)

    insert_times = []
    stored_points_list = []
    query_times = []

    print("[Build] Starting insertion...")
    for i, point_id in enumerate(sample_ids(total_points, idx), 1):
        start = time.time()
        ann.insert(point_id)
        insert_times.append(time.time() - start)

        stored_points_list.append(ann.num_stored_points)
        if i % log_interval == 0 or i == total_points:
            print(f"[Build] Inserted {i}/{total_points} points, "
                  f"avg insert time: {np.mean(insert_times)*1e3:.3f} ms")
            insert_times = []

    print("[Build] Starting queries...")
    for i, q in enumerate(queries, 1):
        start = time.time()
        ann.query_topk(q, K=K)
        query_times.append(time.time() - start)
        if i % log_interval == 0 or i == len(queries):
            print(f"[Query] Processed {i}/{len(queries)} queries, "
                  f"avg query time: {np.mean(query_times)*1e3:.3f} ms")

    dataset_split = np.array([dataset[pid] for pid in idx])
    gt_topk, kth_dists = ground_truth_topk(dataset_split, queries, K)

    exact_hits, eps_hits, total = 0, 0, 0
    for q, gt, d_star in zip(queries, gt_topk, kth_dists):
        res = ann.query_topk(q, K=K)
        ret_ids = [pid for pid, _ in res]
        exact_hits += len(set(gt.tolist()) & set(ret_ids))
        eps_hits += sum(1 for _, dist in res if dist <= (1 + epsilon) * d_star)
        total += K

    recall_exact = exact_hits / total if total > 0 else None
    recall_eps = eps_hits / total if total > 0 else None

    mem_mb = ann.memory_breakdown()["total_MB"] if isinstance(ann.memory_breakdown(), dict) else ann.memory_breakdown()

    print("[Result] Memory breakdown (MB):", ann.memory_breakdown())
    print(f"[Result] Epsilon-recall: {recall_eps:.4f}")

    del ann
    gc.collect()

    return recall_eps, mem_mb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--n_queries", type=int, required=True)
    parser.add_argument("--eta", type=float, required=True)
    args = parser.parse_args()

    total, lengths = count_points(args.files)
    rng = np.random.default_rng(0)

    idx = rng.choice(total, size=args.n, replace=False)
    remaining = list(set(range(total)) - set(idx))
    q_idx = rng.choice(remaining, size=args.n_queries, replace=False)

    accessor = DatasetAccessor(args.files)
    queries = np.array([accessor[pid] for pid in q_idx])

    recall_eps, mem_mb = build_ann(accessor, idx, args.r, args.epsilon, args.K, args.eta, queries)

    print(f"[SUMMARY] eps={args.epsilon}, eta={args.eta}, recall={recall_eps:.4f}, memory={mem_mb}")

if __name__ == "__main__":
    main()
