import argparse, gc, sys, time, os
import numpy as np
from StreamingANN import StreamingANN
import pandas as pd


def load_fvecs_mmap(filename):
    """Memory-map .fvecs file."""
    with open(filename, "rb") as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
    # Each record is (1 int32 + d float32)
    record_dtype = np.dtype([("dim", np.int32), ("vec", np.float32, (d,))])
    arr = np.memmap(filename, dtype=record_dtype, mode="r")
    if not np.all(arr["dim"] == d):
        raise ValueError("Inconsistent dimensions in fvecs file")
    return arr["vec"]



def count_points(files):
    total = 0
    lengths = []
    for f in files:
        if f.endswith(".npy"):
            arr = np.load(f, mmap_mode="r")
            lengths.append(arr.shape[0])
            total += arr.shape[0]
        elif f.endswith(".fvecs"):
            arr = load_fvecs_mmap(f)
            lengths.append(arr.shape[0])
            total += arr.shape[0]
        else:
            raise ValueError(f"Unsupported file type: {f}")
    return total, lengths


class DatasetAccessor:
    """Lightweight accessor across multiple shards with mmap."""
    def __init__(self, files, normalize: bool = True):
        self.files = []
        self.offsets = [0]
        self.normalize = normalize
        for f in files:
            if f.endswith(".npy"):
                arr = np.load(f, mmap_mode="r")
            elif f.endswith(".fvecs"):
                arr = load_fvecs_mmap(f)
            else:
                raise ValueError(f"Unsupported file type: {f}")
            self.files.append(arr)
            self.offsets.append(self.offsets[-1] + arr.shape[0])
        self.offsets = np.array(self.offsets)

    def __getitem__(self, idx):
        shard = np.searchsorted(self.offsets, idx, side="right") - 1
        local_idx = idx - self.offsets[shard]
        vec = self.files[shard][local_idx].astype(np.float32)
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec



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

def evaluate_cr_ann(dataset, ann, queries, r, c, K):
    """Evaluate (c, r)-ANN success rate."""
    successes, total_relevant = 0, 0
    for q in queries:
        dists = np.linalg.norm(np.array(dataset) - q, axis=1)
        r_star = np.min(dists)

        if r_star <= r:  # relevant query
            total_relevant += 1
            res = ann.query(q)
            if res is not None:
                dist = np.linalg.norm(res - q)
                if dist <= c * r:
                    successes += 1

    if total_relevant == 0:
        return None
    print("[Sanity Check] Total Relevant = ", total_relevant)
    return successes / total_relevant

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
        # print(d_star, epsilon, (1+epsilon)*d_star)
        # print([dist for _, dist in res])
        eps_hits += sum(1 for _, dist in res if dist <= (1 + epsilon) * d_star)
        total += K

    recall_exact = exact_hits / total if total > 0 else None
    recall_eps = eps_hits / total if total > 0 else None

    # (c, r)-ANN evaluation
    cr_ann_acc = evaluate_cr_ann(dataset_split, ann, queries, r, c=(1+epsilon), K=K)

    mem_mb = ann.memory_breakdown()["total_MB"] if isinstance(ann.memory_breakdown(), dict) else ann.memory_breakdown()

    print("[Result] Memory breakdown (MB):", ann.memory_breakdown())
    print(f"[Result] Epsilon-recall: {recall_eps:.4f}")
    if cr_ann_acc is not None:
        print(f"[Result] (c,r)-ANN accuracy: {cr_ann_acc:.4f}")
    else:
        print("[Result] (c,r)-ANN accuracy: N/A (no queries within distance r)")

    del ann
    gc.collect()

    return recall_eps, cr_ann_acc, mem_mb

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

    recall_eps, cr_ann_acc, mem_mb = build_ann(accessor, idx, args.r, args.epsilon, args.K, args.eta, queries)

    print(f"[SUMMARY] r={args.r}, eps={args.epsilon}, eta={args.eta}, recall={recall_eps:.4f}, (c,r)-ANN accuracy={cr_ann_acc}, memory={mem_mb}")

if __name__ == "__main__":
    main()
