import argparse, gc, sys, time, os
import numpy as np
from StreamingANN import StreamingANN
import pandas as pd
import faiss


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

def load_csv(filename):
    """Load Fashion-MNIST style CSV (first col=label, rest=pixels)."""
    arr = np.loadtxt(filename, delimiter=",", dtype=np.float32, skiprows=1)
    # Drop first column (labels), keep only pixel values
    return arr[:, 1:]


def count_points(files):
    total = 0
    lengths = []
    for f in files:
        if f.endswith(".npy"):
            arr = np.load(f, mmap_mode="r")
        elif f.endswith(".fvecs"):
            arr = load_fvecs_mmap(f)
        elif f.endswith(".csv"):
            arr = load_csv(f)
        else:
            raise ValueError(f"Unsupported file type: {f}")
        lengths.append(arr.shape[0])
        total += arr.shape[0]
    return total, lengths


class DatasetAccessor:
    """Lightweight accessor across multiple shards with mmap and CSV support."""
    def __init__(self, files, normalize: bool = True):
        self.files = []
        self.offsets = [0]
        self.normalize = normalize
        for f in files:
            if f.endswith(".npy"):
                arr = np.load(f, mmap_mode="r")
            elif f.endswith(".fvecs"):
                arr = load_fvecs_mmap(f)
            elif f.endswith(".csv"):
                arr = load_csv(f)
            else:
                raise ValueError(f"Unsupported file type: {f}")
            self.files.append(arr)
            self.offsets.append(self.offsets[-1] + arr.shape[0])
        self.offsets = np.array(self.offsets)

    def __getitem__(self, idx):
        shard = np.searchsorted(self.offsets, idx, side="right") - 1
        local_idx = idx - self.offsets[shard]
        vec = self.files[shard][local_idx].astype(np.float32)

        # Optionally L2-normalize
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec


def sample_ids(total, indices):
    """Yield IDs instead of vectors."""
    for gid in indices:
        yield gid

def ground_truth_topk(dataset, queries, K, log_interval=100):
    """
    Compute exact top-K neighbors for queries using FAISS (L2 search).
    Much faster than NumPy loops.
    """
    dataset = np.asarray(dataset, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)

    index = faiss.IndexFlatL2(dataset.shape[1])  # brute-force L2
    index.add(dataset)

    t0 = time.time()
    D, I = index.search(queries, K)  # D = squared distances, I = indices
    t_total = time.time() - t0

    # kth distance is the largest in each row of D
    kth_dists = np.sqrt(D[:, -1])  # convert from squared dist to norm

    print(f"[Timing] Ground truth (FAISS): {t_total:.2f} s "
          f"({t_total/len(queries)*1e3:.3f} ms/query)")

    return I, kth_dists



def evaluate_cr_ann(dataset, ann, queries, r, c, K, log_interval=100):
    """
    Evaluate (c, r)-ANN success rate using FAISS to compute r* (nearest neighbor distance).
    """
    dataset = np.asarray(dataset, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)

    index = faiss.IndexFlatL2(dataset.shape[1])
    index.add(dataset)

    t0 = time.time()
    # Get nearest neighbor distance r_star for each query
    D, _ = index.search(queries, 1)  # D is squared distance to 1-NN
    r_stars = np.sqrt(D[:, 0])
    t_total = time.time() - t0
    print(f"[Timing] FAISS r* computation: {t_total:.2f} s "
          f"({t_total/len(queries)*1e3:.3f} ms/query)")

    successes, total_relevant = 0, 0
    query_times = []
    t0 = time.time()

    for i, (q, r_star) in enumerate(zip(queries, r_stars), 1):
        start = time.time()
        if r_star <= r:  # relevant query
            total_relevant += 1
            res = ann.query(q)  # your ANN returns a vector
            if res is not None:
                dist = np.linalg.norm(res - q)
                if dist <= c * r:
                    successes += 1
        query_times.append(time.time() - start)

        if i % log_interval == 0 or i == len(queries):
            avg_ms = np.mean(query_times) * 1e3
            print(f"[CR-ANN] {i}/{len(queries)} processed, "
                  f"avg time (ann.query): {avg_ms:.3f} ms")
            query_times = []

    t_total = time.time() - t0
    print(f"[Timing] (c,r)-ANN eval: {t_total:.2f} s "
          f"({t_total/len(queries)*1e3:.3f} ms/query)")

    if total_relevant == 0:
        print("[CR-ANN] No relevant queries. Returning None.")
        return None

    print(f"[Sanity Check] Total Relevant = {total_relevant}")
    return successes / total_relevant



def build_ann(dataset, idx, r, epsilon, K, eta, queries, log_interval=100):
    d = queries.shape[1]
    ann = StreamingANN(d=d, n_estimate=len(idx), eta=eta, epsilon=epsilon, r=r, dataset=dataset)
    total_points = len(idx)

    print("[Build] Starting insertion...")
    t0 = time.time()
    insert_times = []
    for i, point_id in enumerate(sample_ids(total_points, idx), 1):
        start = time.time()
        ann.insert(point_id)
        insert_times.append(time.time() - start)

        if i % log_interval == 0 or i == total_points:
            print(f"[Build] Inserted {i}/{total_points} points, "
                  f"avg insert time (ms): {np.mean(insert_times)*1e3:.3f}")
            insert_times = []
    t_insert = time.time() - t0
    print(f"[Timing] Total insertion time: {t_insert:.2f} s")

    print("[Build] Starting queries...")
    t0 = time.time()
    query_times = []
    for i, q in enumerate(queries, 1):
        start = time.time()
        ann.query_topk(q, K=K)
        query_times.append(time.time() - start)
        if i % log_interval == 0 or i == len(queries):
            print(f"[Query] Processed {i}/{len(queries)} queries, "
                  f"avg query time (ms): {np.mean(query_times)*1e3:.3f}")
            query_times = []
    t_query = time.time() - t0
    print(f"[Timing] Total query time: {t_query:.2f} s")

    # Ground truth computation
    print("[Eval] Computing ground truth top-K...")
    t0 = time.time()
    dataset_split = np.array([dataset[pid] for pid in idx])
    gt_topk, kth_dists = ground_truth_topk(dataset_split, queries, K)
    t_gt = time.time() - t0
    print(f"[Timing] Ground truth computation: {t_gt:.2f} s")

    # Recall computation
    print("[Eval] Computing recall metrics...")
    t0 = time.time()
    exact_hits, eps_hits, total = 0, 0, 0
    for q, gt, d_star in zip(queries, gt_topk, kth_dists):
        res = ann.query_topk(q, K=K)
        ret_ids = [pid for pid, _ in res]
        exact_hits += len(set(gt.tolist()) & set(ret_ids))
        eps_hits += sum(1 for _, dist in res if dist <= (1 + epsilon) * d_star)
        total += K
    t_recall = time.time() - t0
    print(f"[Timing] Recall evaluation: {t_recall:.2f} s")

    recall_exact = exact_hits / total if total > 0 else None
    recall_eps = eps_hits / total if total > 0 else None

    # (c, r)-ANN evaluation
    print("[Eval] Computing (c,r)-ANN accuracy...")
    t0 = time.time()
    cr_ann_acc = evaluate_cr_ann(dataset_split, ann, queries, r, c=(1+epsilon), K=K)
    t_crann = time.time() - t0
    print(f"[Timing] (c,r)-ANN evaluation: {t_crann:.2f} s")

    mem_mb = ann.memory_breakdown()["total_MB"] if isinstance(ann.memory_breakdown(), dict) else ann.memory_breakdown()

    print("[Result] Memory breakdown (MB):", ann.memory_breakdown())
    print(f"[Result] Epsilon-recall: {recall_eps:.4f}")
    if cr_ann_acc is not None:
        print(f"[Result] (c,r)-ANN accuracy: {cr_ann_acc:.4f}")
    else:
        print("[Result] (c,r)-ANN accuracy: N/A (no queries within distance r)")

    # Phase timing summary
    print("\n[Timing Summary]")
    print(f"  Insert phase: {t_insert:.2f} s")
    print(f"  Query phase: {t_query:.2f} s")
    print(f"  Ground truth: {t_gt:.2f} s")
    print(f"  Recall eval: {t_recall:.2f} s")
    print(f"  (c,r)-ANN eval: {t_crann:.2f} s")
    print(f"  Total time: {t_insert + t_query + t_gt + t_recall + t_crann:.2f} s")

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
