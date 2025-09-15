import argparse, gc, sys, time, os
import numpy as np
from StreamingANN import StreamingANN
import psutil
import pandas as pd

def count_points(files):
    total = 0
    lengths = []
    for f in files:
        arr = np.load(f, mmap_mode="r")
        lengths.append(arr.shape[0])
        total += arr.shape[0]
    return total, lengths

def sample_points(files, lengths, indices):
    start = 0
    idx_set = set(indices)
    for f, L in zip(files, lengths):
        arr = np.load(f, mmap_mode="r")
        for i in range(L):
            gid = start + i
            if gid in idx_set:
                yield arr[i]
        start += L

def ground_truth_topk(dataset, queries, K):
    res, kth_dists = [], []
    dataset = np.array(dataset)
    for q in queries:
        dists = np.linalg.norm(dataset - q, axis=1)
        topk = np.argsort(dists)[:K]
        res.append(topk)
        kth_dists.append(dists[topk[-1]])
    return res, np.array(kth_dists)

def build_ann(files, lengths, idx, r, epsilon, K, eta, queries, log_interval=100, save_dir="results"):
    d = queries.shape[1]
    ann = StreamingANN(d=d, n_estimate=len(idx), eta=eta, epsilon=epsilon, r=r)
    total_points = len(idx)
    process = psutil.Process(os.getpid())

    # Prepare logging arrays
    memory_usage = []
    avg_insert_time = []
    stored_points_list = []

    gen = sample_points(files, lengths, idx)
    insert_times = []

    start_total = time.time()
    for i in range(total_points):
        t0 = time.time()
        p = next(gen)
        t1 = time.time()
        insert_start = time.time()
        ann.insert(p)
        insert_end = time.time()
        insert_times.append(insert_end - insert_start)

        # Track stats every log_interval
        if (i + 1) % log_interval == 0 or (i + 1) == total_points:
            mem_now = process.memory_info().rss
            memory_usage.append(mem_now / 1024 / 1024)  # MB
            avg_insert_time.append(np.mean(insert_times) * 1e3)  # ms
            stored_points_list.append(len(ann.points))
            insert_times = []

    # Ground truth for recall
    dataset_split = np.array([p for p in sample_points(files, lengths, idx)])
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

    # Save memory & timing results
    folder_name = f"eta{eta}_epsilon{epsilon}_r{r}_K{K}_n{len(idx)}"
    save_path = os.path.join(save_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame({
        "stored_points": stored_points_list,
        "memory_MB": memory_usage,
        "avg_insert_time_ms": avg_insert_time
    })
    df.to_csv(os.path.join(save_path, "memory_insert_stats.csv"), index=False)

    total_memory = memory_usage[-1]
    del ann
    gc.collect()
    return total_memory, recall_eps

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
    queries = np.array([p for p in sample_points(args.files, lengths, q_idx)])

    total_memory, recall_eps = build_ann(
        args.files, lengths, idx,
        args.r, args.epsilon, args.K, args.eta, queries
    )

    print(f"Total memory used: {total_memory:.2f} MB")
    print(f"Epsilon-recall: {recall_eps:.4f}")

if __name__ == "__main__":
    main()
