import argparse
import time
import gc
import numpy as np
import pandas as pd
import faiss
from StreamingANN import StreamingANN
from JL_baseline import StreamingJL


# ---------- Utility loaders ----------

def load_csv(filename):
    arr = np.loadtxt(filename, delimiter=",", dtype=np.float32, skiprows=1)
    return arr


# ---------- Ground truth & recall utils ----------

def build_ground_truth(dataset, queries, K):
    index = faiss.IndexFlatL2(dataset.shape[1])
    index.add(dataset)
    D, I = index.search(queries, K)
    kth_dists = np.sqrt(D[:, -1])
    return I, kth_dists


def compute_recall(ann, dataset, queries, gt_topk, kth_dists, epsilon, K, log_prefix=""):
    exact_hits, eps_hits, total = 0, 0, 0
    q_times = []
    for qi, (q, gt, d_star) in enumerate(zip(queries, gt_topk, kth_dists)):
        t0 = time.time()
        res = ann.query_topk(q, K=K)
        q_times.append(time.time() - t0)
        ret_ids = [pid for pid, _ in res]
        exact_hits += len(set(gt.tolist()) & set(ret_ids))
        eps_hits += sum(1 for _, dist in res if dist <= (1 + epsilon) * d_star)
        total += K

        if (qi + 1) % 10 == 0:
            print(f"[Query] {log_prefix} processed {qi+1}/{len(queries)} queries")

    recall_eps = eps_hits / total if total > 0 else 0
    qps = len(queries) / np.sum(q_times)
    return recall_eps, qps


# ---------- StreamingANN Benchmark ----------

def run_experiment_ann(dataset, n_points, n_queries, K, epsilon, eta, r):
    n_total = dataset.shape[0]
    d = dataset.shape[1]
    rng = np.random.default_rng(0)

    if n_points + n_queries > n_total:
        raise ValueError("Not enough data for split.")

    insert_idx = rng.choice(n_total - n_queries, n_points, replace=False)
    query_idx = np.arange(n_total - n_queries, n_total)

    X_insert = dataset[insert_idx]
    X_query = dataset[query_idx]

    ann = StreamingANN(d=d, n_estimate=n_points, eta=eta, epsilon=epsilon, r=r, dataset=dataset)

    print(f"[Insert] Starting insertion of {len(insert_idx)} points...")
    for i, pid in enumerate(insert_idx):
        ann.insert(pid)
        if (i + 1) % 10 == 0:
            print(f"[Insert] Inserted {i+1}/{len(insert_idx)} points")

    print(f"[Insert] Done inserting {len(insert_idx)} points. Starting queries...")

    gt_topk, kth_dists = build_ground_truth(X_insert, X_query, K)
    recall_eps, qps = compute_recall(ann, X_insert, X_query, gt_topk, kth_dists, epsilon, K, log_prefix="ANN")

    mem_mb = ann.memory_breakdown()["total_MB"] if isinstance(ann.memory_breakdown(), dict) else ann.memory_breakdown()
    del ann
    gc.collect()

    return {"recall": recall_eps, "QPS": qps, "memory_MB": mem_mb}


# ---------- JL Baseline Benchmark ----------

def run_experiment_jl(dataset, n_insert, n_queries, K, c, r, k):
    total = len(dataset)
    if n_insert + n_queries > total:
        raise ValueError("Not enough points for insert/query split")

    rng = np.random.default_rng(42)
    insert_indices = rng.choice(total - n_queries, size=n_insert, replace=False)
    query_indices = np.arange(total - n_queries, total)

    insert_points = dataset[insert_indices].astype(np.float32)
    query_points = dataset[query_indices].astype(np.float32)

    jl = StreamingJL(d=dataset.shape[1], n_max=n_insert, c=c, r=r, k=k, random_state=42)

    print(f"[Insert] Starting insertion of {len(insert_points)} points...")
    for i, x in enumerate(insert_points):
        jl.insert(x, i)
        if (i + 1) % 10 == 0:
            print(f"[Insert] Inserted {i+1}/{len(insert_points)} points")

    print(f"[Insert] Done inserting {len(insert_points)} points. Starting queries...")

    gt_index = faiss.IndexFlatL2(dataset.shape[1])
    gt_index.add(insert_points)
    D, I = gt_index.search(query_points, K)
    rK = np.sqrt(D[:, -1])

    recalls, q_times = [], []
    for qi, (q, gt_ids, rK_q) in enumerate(zip(query_points, I, rK)):
        t0 = time.time()
        approx = jl.query_topk(q, K)
        q_times.append(time.time() - t0)
        hits = sum(1 for pid, _ in approx if np.linalg.norm(insert_points[pid] - q) <= c * rK_q)
        recalls.append(hits / K)

        if (qi + 1) % 10 == 0:
            print(f"[Query] JL processed {qi+1}/{len(query_points)} queries")

    recall = np.mean(recalls)
    qps = len(query_points) / np.sum(q_times)
    mem_mb = jl.memory_breakdown()["total_MB"]

    del jl
    gc.collect()

    return {"recall": recall, "QPS": qps, "memory_MB": mem_mb}


# ---------- Main entry ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--method", type=str, choices=["ann", "jl"], required=True)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--n_points", type=int, default=10000)
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--c", type=float, default=1.1)
    parser.add_argument("--k", type=int, default=128)
    args = parser.parse_args()

    for file in args.files:
        print(f"\n[Dataset] {file}")
        X = load_csv(file) if file.endswith(".csv") else np.load(file)

        if args.method == "ann":
            res = run_experiment_ann(X, args.n_points, args.n_queries, args.K, args.epsilon, args.eta, args.r)
        else:
            res = run_experiment_jl(X, args.n_points, args.n_queries, args.K, args.c, args.r, args.k)

        print(
            f"[Result] method={args.method.upper()} dataset={file} eta={args.eta} k={args.k} "
            f"recall={res['recall']:.4f}, QPS={res['QPS']:.2f}, memory={res['memory_MB']:.2f} MB"
        )


if __name__ == "__main__":
    main()
