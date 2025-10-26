import numpy as np
import time
import argparse
from JL_baseline import StreamingJL 
import gc
import pandas as pd
import faiss

def load_fvecs_mmap(fname):
    """Memory-map .fvecs file (float32 vectors with dimension prefix)."""
    fv = np.memmap(fname, dtype=np.float32, mode='r')
    dim = fv[0].view(np.int32)  # first 4 bytes = dimension
    assert dim > 0 and dim < 20000, f"Unrealistic dimension: {dim}"
    fv = fv.reshape(-1, dim + 1)
    return fv[:, 1:]  # drop dimension prefix


def load_csv(fname):
    """Load CSV from data_csvs """
    arr = np.loadtxt(fname, delimiter=",", dtype=np.float32, skiprows=1)
    return arr


def build_faiss_index(data, metric=faiss.METRIC_L2):
    d = data.shape[1]
    index = faiss.IndexFlatL2(d) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(d)
    index.add(data.astype(np.float32))
    return index


def run_experiment(dataset, n_insert, n_queries, K, c, r, k, log_interval=100):
    total = len(dataset)
    n_tail = n_queries  # number of query points at the end

    if n_insert > total - n_tail:
        raise ValueError(f"Requested {n_insert} insert points, but only {total - n_tail} available (since last {n_tail} are reserved for queries).")

    # sample from the initial part (excluding the last 5000)
    rng = np.random.default_rng(42)
    insert_indices = rng.choice(total - n_tail, size=n_insert, replace=False)

    # select query points from the last 5000
    query_indices = np.arange(total - n_tail, total)

    insert_points = dataset[insert_indices].astype(np.float32)
    query_points = dataset[query_indices].astype(np.float32)

    jl = StreamingJL(d=dataset.shape[1], n_max=n_insert, c=c, r=r, k=k, random_state=42)

    # Insert points
    insert_times = []
    for i, x in enumerate(insert_points):
        t0 = time.time()
        jl.insert(x, i)
        t1 = time.time()
        insert_times.append(t1 - t0)
        if (i + 1) % log_interval == 0:
            print(f"[Build] Inserted {i+1}/{n_insert} points, "
                  f"avg insert time: {np.mean(insert_times[-log_interval:])*1e3:.3f} ms")

    avg_insert_time = np.mean(insert_times)

    # Build FAISS index for ground truth and ANN eval
    gt_index = build_faiss_index(insert_points, metric=faiss.METRIC_L2)

    # ApproxRecall@K using FAISS
    def approximate_recall(query_points, insert_points, jl, K, c, log_interval=10):
        recalls = []
        t0 = time.time()

        # Ground truth in one batched call
        D, I = gt_index.search(query_points, K)  # D = distances, I = indices
        rK = np.sqrt(D[:, -1])  # distance to K-th NN

        for i, (q, gt_ids, rK_q) in enumerate(zip(query_points, I, rK), start=1):
            approx = jl.query_topk(q, K)
            hits = sum(1 for pid, _ in approx
                       if np.linalg.norm(insert_points[pid] - q) <= c * rK_q)
            recalls.append(hits / K)

            if i % log_interval == 0 or i == len(query_points):
                avg_recall = np.mean(recalls)
                avg_time = (time.time() - t0) / i
                print(f"[Recall] {i}/{len(query_points)} queries, "
                      f"avg recall: {avg_recall:.4f}, avg query time: {avg_time*1e3:.3f} ms")

        return np.mean(recalls)

    recall = approximate_recall(query_points, insert_points, jl, K, c)

    # (c,r)-ANN accuracy using FAISS
    def ann_accuracy(query_points, dataset, ann, c, r, log_interval=10):
        successes, total_relevant = 0, 0
        t0 = time.time()

        # For each query, get NN distance (r*) quickly
        D, I = gt_index.search(query_points, 1)
        r_stars = np.sqrt(D[:, 0])

        for i, (q, r_star) in enumerate(zip(query_points, r_stars), start=1):
            if r_star <= r:
                total_relevant += 1
                res = ann.query_topk(q, 1)
                if res:
                    pid, _ = res[0]
                    dist = np.linalg.norm(dataset[pid] - q)
                    if dist <= c * r:
                        successes += 1

            if i % log_interval == 0 or i == len(query_points):
                print(f"[ANN] {i}/{len(query_points)} queries, "
                      f"valid={total_relevant}, "
                      f"success={successes/(total_relevant or 1):.4f}")

        t_total = time.time() - t0
        print(f"[Timing] (c,r)-ANN eval: {t_total:.2f} s "
              f"({t_total/len(query_points)*1e3:.3f} ms/query)")

        return None if total_relevant == 0 else successes / total_relevant

    cr_acc = ann_accuracy(query_points, insert_points, jl, c, r)

    mem_mb = jl.memory_breakdown()["total_MB"]

    print(f"[SUMMARY] k={k}, c={c}, r={r}, K={K}, "
          f"n_insert={n_insert}, n_queries={n_queries}, "
          f"recall={recall:.4f}, (c,r)-ANN={cr_acc}, memory={mem_mb:.2f} MB")

    del jl
    gc.collect()

    return recall, cr_acc, mem_mb
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--c", type=float, default=1.1)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--n_insert", type=int, default=1000)
    parser.add_argument("--n_queries", type=int, default=50)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    if args.file.endswith(".fvecs"):
        dataset = load_fvecs_mmap(args.file).astype(np.float32)
        norms = np.linalg.norm(dataset, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        dataset = dataset / norms  
    elif args.file.endswith(".csv"):
        dataset = load_csv(args.file)
        norms = np.linalg.norm(dataset, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        dataset = dataset / norms  
    else:
        dataset = np.load(args.file).astype(np.float32)

     
    run_experiment(dataset, args.n_insert, args.n_queries, args.K, args.c, args.r, args.k)
