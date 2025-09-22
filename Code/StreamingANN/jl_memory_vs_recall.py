import numpy as np
import time
import argparse
from JL_baseline import StreamingJL  # your class
import gc

def load_fvecs_mmap(fname):
    """Memory-map .fvecs file (float32 vectors with dimension prefix)."""
    fv = np.memmap(fname, dtype=np.float32, mode='r')
    dim = fv[0].view(np.int32)  # first 4 bytes = dimension
    assert dim > 0 and dim < 20000, f"Unrealistic dimension: {dim}"
    fv = fv.reshape(-1, dim + 1)
    return fv[:, 1:]  # drop dimension prefix

def run_experiment(dataset, n_insert, n_queries, K, c, r, k, log_interval=100):
    insert_points = dataset[:n_insert]
    query_points = dataset[n_insert:n_insert+n_queries]

    jl = StreamingJL(d=dataset.shape[1], n_max=n_insert, c=c, r=r, k=k, random_state=42)

    # Insert points
    insert_times = []
    for i, x in enumerate(insert_points):
        t0 = time.time()
        jl.insert(x, i)
        t1 = time.time()
        insert_times.append(t1 - t0)
        if (i+1) % log_interval == 0:
            print(f"[Build] Inserted {i+1}/{n_insert} points, avg insert time: {np.mean(insert_times[-log_interval:])*1e3:.3f} ms")

    avg_insert_time = np.mean(insert_times)

    # ApproxRecall@K
    def approximate_recall(query_points, insert_points, jl, K, c, log_interval=10):
        recalls = []
        total_time = 0.0

        for i, q in enumerate(query_points, start=1):
            t0 = time.time()
            dists = np.linalg.norm(insert_points - q, axis=1)
            gt_topk = np.argsort(dists)[:K]
            rK = dists[gt_topk[-1]]
            approx = jl.query_topk(q, K)
            t1 = time.time()
            total_time += (t1 - t0)

            hits = sum(1 for pid, _ in approx if np.linalg.norm(insert_points[pid]-q) <= c*rK)
            recalls.append(hits / K)

            if i % log_interval == 0 or i == len(query_points):
                avg_recall = np.mean(recalls)
                avg_time = total_time / i
                print(f"[Recall] Processed {i}/{len(query_points)} queries, "
                    f"avg recall so far: {avg_recall:.4f}, avg query time: {avg_time*1e3:.3f} ms")

        return np.mean(recalls)
    
    recall = approximate_recall(query_points, insert_points, jl, K, c)

    # (c,r)-ANN    
    def ann_accuracy(query_points, dataset, ann, c, r, log_interval=10):
        successes, total_relevant = 0, 0
        total_time = 0.0

        for i, q in enumerate(query_points, start=1):
            t0 = time.time()
            dists = np.linalg.norm(dataset - q, axis=1)
            r_star = np.min(dists)
            t1 = time.time()
            total_time += (t1 - t0)

            if r_star <= r:
                total_relevant += 1
                t0_q = time.time()
                res = ann.query_topk(q, 1)
                t1_q = time.time()
                total_time += (t1_q - t0_q)

                if res is not None:
                    dist = np.linalg.norm(dataset[res[0][0]] - q)
                    if dist <= c*r:
                        successes += 1

            if i % log_interval == 0 or i == len(query_points):
                print(f"[ANN] Processed {i}/{len(query_points)} queries, "
                    f"valid queries so far: {total_relevant}, "
                    f"success rate: {successes/total_relevant if total_relevant>0 else 0:.4f}")

        if total_relevant == 0:
            return None
        return successes / total_relevant

    cr_acc = ann_accuracy(query_points, insert_points, jl, c, r)

    mem_mb = jl.memory_breakdown()["total_MB"]

    print(f"[SUMMARY] k={k}, c={c}, r={r}, K={K}, n_insert={n_insert}, n_queries={n_queries}, recall={recall:.4f}, (c,r)-ANN accuracy={cr_acc}, memory={mem_mb:.2f}")

    del jl
    import gc; gc.collect()

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
    else:
        dataset = np.load(args.file).astype(np.float32)

    norms = np.linalg.norm(dataset, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    dataset = dataset / norms   
    run_experiment(dataset, args.n_insert, args.n_queries, args.K, args.c, args.r, args.k)
