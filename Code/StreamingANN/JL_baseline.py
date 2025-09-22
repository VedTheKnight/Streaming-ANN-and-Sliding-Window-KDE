import numpy as np
import heapq
import time

# ---------------- JL Classes ----------------
class JLTransformer:
    def __init__(self, d, k, random_state=None):
        rng = np.random.RandomState(random_state)
        self.A = rng.normal(0, 1 / np.sqrt(k), size=(k, d)).astype(np.float32)

    def transform(self, x):
        return np.dot(self.A, x)


class StreamingJL:
    def __init__(self, d, n_max, c=2.0, r=1.0, delta=0.1, k=None, random_state=None):
        self.d = d
        self.n_max = n_max
        self.c = c
        self.r = r
        self.delta = delta
        self.eps = (c - 1) / (2 * c)
        if k is None:
            self.k = int(np.ceil(8 * np.log(n_max / delta) / (self.eps ** 2)))
        else:
            self.k = k
        self.jl = JLTransformer(d, self.k, random_state)
        self.points = []
        self.ids = []

    def insert(self, point, point_id=None):
        z = self.jl.transform(point)
        pid = point_id if point_id is not None else len(self.ids)
        self.ids.append(pid)
        self.points.append(z)

    def query_topk(self, q, K=10):
        zq = self.jl.transform(q)
        heap = []
        for pid, p in zip(self.ids, self.points):
            d = np.linalg.norm(zq - p)
            heapq.heappush(heap, (d, pid))
        topk = heapq.nsmallest(K, heap)
        return [(pid, dist) for dist, pid in topk]

    def memory_breakdown(self):
        breakdown = {}
        breakdown["projection_matrix_MB"] = self.jl.A.nbytes / (1024**2)
        if len(self.points) > 0:
            point_bytes = sum(32 * len(p) for p in self.points)  # crude estimate
        else:
            point_bytes = 0
        breakdown["projected_points_MB"] = point_bytes / (1024**2)
        id_bytes = len(self.ids) * 28
        breakdown["ids_MB"] = id_bytes / (1024**2)
        list_overhead = (len(self.points) + len(self.ids)) * 8
        breakdown["list_overhead_MB"] = list_overhead / (1024**2)
        breakdown["num_points"] = len(self.points)
        breakdown["dimension_reduced"] = self.k
        breakdown["total_MB"] = (
            breakdown["projection_matrix_MB"]
            + breakdown["projected_points_MB"]
            + breakdown["ids_MB"]
            + breakdown["list_overhead_MB"]
        )
        return breakdown


# ---------------- Evaluation ----------------
def approximate_recall(query_points, insert_points, jl, K, c):
    recalls = []
    total_time = 0.0

    for q in query_points:
        # Ground truth distances
        dists = np.linalg.norm(insert_points - q, axis=1)
        gt_topk = np.argsort(dists)[:K]
        rK = dists[gt_topk[-1]]  # distance to the K-th nearest neighbor

        # Approximate results with timing
        start = time.time()
        approx = jl.query_topk(q, K)
        end = time.time()
        total_time += (end - start)

        # Recall calculation
        hits = 0
        for pid, _ in approx:
            dist = np.linalg.norm(insert_points[pid] - q)
            if dist <= c * rK:
                hits += 1
        recalls.append(hits / K)

    avg_recall = np.mean(recalls)
    avg_query_time = total_time / len(query_points)

    return avg_recall, avg_query_time


def ann_accuracy(query_points, insert_points, jl, c, r):
    """Check (c, r)-ANN success rate."""
    successes = 0
    total_relevant = 0
    total_time = 0.0

    for q in query_points:
        # Ground truth nearest neighbor distance
        dists = np.linalg.norm(insert_points - q, axis=1)
        r_star = np.min(dists)

        if r_star <= r:
            # print("valid query")
            total_relevant += 1
            start = time.time()
            approx = jl.query_topk(q, 1)
            end = time.time()
            total_time += (end - start)

            # Check if ANN condition satisfied
            found = False
            for pid, _ in approx:
                dist = np.linalg.norm(insert_points[pid] - q)
                if dist <= c * r:
                    found = True
                    break

            if found:
                successes += 1
        else:
            pass
            # print(r_star, r)
    if total_relevant == 0:
        return None, None  # no queries with a true neighbor within r
    ann_success_rate = successes / total_relevant
    avg_query_time = total_time / max(total_relevant, 1)

    return ann_success_rate, avg_query_time


# ---------------- Experiment ----------------
if __name__ == "__main__":
    # Experiment parameters
    n_insert_points = 1000
    n_queries = 50
    K = 10
    c = 1.1
    r = 1.0
    delta = 0.1
    log_interval = 100
    random_state = 42

    rng = np.random.default_rng(random_state)

    dataset = np.load("data/encodings_combined.npy").astype(np.float32)
    n_total, d = dataset.shape
    print(f"Loaded dataset with {n_total} points, dimension {d}")
    insert_points = dataset[:n_insert_points]
    query_points = dataset[n_insert_points:n_insert_points+n_queries]

    # Compute JL bound
    eps_jl = (c - 1) / (2 * c)
    k_bound = d  # int(np.ceil(8 * np.log(n_insert_points / delta) / (eps_jl ** 2)))

    # Sweep around JL bound
    k_values = [int(0.2 * k_bound), int(0.4 * k_bound), int(0.5 * k_bound),
                int(0.7 * k_bound), int(0.8 * k_bound), int(0.9 * k_bound)]
    results = []

    for k in k_values:
        print(f"\n=== Running experiment for k={k} ===")
        jl = StreamingJL(d=d, n_max=n_insert_points, c=c, r=r,
                         delta=delta, k=k, random_state=random_state)

        insert_times = []
        for i, x in enumerate(insert_points, start=0):
            t0 = time.time()
            jl.insert(x, i)
            t1 = time.time()
            insert_times.append(t1 - t0)

            if i % log_interval == 0:
                avg_time = np.mean(insert_times[-log_interval:])
                print(f"Inserted {i} points, avg time per insert (last {log_interval}): {avg_time:.6f}s")

        avg_insert_time = np.mean(insert_times)

        # Approximate recall (with avg query time)
        avg_recall, avg_query_time = approximate_recall(query_points, insert_points, jl, K, c)

        # (c, r)-ANN accuracy
        ann_success, ann_qtime = ann_accuracy(query_points, insert_points, jl, c, r)

        print(jl.memory_breakdown())
        mem = jl.memory_breakdown()["total_MB"]
        results.append((k, mem, avg_recall, avg_insert_time, avg_query_time,
                        ann_success, ann_qtime))

        print(f"[k={k}] Total Memory={mem:.2f} MB, "
              f"ApproxRecall@{K}={avg_recall:.4f}, "
              f"(c,r)-ANN Success={ann_success if ann_success is not None else 'N/A'}, "
              f"AvgInsertTime={avg_insert_time:.6f}s, "
              f"AvgQueryTime={avg_query_time*1000:.2f} ms, "
              f"ANNQueryTime={ann_qtime*1000 if ann_qtime else 0:.2f} ms")

    # Final table
    print("\n--- Memory vs ApproxRecall vs ANN Success vs Time ---")
    print("k\tMemory(MB)\tApproxRecall@K\t(c,r)-ANN\tAvgInsertTime(s)\tAvgQueryTime(ms)\tANNQueryTime(ms)")
    for k, mem, approx, avg_insert, avg_q_time, ann_success, ann_qtime in results:
        ann_str = f"{ann_success:.4f}" if ann_success is not None else "N/A"
        ann_qtime_str = f"{ann_qtime*1000:.2f}" if ann_qtime is not None else "N/A"
        print(f"{k}\t{mem:.2f}\t{approx:.4f}\t{ann_str}\t{avg_insert:.6f}\t{avg_q_time*1000:.2f}\t{ann_qtime_str}")
