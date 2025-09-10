import numpy as np
import time

class JLTransformer:
    def __init__(self, d, k, random_state=None):
        rng = np.random.RandomState(random_state)
        self.A = rng.normal(0, 1/np.sqrt(k), size=(k, d))

    def transform(self, x):
        return np.dot(self.A, x)


class StreamingJL:
    def __init__(self, d, n_max, c=2.0, r=1.0, delta=0.1, random_state=None):
        self.d = d
        self.n_max = n_max
        self.c = c
        self.r = r
        self.delta = delta

        self.eps = (c - 1) / (2 * c) # this is a parameter for JL 
        self.k = int(np.ceil(8 * np.log(n_max / delta) / (self.eps ** 2)))

        self.jl = JLTransformer(d, self.k, random_state)
        self.points = []
        self.ids = []


    def insert(self, point, point_id=None):
        z = self.jl.transform(point)
        pid = point_id if point_id is not None else len(self.ids)

        self.ids.append(pid)
        self.points.append(z)

    def query(self, q):
        zq = self.jl.transform(q)

        # linear scan in reduced dimension
        best_id, best_dist = None, float("inf")
        for pid, p in zip(self.ids, self.points):
            d = np.linalg.norm(zq - p)
            if d < best_dist and d <= (1 + self.eps) * self.c * self.r:
                best_id, best_dist = pid, d
        return best_id

# # ------------------------------
# # Test setup
# # ------------------------------
# np.random.seed(42)
# d = 5000
# n_max = 5000
# points = np.random.randn(n_max, d)

# # Initialize two ANN structures
# ann_brute = StreamingJL(d, n_max, c=1.5, r=1.0, delta = 0.1, backend="brute", random_state=42)

# # Insert points
# def time_insert(ann, points):
#     start = time.time()
#     for i, p in enumerate(points):
#         ann.insert(p, i)
#     end = time.time()
#     return (end - start) / len(points)  # avg insert time

# t_insert_brute = time_insert(ann_brute, points)

# print(f"Avg insert time (brute): {t_insert_brute*1e6:.2f} µs")

# # ------------------------------
# # Generate queries with YES outcome
# # ------------------------------
# n_queries = 100
# queries = []
# true_ids = []

# for _ in range(n_queries):
#     idx = np.random.randint(0, n_max)
#     base = points[idx]
#     noise = np.random.randn(d)
#     noise = noise / np.linalg.norm(noise)  # unit vector
#     q = base + 0.5 * noise   # ensures distance = 0.5 < r=1
#     queries.append(q)
#     true_ids.append(idx)

# queries = np.array(queries)

# # ------------------------------
# # Evaluation
# # ------------------------------
# def evaluate(ann, points, queries, r, c):
#     n_correct = 0
#     start = time.time()
#     for q in queries:
#         ans_id = ann.query(q)
#         if ans_id is None:
#             continue
#         dist = np.linalg.norm(points[ans_id] - q)
#         if dist <= c * r:
#             n_correct += 1
#     end = time.time()
#     return n_correct / len(queries), (end - start) / len(queries)

# acc_brute, t_query_brute = evaluate(ann_brute, points, queries, r=1.0, c=1.5)

# print(f"Brute: success rate={acc_brute:.2f}, avg query={t_query_brute*1e6:.2f} µs")
