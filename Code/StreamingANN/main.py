import numpy as np
import time 
from StreamingANN import StreamingANN  

# ----------------------------
# 1. Load points from file
# ----------------------------
points = np.loadtxt("points.txt")  # space-separated floats
print(f"Loaded {points.shape[0]} points with dimension {points.shape[1]}")

# ----------------------------
# 2. Initialize StreamingANN
# ----------------------------
d = points.shape[1]
n_estimate = len(points)       # upper bound on stream size
eta = 0.2                      # sampling parameter, 0 = store all points
r = 1.0                        # inner radius for LSH
epsilon = 0.5                  # approximation factor

ann = StreamingANN(d=d, n_estimate=n_estimate, eta=eta, r=r, epsilon=epsilon)

# ----------------------------
# 3. Insert points into ANN
# ----------------------------
for pt in points:
    ann.insert(pt)

print(f"Inserted points: {len(ann.points)}, Dropped points: {ann.dropped_points}")



# ----------------------------
# 4. Generate 100 queries in the appropriate range
# ----------------------------

num_queries = 100
R_query = 10.0  # cube side for queries
rng = np.random.default_rng(42)
queries = rng.uniform(0, R_query, size=(num_queries, d))

# ----------------------------
# 5. Query in the dataset (brute-force)
# ----------------------------
print("\n--- Brute-force dataset queries ---")
for i, q in enumerate(queries):
    start = time.time()
    dists = np.linalg.norm(points - q, axis=1)
    idx = np.argmin(dists)
    nearest = points[idx]
    elapsed = time.time() - start
    print(f"Query {i}: Nearest point = {nearest}, distance = {dists[idx]:.4f}, time = {elapsed:.6f}s")

# ----------------------------
# 6. Query in StreamingANN
# ----------------------------
print("\n--- StreamingANN queries ---")
for i, q in enumerate(queries):
    start = time.time()
    neighbor = ann.query(q)
    elapsed = time.time() - start
    if neighbor is not None:
        dist = np.linalg.norm(neighbor - q)
        print(f"Query {i}: Neighbor = {neighbor}, distance = {dist:.4f}, time = {elapsed:.6f}s")
    else:
        print(f"Query {i}: No neighbor found, time = {elapsed:.6f}s")