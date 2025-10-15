import argparse
import random
import time
import gc
import sys
import os
import math
import numpy as np
from tqdm import tqdm
from Ang_hash_AKDE import RACE_Ah
from L2_hash_AKDE import RACE_L2, l2_lsh_collision_probability
from graph_plotter import plot_from_textfile
import matplotlib.pyplot as plt


# reproducible
SEED = 126
random.seed(SEED)
np.random.seed(SEED)


def compute_true_kde_angular(data, query, k, window_size):
    """Compute true KDE for angular LSH using numpy."""
    window = data[-window_size:] # (N, dim)
    # normalize
    window_norm = np.linalg.norm(window, axis=1, keepdims=True).clip(min=1e-12)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
    cosines = (query @ window.T) / (query_norm * window_norm.T)
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.arccos(cosines)
    kde = np.sum((1.0 - angles / math.pi) ** k, axis=1)
    return kde

def compute_true_kde_l2(data, query, k, window_size, wd):
    """Compute true KDE for L2 LSH using numpy."""
    window = data[-window_size:]
    q_norm2 = np.sum(query ** 2, axis=1, keepdims=True)
    w_norm2 = np.sum(window ** 2, axis=1, keepdims=True).T
    sqdist = np.clip(q_norm2 + w_norm2 - 2.0 * (query @ window.T), a_min=0.0, a_max=None)
    dists = np.sqrt(sqdist)
    vec_fn = np.vectorize(lambda x: l2_lsh_collision_probability(float(x), wd))
    probs = vec_fn(dists)
    kde = np.sum(probs, axis=1)
    return kde


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lsh", choices=["1", "2"], required=True)
    parser.add_argument("--w", type=int, default=4)
    parser.add_argument("--r", type=int, default=1000)
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--eps", type=float, default=0.05)
    args = parser.parse_args()


    for data_idx in range(50):
        # Load data
        data=np.load(f'synthetic_data/data_{data_idx+1}.npy')
        print("\nData loaded successfully")
        data = data.astype(np.float32)
        dim = data.shape[1]
        print(f"Data shape: {data.shape}")
        k = args.b
        num_data = data.shape[0]
        n_query = 100
        eps = args.eps
        N_window = 450
        available_indices = list(range(0, num_data))
        if n_query > len(available_indices):
            raise ValueError("n_query larger than available data points")
        query_idx = random.sample(available_indices, n_query)
        query = data[query_idx]
        del available_indices

        t0 = time.time()
        if args.lsh == "1":
            true_kde = compute_true_kde_angular(data, query, k, N_window)
        else:
            true_kde = compute_true_kde_l2(data, query, k, N_window, args.w)

        print(f"Mean True KDE={np.mean(true_kde):.6f} (time {time.time()-t0:.2f}s)")
        n_row_values = [100*2**i for i in range(0,6)]
        sk_sz = []
        err_log = []
  
        dir_name ="Synthetic_data_outputs_L2"
        os.makedirs(dir_name, exist_ok=True)
        lb="Angular" if args.lsh=="1" else "L2"
        # f_n=f"error_vs_sz_{args.file_name}_{lb}.txt"
        # with open(f_n,"w") as f:
        #     f.write(str(len(n_row_values))+"\n")
        #     f.write("Log Mean Error\nSketch size in KB\n"+lb+"\n")
        #     f.write("Mean Relative Error vs Sketch size\n")
        #     f.write(os.path.join(dir_name, f"error_vs_sz_{lb}.pdf") + "\n")

        for rows in n_row_values:
            print(f"\nRows {rows}")
            app_kde = np.zeros(n_query, dtype=np.float32)
            if args.lsh == "1":
                r_sketch = RACE_Ah(rows, 2, k, dim, N_window, eps)
            else:
                r_sketch = RACE_L2(rows, args.r, k, dim, N_window, args.w, eps)
            print("Adding data to sketch")
            t0 = time.time()
            for j in range(num_data):
                r_sketch.update_counter(data[j], j + 1)
            print(f"Update time: {time.time()-t0:.2f}s")
            print("Querying the sketch")
            t0 = time.time()
            for i in range(n_query):
                app_kde[i] = r_sketch.query1(query[i])
            print(f"Query time: {time.time()-t0:.2f}s")

            eps_rel = 1e-12
            rel_err = np.mean(np.abs(app_kde - true_kde) / np.clip(np.abs(true_kde), eps_rel, None))
            print(f"Mean A-KDE={np.mean(app_kde):.6f}, Mean Relative Error={rel_err:.6f}")
            total_bytes = sum(v.nbytes if hasattr(v, 'nbytes') else np.array(v).nbytes for v in r_sketch.sparse_dic.values())
            print(f"Size ={total_bytes/1024}")
            err_log.append(math.log(rel_err+1e-16))
            sk_sz.append(total_bytes/1024)
            del r_sketch
        np.save(f'{dir_name}/err_values_{data_idx+1}.npy',np.array(err_log))
        np.save(f'{dir_name}/sketch_sizes_{data_idx+1}.npy',np.array(sk_sz))    
        del err_log
        del sk_sz

    gc.collect()
