import torch
import numpy as np
import random, math, time, gc,os
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from Exponential_Histogram import ExpHst  # keep as-is (CPU)
from p_stable import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RACE_2:
    def __init__(self, rows:int, hash_range:int, k:int, dim:int, N:int, width:int, eps=0.5):
        self.L = rows
        self.R = hash_range
        self.k = k
        self.winsize = N
        self.err = eps
        self.sparse_dic = {}
        self.hasher = UniversalHasher(self.R, seed=42)
        self.hash_list = []
        for _ in range(self.L):
            h = []
            for _ in range(k):
                h.append(PStableLSH(dim, width, 2))  # leave logic unchanged
            self.hash_list.append(h)

    def update_counter(self, data, t):
        for k, i in enumerate(self.hash_list):
            r = []
            for j in i:
                r.append(self.hasher.hash(j.hash(data)))
                s = sum(r)
            if (k, s) not in self.sparse_dic:
                self.sparse_dic[(k, s)] = ExpHst(self.winsize, math.ceil(1/self.err))
            else:
                self.sparse_dic[(k, s)].new_bucket(t)

    def query1(self, data):
        val = 0
        for k, i in enumerate(self.hash_list):
            r = []
            for j in i:
                r.append(self.hasher.hash(j.hash(data)))
                s = sum(r)
            if (k, s) in self.sparse_dic:
                val += self.sparse_dic[(k, s)].count_est()
        return val / self.L


def l2_lsh_collision_probability(d, w):
    if d == 0:
        return 1.0
    term1 = 1 - 2 * norm.cdf(-w / d)
    term2 = (2 * d / (np.sqrt(2 * np.pi) * w)) * (1 - np.exp(-(w**2) / (2 * d**2)))
    return term1 - term2


if __name__ == "__main__":
    files = ["data/encodings.npy", "data/encodings_2.npy", "data/encodings_3.npy", "data/encodings_4.npy"]
    arrays = [np.load(f) for f in files]
    data_np = np.vstack(arrays)
    data = torch.tensor(data_np, dtype=torch.float32, device=device)  # move to GPU

    dim = data.shape[1]
    print(f"Data shape: {data.shape}")

    k = 1
    num_data = 1000
    n_query = 100
    wd = 4
    eps = 0.1
    N = [500]
    eps_ = 2*eps + eps*eps
    R=1000
    print('---------Parameters for Sliding window RACE----------')
    print(f'Bandwidth parameter={k}, Window size={N[0]}, Relative error of EH={eps}, Data dimension {dim}')
    print(f'Relative error of A-KDE ={eps_:.6f}, Number of streaming data={num_data}, Number of queries={n_query}')
    print('----------------------------')
    random.seed(124)
    query = data[random.sample(range(15000, 20000), n_query), :]

    true_kde = np.zeros(n_query)
    st_time = time.time()
    for j in tqdm(range(n_query), desc="Traversing the data for true KDE"):
        for i in range(num_data - N[0], num_data):
            d = torch.linalg.norm(data[i, :] - query[j]).item()
            true_kde[j] += l2_lsh_collision_probability(d, w=wd)
    end_time = time.time()
    true_kde=true_kde/N[0]
    print(f"Time taken to compute true KDE for {n_query} queries is {end_time - st_time:.2f} seconds")
    print(f"Mean True KDE={np.mean(true_kde):.6f}")

    n_row = [100,200]#[100,200,300,400,500,600]
    err=[]
    for i in range(len(n_row)):
        print(f'Rows ={n_row[i]}')
        r_sketch = RACE_2(n_row[i], R, k, dim, N[0], wd, eps)
        app_kde = np.zeros(n_query)
        st_time = time.time()
        for j in tqdm(range(num_data), desc="Adding data to (SW) RACE sketch"):
            r_sketch.update_counter(data[j, :], j+1)
        end_time = time.time()
        st_time = time.time()
        for j in range(n_query):
            app_kde[j] = r_sketch.query1(query[j])*R/(N[0]*(R-1))-1/(R-1)
        end_time = time.time()
        rel_err = np.mean(abs((app_kde - true_kde)/true_kde))
        err.append(np.log(rel_err))
        print(f' Mean A-KDE={np.mean(app_kde):.6f} Mean Relative error={rel_err:.6f}\n')
        del r_sketch
    print(" In plot section ")
    os.makedirs("./Code/SlidingWindowKDE/Outputs/Plots", exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(n_row, err, marker='+',mec='blue',linestyle='-',color='red',lw=1.75, label='SW RACE')
    plt.xlabel('Number of Rows in RACE Sketch')
    plt.ylabel('Log(Mean Relative Error)')
    plt.title('Mean Relative Error vs Number of Rows')
    plt.legend()
    # plt.show()
    plt.savefig("Outputs/Plots/L2_hash_mean_relative_error_vs_rows.pdf")
    plt.close()
    gc.collect()
