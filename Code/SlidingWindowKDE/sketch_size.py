# to check the variation of mean relative error of sliding window RACE with sketch size
import argparse, random, time, gc, sys, torch
from tqdm import tqdm
from Ang_hash_AKDE import RACE_Ah
from L2_hash_AKDE import RACE_L2, l2_lsh_collision_probability

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_true_kde_angular(data, query, k, window_size):
    """
    Compute true KDE for angular LSH using vectorized PyTorch operations.
    data: (num_data, dim) tensor
    query: (n_query, dim) tensor
    k: bandwidth parameter
    window_size: number of last points to consider
    """
    window = data[-window_size:]  # shape (N, dim)
    # normalize to unit vectors
    window_norm = torch.norm(window, dim=1, keepdim=True).clamp_min(1e-12)
    query_norm = torch.norm(query, dim=1, keepdim=True).clamp_min(1e-12)

    # cosine similarities: (n_query, N)
    cosines = (query @ window.T) / (query_norm * window_norm.T)
    cosines = cosines.clamp(-1.0, 1.0)
    angles = torch.acos(cosines)
    kde = torch.sum((1.0 - angles / torch.pi) ** k, dim=1)
    return kde

def compute_true_kde_l2(data, query, k, window_size, wd):
    """
    Compute true KDE for L2 LSH using vectorized PyTorch ops.
    """
    window = data[-window_size:]  # (N, dim)
    # pairwise squared distances
    q_norm2 = torch.sum(query ** 2, dim=1, keepdim=True)  # (nq, 1)
    w_norm2 = torch.sum(window ** 2, dim=1).unsqueeze(0)  # (1, N)
    sqdist = (q_norm2 + w_norm2 - 2.0 * (query @ window.T)).clamp_min(0.0)
    dists = torch.sqrt(sqdist)  # (nq, N)

    # apply l2_lsh_collision_probability elementwise
    # (this function is likely scalar-based, so we vectorize via Python)
    probs = torch.empty_like(dists)
    for i in range(dists.shape[0]):
        # convert to numpy for compatibility if function is not torch-based
        d_np = dists[i].cpu().numpy()
        probs[i] = torch.from_numpy(
            l2_lsh_collision_probability(d_np, wd)
        ).to(device)
    kde = probs.sum(dim=1)
    return kde

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", choices=['text', 'image'], required=True)
    parser.add_argument("--n", type=int, required=True, help="Number of streaming data")
    parser.add_argument("--n_query", type=int, required=True, help="Number of queries")
    parser.add_argument("--lsh", choices=["1", "2"], required=True)
    parser.add_argument("--w", type=int, default=4)
    parser.add_argument("--r", type=int, default=1000)
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--eps", type=float, default=0.05)
    args = parser.parse_args()

    # Load data into torch
    if args.file_name == 'text':
        import numpy as np
        files = ["data/encodings.npy", "data/encodings_2.npy", "data/encodings_3.npy", "data/encodings_4.npy"]
        arrays = [np.load(f) for f in files]
        data_np = np.vstack(arrays)
    else:
        import numpy as np
        data_np = np.load('data/hsi_data_points.npy')

    data = torch.from_numpy(data_np).float().to(device)
    dim = data.shape[1]
    print(f"Data shape: {tuple(data.shape)}")

    k = args.b
    num_data = args.n
    n_query = args.n_query
    eps = args.eps
    N = [450]
    eps_ = 2 * eps + eps * eps

    # pick queries safely
    rng_start = max(0, min(15000, data.shape[0] - n_query - 1))
    query_idx = random.sample(range(rng_start, min(data.shape[0], rng_start + 5000)), n_query)
    query = data[query_idx]

    # compute true KDE
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    if args.lsh == "1":
        true_kde = compute_true_kde_angular(data[:num_data], query, k, N[0])
    else:
        true_kde = compute_true_kde_l2(data[:num_data], query, k, N[0], args.w)
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"Mean True KDE={true_kde.mean().item():.6f} (time {time.time()-t0:.2f}s)")

    # experiment loop
    n_row_values = torch.arange(100, 2000, 50)
    sk_sz = []
    err = []

    for rows in n_row_values:
        print(f"Rows {rows.item()}")
        app_kde = torch.zeros(n_query, device=device)

        if args.lsh == "1":
            r_sketch = RACE_Ah(rows.item(), 2, k, dim, N[0], eps)
        else:
            r_sketch = RACE_L2(rows.item(), args.r, k, dim, N[0], args.w, eps)

        # update sketch
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        for j in tqdm(range(num_data), desc="Updating sketch"):
            r_sketch.update_counter(data[j], j + 1)
        torch.cuda.synchronize() if device.type == "cuda" else None
        print(f"Update time: {time.time()-t0:.2f}s")

        # query sketch
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        for i in range(n_query):
            app_kde[i] = r_sketch.query1(query[i])
        torch.cuda.synchronize() if device.type == "cuda" else None
        print(f"Query time: {time.time()-t0:.2f}s")

        # compute relative error safely
        eps_rel = 1e-12
        rel_err = torch.mean(torch.abs(app_kde - true_kde) / (true_kde.abs().clamp_min(eps_rel)))
        print(f"Mean A-KDE={app_kde.mean().item():.6f}, Mean Relative Error={rel_err.item():.6f}")

        # memory size estimate
        total_bytes = 0
        for v in getattr(r_sketch, "sparse_dic", {}).values():
            if isinstance(v, torch.Tensor):
                total_bytes += v.element_size() * v.nelement()
            else:
                total_bytes += sys.getsizeof(v)
        sk_sz.append(total_bytes / 1024)
        err.append(torch.log(rel_err + 1e-16).item())

        del r_sketch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


# plotting the graphs
# create a directory to store the graph
    print(" In plot section ")
    current_dir = os.getcwd()
    dir_name="Outputs/"+arg.file_name
    os.makedirs(os.path.join(current_dir,dir_name),exist_ok=True)
    lb="Angular hash" if arg.lsh=="1" else "L2 Hash"
    plt.plot(sk_sz, err, marker='+',mec='blue',linestyle='-',color='red',lw=1.75,label=lb)
    del err
        
    plt.xlabel('Sketch size (KB)')
    plt.ylabel('Log(Mean Relative Error)')
    plt.title('Mean Relative Error vs Sketch size')
    plt.legend()
    f_n=dir_name+"/Effect_of_sketch_size.pdf"
    plt.savefig(f_n)
    plt.close()
    gc.collect()