# to check the variation of mean relative error of sliding window RACE with window size
from Ang_hash_AKDE import RACE_Ah
from L2_hash_AKDE import RACE_L2, l2_lsh_collision_probability
import numpy as np
import time,random,gc,math,sys,os,argparse
from tqdm import tqdm
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

if __name__=="__main__":
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


    # Load data
    if args.file_name == 'text':
        files = ["data/encodings.npy", "data/encodings_2.npy", "data/encodings_3.npy", "data/encodings_4.npy"]
        arrays = [np.load(f) for f in files]
        data = np.vstack(arrays)
    else:
        data = np.load('data/hsi_data_points.npy')


    if args.n > data.shape[0]:
        raise ValueError(f"Requested num_data={args.n} but dataset has only {data.shape[0]}")


    data = data[:args.n].astype(np.float32)
    dim = data.shape[1]
    print(f"Data shape: {data.shape}")
 # parameters for our RACE in sliding window
    k = args.b
    num_data = args.n
    n_query = args.n_query
    eps = args.eps
    N_window = 450
    available_indices = list(range(0, num_data))
    if n_query > len(available_indices):
        raise ValueError("n_query larger than available data points")
    query_idx = random.sample(available_indices, n_query)
    query = data[query_idx]

    N=[2**i for i in range(6,12)] # list of window sizes
    eps_=2*eps+eps*eps # relative error of A-KDE


    print('---------Parameters for Sliding window RACE----------')
    print(f'Bandwidth parameter={k}, Relative error of EH={eps}, Data dimension {dim}')
    print(f'Relative error of A-KDE ={eps_:.6f}, Number of streaming data={num_data}, Number of queries={n_query}')
    print('----------------------------')

# computing true KDE in the sliding window setting
    true_kde=np.zeros((len(N),n_query)) # true kde for the last N data points 
    t0=time.time()
    if args.lsh == "1":
        for i in range(len(N)):
            true_kde[i,:] = compute_true_kde_angular(data, query, k, N[i])
    else:
        for i in range(len(N)):
            true_kde[i,:] = compute_true_kde_l2(data, query, k, N[i],args.w)
    print(f"True KDE calculation done in {time.time()-t0:.2f} s")
    
# number of rows in RACE structure to be used for the experiments
    n_row=[100,200,400,800,1600,3200]
    color_list=['b','g','r','c','m','y']
    
    # create a directory to store the graph
    dir_name = os.path.join(os.getcwd(), "Outputs", args.file_name)
    os.makedirs(dir_name, exist_ok=True)
    lb="Angular hash" if args.lsh=="1" else "L2 Hash"
    f_n=f"{dir_name}/Window_variation_{args.file_name}.pdf"
    print(f"Destination path {f_n}")
    plt.figure(figsize=(10,6))
    f=open(f"Window_data_{args.file_name}_{lb}.txt","w")
    for k_ in range(len(N)):
        err2=[] # list of log of mean relative errors for sliding window RACE
        print(f'\nWindow size {N[k_]}')
        f.write(f'Window size {N[k_]}\n')
        for i in range(len(n_row)):
            print(f'\nRows ={n_row[i]}')
            if args.lsh == "1":
                r_sketch = RACE_Ah(n_row[i], 2, k, dim, N[k_], eps)
            else:
                r_sketch = RACE_L2(n_row[i], args.r, k, dim, N[k_], args.w, eps)
            app_kde=np.zeros(n_query)
            print("Adding data to sketch")
            t0 = time.time()
            for j in range(num_data):
                r_sketch.update_counter(data[j], j + 1)
            print(f"Update time: {time.time()-t0:.2f}s")
            print("Querying the sketch")
            t0 = time.time()
            for j in range(n_query):
                app_kde[j] = r_sketch.query1(query[j])
            print(f"Query time: {time.time()-t0:.2f}s")

            eps_rel = 1e-12
            rel_err = np.mean(np.abs(app_kde - true_kde[k_,:]) / np.clip(np.abs(true_kde[k_,:]), eps_rel, None))
            print(f"Mean A-KDE={np.mean(app_kde):.6f}, Mean Relative Error={rel_err:.6f}")
            f.write(str(n_row[i])+","+str(np.log(rel_err))+'\n')
            err2.append(np.log(rel_err))
            del r_sketch # delete the sketch to free memory

# plotting the graphs   
        print(" In plot section ")
        l_name='N='+str(N[k_])
        plt.plot(n_row, err2, marker='*',mec='black',linestyle='-',color=color_list[k_],lw=1.75,label=l_name)
        del err2
        
    plt.xlabel('Number of rows')
    plt.ylabel('Log(Mean Relative Error)')
    plt.title('Mean Relative Error vs Number of Rows '+lb)
    plt.legend()
    # plt.show()
    plt.savefig(f_n)
    plt.close()
    f.close()
    gc.collect()