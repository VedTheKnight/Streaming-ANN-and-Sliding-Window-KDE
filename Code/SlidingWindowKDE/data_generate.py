import numpy as np
import os

os.makedirs('synthetic_data',exist_ok=True)

n_data=10000
chunks=1000
dim=200

n_sim=50
np.random.seed(465)

for num in range(n_sim):

    data=np.zeros((n_data, dim), dtype=np.float32)
    for i in range(n_data//chunks):
        mean = np.random.rand(dim)
        cov = np.diag(np.random.rand(dim))
        tmp=np.random.multivariate_normal(mean,cov,chunks)
        data[i*chunks:(i+1)*chunks,:]=tmp 

    d_n=f'synthetic_data/data_{num+1}.npy'
    np.save(d_n,data)
    print("data saved")
