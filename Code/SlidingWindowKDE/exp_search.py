from RACE_AKDE import *
import numpy as np
import random,gc,sys,time
from tqdm import tqdm
import matplotlib.pyplot as plt

def angle_between_vectors(x, y):
    # Ensure input is numpy array
    x = np.array(x)
    y = np.array(y)
    # Compute dot product and norms
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    # Avoid division by zero
    if norm_x == 0 or norm_y == 0:
        raise ValueError("Zero vector has no defined angle.")
    # Compute cosine of angle
    cos_theta = dot_product / (norm_x * norm_y)
    # Clamp value to avoid numerical errors outside [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # Compute angle in radians
    angle = np.arccos(cos_theta)
    return angle

# Load data
files = ["data/encodings.npy", "data/encodings_2.npy", "data/encodings_3.npy", "data/encodings_4.npy"]
arrays = [np.load(f) for f in files]
data = np.vstack(arrays)
print(f"Data shape: {data.shape}")
# data=np.load('encodings.npy')
# data=np.random.normal(25,10,(10000,500)) # for random data

dim=data.shape[1]


eps=0.1 # relative error for Exponential Histogram
k=1 # the bandwidth parameter of hash function
N=450 # window size
num_data=10000 # number of streaming data
eps_=2*eps+eps*eps # relative error of A-KDE
n_query=1000 # number of queries

print('---------Parameters----------')
print(f'Bandwidth parameter={k}, Window size={N}, Relative error of EH={eps}, Data dimension {dim}')
print(f'Relative error of A-KDE ={eps_}, Number of streaming data={num_data}, Number of queries={n_query}')
print('----------------------------')
query=data[random.sample(range(71000,80000),n_query)] # n_query random queries from the data points

# query=np.random.randn(dim) # random query
n_row=2560 # number of rows initialized to 1

true_kde=np.zeros(n_query) # true kde for the last N data points
st_time=time.time()
for j in tqdm(range(n_query), desc="Traversing the data for true KDE"):
    for i in range(num_data-N,num_data):
        true_kde[j]+=math.pow(1-1/np.pi*angle_between_vectors(data[i,:],query[j]),k) # calculating the collision probability
# print(f'True KDE={true_kde:.6f}')
end_time=time.time()
print(f'Time taken to compute true KDE for {n_query} queries is {end_time-st_time:.2f} seconds')
true_kde=true_kde # true kde
print(f'Mean True KDE={np.mean(true_kde):.6f}')
sketch_sizes=[]
mean_error=[]
for i in range(1,10):
    r_sketch=RACE(n_row,2,k,dim,N,eps)# creating an instance of a RACE sketch
    app_kde=np.zeros(n_query) # approximate kde
    
    st_time=time.time()
    for j in tqdm(range(num_data), desc="Adding data to RACE sketch"):
        r_sketch.update_counter(data[j,:],j+1) # updating the sketch with the streaming data  
    end_time=time.time()
    # print(f'Time taken to add {num_data} data points to RACE with R={n_row} is {end_time-st_time:.2f} seconds') 

    st_time=time.time()
    for j in range(n_query):
        app_kde[j]=r_sketch.query1(query[j]) # calculate the approximate kde from the race sketch
    end_time=time.time()
    # print(f'Time taken to answer {n_query} queries with R={n_row} is {end_time-st_time:.2f} seconds')
    rel_err=np.mean(abs((app_kde-true_kde)/true_kde)) # calculate relative error
    print(f'R={n_row} Mean A-KDE={np.mean(app_kde):.6f} Mean Relative error={rel_err:.6f}')
    size_sketch=sys.getsizeof(r_sketch.sparse_dic)
    sketch_sizes.append(size_sketch/1024)
    mean_error.append(np.log(rel_err))
    del r_sketch # delete the sketch to free memory
    if rel_err<= eps_:
        print('Relative error below threshold')
        # break
    print()
    n_row*=2 # doubling the number of rows
    break # for debugging purpose only one iteration is done

plt.figure(figsize=(10,6))
plt.plot(sketch_sizes,mean_error,marker='+',mec='blue',mew=5,linestyle='-.',color='red',lw=2.5,label='For k=1')
plt.xlabel('Sketch size (KB)')
plt.ylabel("Log(Mean Relative Error)")
plt.title("Effect of sketch size on mean relative error")
plt.legend()
plt.show()
gc.collect()
