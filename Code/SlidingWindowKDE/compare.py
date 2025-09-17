from RACE_19 import RACE_1
from RACE_AKDE import RACE_2
import numpy as np
import time,random,gc,math,sys,os
from tqdm import tqdm
import matplotlib.pyplot as plt

def angle_between_vectors(x, y): # find the angle in radians between two vectors x and y
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

if __name__=="__main__":
    files = ["data/encodings.npy", "data/encodings_2.npy", "data/encodings_3.npy", "data/encodings_4.npy"]
    arrays = [np.load(f) for f in files]
    data = np.vstack(arrays)
    dim=data.shape[1]
    print(f"Data shape: {data.shape}")


    k=1 # the bandwidth parameter of hash function
    num_data=10000 # number of streaming data
    n_query=1000 # number of queries
    print('---------Parameters for original RACE----------')
    print(f'Bandwidth parameter={k},  Data dimension {dim}')
    print(f'Number of streaming data={num_data}, Number of queries={n_query}')
    print('----------------------------')

    # parameters for our RACE in sliding window
    eps=0.1 # relative error for Exponential Histogram
    N=[300,450,600,1000,2000,5000] # list of window sizes
    eps_=2*eps+eps*eps # relative error of A-KDE


    print('---------Parameters for Sliding window RACE----------')
    print(f'Bandwidth parameter={k}, Window size={N[1]}, Relative error of EH={eps}, Data dimension {dim}')
    print(f'Relative error of A-KDE ={eps_:.6f}, Number of streaming data={num_data}, Number of queries={n_query}')
    print('----------------------------')
    random.seed(124)
    query=data[random.sample(range(15000,20000),n_query)] # n_query random queries from the data points

# computing true KDE in the normal setting 
    true_kde_1=np.zeros(n_query) # true kde for the n_query query points
    st_time=time.time()
    for j in tqdm(range(n_query), desc="Traversing the data for true KDE"):
        for i in range(num_data):
            true_kde_1[j]+=math.pow(1-1/np.pi*angle_between_vectors(data[i,:],query[j]),k) # calculating the collision probability

    end_time=time.time()
    print(f'Time taken to compute true KDE for {n_query} queries is {end_time-st_time:.2f} seconds')
    print(f'Mean True KDE={np.mean(true_kde_1):.6f}')

# computing true KDE in the sliding window setting 
    true_kde_2=np.zeros(n_query) # true kde for the last N data points
    st_time=time.time()
    for j in tqdm(range(n_query), desc="Traversing the data for true KDE"):
        for i in range(num_data-N[1],num_data):
            true_kde_2[j]+=math.pow(1-1/np.pi*angle_between_vectors(data[i,:],query[j]),k) # calculating the collision probability
    # print(f'True KDE={true_kde:.6f}')
    end_time=time.time()
    print(f'Time taken to compute true KDE for {n_query} queries is {end_time-st_time:.2f} seconds')
    print(f'Mean True KDE={np.mean(true_kde_2):.6f}')

# number of rows in RACE structure to be used for the experiments
    n_row=[100,200,300,400,500,600,700,800]
    
    err1=[] # list of log of mean relative errors for original RACE
    err2=[] # list of log of mean relative errors for sliding window RACE

    # sk_sz_1=[] # size of RACE sketch
    # sk_sz_2=[] # size of sliding window RACE sketch

    for i in range(len(n_row)):
        r_sketch=RACE_1(n_row[i],2,k,dim)# creating an instance of a RACE'19 sketch
        app_kde=np.zeros(n_query) # approximate kde
        print(f'Rows ={n_row[i]}')
        st_time=time.time()
        for j in tqdm(range(num_data), desc="Adding data to RACE sketch"):
            r_sketch.update_counter(data[j,:]) # updating the sketch with the streaming data  
        end_time=time.time()
        # print(f'Time taken to add {num_data} data points to RACE with R={n_row} is {end_time-st_time:.2f} seconds') 
        st_time=time.time()
        for j in tqdm(range(n_query),desc='Answering queries'):
            app_kde[j]=r_sketch.query(query[j]) # calculate the approximate kde from the race sketch
        end_time=time.time()
        # print(f'Time taken to answer {n_query} queries with R={n_row} is {end_time-st_time:.2f} seconds')
        rel_err=np.mean(abs((app_kde-true_kde_1)/true_kde_1)) # calculate relative error
        err1.append(np.log(rel_err))
        print(f'Mean A-KDE={np.mean(app_kde):.6f} Mean Relative error={rel_err:.6f}')
        # tmp=sys.getsizeof(r_sketch.sparse_dic)
        # sk_sz_1.append(tmp/1024)
        del r_sketch # delete the sketch to free memory
 
        r_sketch=RACE_2(n_row[i],2,k,dim,N[1],eps)# creating an instance of a SW RACE sketch
        
        st_time=time.time()
        for j in tqdm(range(num_data), desc="Adding data to (SW) RACE sketch"):
            r_sketch.update_counter(data[j,:],j+1) # updating the sketch with the streaming data  
        end_time=time.time()
        st_time=time.time()
        for j in range(n_query):
            app_kde[j]=r_sketch.query1(query[j]) # calculate the approximate kde from the race sketch
        end_time=time.time()
        # print(f'Time taken to answer {n_query} queries with R={n_row} is {end_time-st_time:.2f} seconds')
        rel_err=np.mean(abs((app_kde-true_kde_2)/true_kde_2)) # calculate relative error
        print(f'Mean A-KDE={np.mean(app_kde):.6f} Mean Relative error={rel_err:.6f}\n')
        # tmp=sys.getsizeof(r_sketch.sparse_dic)
        # sk_sz_2.append(tmp/1024)
        err2.append(np.log(rel_err))
        del r_sketch # delete the sketch to free memory

    # print(sk_sz_1)
    # print(sk_sz_2)
# plotting the graphs
# create a directory to store the graph
    print(" In plot section ")
    os.makedirs("./Code/SlidingWindowKDE/Outputs", exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(n_row, err1, marker='+',mec='blue',linestyle='-',color='red',lw=1.75, label='Original RACE')
    plt.plot(n_row, err2, marker='o',mec='green',linestyle='-',color='yellow',lw=1.75,label='Sliding Window RACE')
    plt.xlabel('Number of Rows in RACE Sketch')
    plt.ylabel('Log(Mean Relative Error)')
    plt.title('Mean Relative Error vs Number of Rows')
    plt.legend()
    # plt.show()
    plt.savefig("Outputs/mean_relative_error_vs_rows.pdf")
    plt.close()

    # plt.figure(figsize=(10,6))
    # plt.plot(n_row, sk_sz_1, marker='+',mec='blue',linestyle='-',color='red',lw=1.75, label='Original RACE')
    # plt.plot(n_row, sk_sz_2, marker='o', mec='green',linestyle='-',color='yellow',lw=1.75, label='Sliding Window RACE')
    # plt.xlabel('Number of Rows in RACE Sketch')
    # plt.ylabel('Sketch size (KB)')
    # plt.title('Sketch Size vs Number of Rows')
    # plt.legend()
    # plt.savefig("Outputs/sketch_size_vs_rows.pdf")
    # plt.close()

    gc.collect()