# to check the variation of mean relative error of sliding window RACE with window size

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

    # parameters for our RACE in sliding window
    eps=0.1 # relative error for Exponential Histogram
    N=[300,450,600,1000,2000,5000] # list of window sizes
    eps_=2*eps+eps*eps # relative error of A-KDE


    print('---------Parameters for Sliding window RACE----------')
    print(f'Bandwidth parameter={k}, Relative error of EH={eps}, Data dimension {dim}')
    print(f'Relative error of A-KDE ={eps_:.6f}, Number of streaming data={num_data}, Number of queries={n_query}')
    print('----------------------------')
    random.seed(124)
    query=data[random.sample(range(15000,20000),n_query)] # n_query random queries from the data points

# computing true KDE in the sliding window setting 
    true_kde=np.zeros((len(N),n_query)) # true kde for the last N data points
    st_time=time.time()
    for k_ in tqdm(range(len(N)),desc="Traversing the data for true KDE"):
        for j in range(n_query):
            for i in range(num_data-N[k_],num_data):
                true_kde[k_,j]+=math.pow(1-1/np.pi*angle_between_vectors(data[i,:],query[j]),k) # calculating the collision probability
    # print(f'True KDE={true_kde:.6f}')
    end_time=time.time()
    print(f'Time taken to compute true KDE for {n_query} queries is {end_time-st_time:.2f} seconds')
    print(f'Mean True KDE={np.mean(true_kde,axis=1)}')

# number of rows in RACE structure to be used for the experiments
    n_row=[100,200,300,400,500,600,700,800]
    color_list=['b','g','r','c','m','y']
    
    # create a directory to store the graph
    
    os.makedirs("./Code/SlidingWindowKDE/Outputs", exist_ok=True)
    plt.figure(figsize=(10,6))
    for k_ in range(len(N)):
        err2=[] # list of log of mean relative errors for sliding window RACE
        print(f'window size {N[k_]}')
        for i in range(len(n_row)):
            print(f'Rows ={n_row[i]}')
            r_sketch=RACE_2(n_row[i],2,k,dim,N[k_],eps)# creating an instance of a SW RACE sketch
            app_kde=np.zeros(n_query)
            st_time=time.time()
            for j in tqdm(range(num_data), desc="Adding data to (SW) RACE sketch"):
                r_sketch.update_counter(data[j,:],j+1) # updating the sketch with the streaming data  
            end_time=time.time()
            st_time=time.time()
            for j in range(n_query):
                app_kde[j]=r_sketch.query1(query[j]) # calculate the approximate kde from the race sketch
            end_time=time.time()
            # print(f'Time taken to answer {n_query} queries with R={n_row} is {end_time-st_time:.2f} seconds')
            rel_err=np.mean(abs((app_kde-true_kde[k_,:])/true_kde[k_,:])) # calculate relative error
            print(f'Mean A-KDE={np.mean(app_kde):.6f} Mean Relative error={rel_err:.6f}\n')
            err2.append(np.log(rel_err))
            del r_sketch # delete the sketch to free memory


# plotting the graphs

        print(" In plot section ")
        l_name='N='+str(N[k_])
        plt.plot(n_row, err2, marker='+',mec='lime',linestyle='-',color=color_list[k_],lw=1.75,label=l_name)
        del err2
        
    plt.xlabel('Number of rows')
    plt.ylabel('Log(Mean Relative Error)')
    plt.title('Mean Relative Error vs Number of Rows')
    plt.legend()
    # plt.show()
    plt.savefig("Outputs/Window_variation.pdf")
    plt.close()
    gc.collect()