from RACE_19 import RACE_1
from Ang_hash_AKDE import RACE_Ah
import numpy as np
import time,random,gc,math,sys,os,argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type",choices=['text','image'],help="type of file : text or image encoding")
    parser.add_argument("--n",help="Number of streaming data")
    parser.add_argument("--n_query",help="Number of queries")
    parser.add_argument("--b",help="Bandwidth of the LSH kernel")
    parser.add_argument("--eps",help="Relative error of exponential histogram")
    arg=parser.parse_args()
    if arg.data_type=='text':
        files = ["data/encodings.npy", "data/encodings_2.npy", "data/encodings_3.npy", "data/encodings_4.npy"]
        arrays = [np.load(f) for f in files]
        data = np.vstack(arrays)
        dim=data.shape[1]
        print(f"Data shape: {data.shape}")
    
    elif arg.data_type=='image':
        data=np.load('data/hsi_data_points.npy')
        dim=data.shape[1]
        print(f"Data shape: {data.shape}")



    k=int(arg.b) # the bandwidth parameter of hash function
    num_data=int(arg.n) # number of streaming data
    n_query=int(arg.n_query) # number of queries
    print('---------Parameters for original RACE----------')
    print(f'Bandwidth parameter={k},  Data dimension {dim}')
    print(f'Number of streaming data={num_data}, Number of queries={n_query}')
    print('----------------------------')

    # parameters for our RACE in sliding window
    eps=float(arg.eps) # relative error for Exponential Histogram
    N=260 # window size
    eps_=2*eps+eps*eps # relative error of A-KDE


    print('---------Parameters for Sliding window RACE----------')
    print(f'Bandwidth parameter={k}, Window size={N}, Relative error of EH={eps}, Data dimension {dim}')
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
        for i in range(num_data-N,num_data):
            true_kde_2[j]+=math.pow(1-1/np.pi*angle_between_vectors(data[i,:],query[j]),k) # calculating the collision probability
    # print(f'True KDE={true_kde:.6f}')
    end_time=time.time()
    print(f'Time taken to compute true KDE for {n_query} queries is {end_time-st_time:.2f} seconds')
    print(f'Mean True KDE={np.mean(true_kde_2):.6f}')

# number of rows in RACE structure to be used for the experiments
    n_row=[100,200,400,800,1600,3200]
    
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
        st_time=time.time()
        for j in tqdm(range(n_query),desc='Answering queries'):
            app_kde[j]=r_sketch.query(query[j]) # calculate the approximate kde from the race sketch
        end_time=time.time()
        rel_err=np.mean(abs((app_kde-true_kde_1)/true_kde_1)) # calculate relative error
        err1.append(np.log(rel_err))
        print(f'Mean A-KDE={np.mean(app_kde):.6f} Mean Relative error={rel_err:.6f}')
        del r_sketch # delete the sketch to free memory
 
        r_sketch=RACE_Ah(n_row[i],2,k,dim,N,eps)# creating an instance of a SW RACE sketch
        
        st_time=time.time()
        for j in tqdm(range(num_data), desc="Adding data to (SW) RACE sketch"):
            r_sketch.update_counter(data[j,:],j+1) # updating the sketch with the streaming data  
        end_time=time.time()
        st_time=time.time()
        for j in range(n_query):
            app_kde[j]=r_sketch.query1(query[j]) # calculate the approximate kde from the race sketch
        end_time=time.time()
        rel_err=np.mean(abs((app_kde-true_kde_2)/true_kde_2)) # calculate relative error
        print(f'Mean A-KDE={np.mean(app_kde):.6f} Mean Relative error={rel_err:.6f}\n')
        err2.append(np.log(rel_err))
        del r_sketch # delete the sketch to free memory
    
# plotting the graphs
# create a directory to store the graph
    print(" In plot section ")
    current_dir = os.getcwd()
    tar_dir=os.path.join(current_dir,"Outputs",arg.data_type)
    os.makedirs(tar_dir,exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.plot(n_row, err1, marker='*',mec='black',linestyle='-',color='blue',lw=1.75, label='Original RACE')
    plt.plot(n_row, err2, marker='*',mec='black',linestyle='-',color='green',lw=1.75,label='Sliding Window RACE')
    plt.xlabel('Number of Rows in RACE Sketch')
    plt.ylabel('Log(Mean Relative Error)')
    plt.title('Mean Relative Error vs Number of Rows')
    plt.legend()
    # plt.show()
    f_n=tar_dir+"/mean_relative_error_vs_rows.pdf"
    plt.savefig(f_n)
    plt.close()

    gc.collect()