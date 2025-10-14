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
    parser.add_argument("--b",help="Bandwidth of the LSH kernel")
    parser.add_argument("--eps",help="Relative error of exponential histogram")
    args=parser.parse_args()
    #k=int(arg.b) the bandwidth parameter of hash function


    for data_idx in range(50):
        # Load data
        data=np.load(f'synthetic_data/data_{data_idx+1}.npy')
        print(f"\n----------------Data {data_idx} loaded successfully")
        data = data.astype(np.float32)
        dim = data.shape[1]
        print(f"Data shape: {data.shape}")
        k = int(args.b)
        num_data = data.shape[0]-2000
        n_query = 1000
        eps = float(args.eps)
        N = 260
        available_indices = list(range(num_data,data.shape[0]))
        if n_query > len(available_indices):
            raise ValueError("n_query larger than available data points")
        query_idx = random.sample(available_indices, n_query)
        query = data[query_idx]
        del available_indices

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
        np.savez(f'Synthetic_data_outputs/results_{data_idx+1}.npz', race=np.array(err1), akde=np.array(err2))
    print("computation done")
    

    """
    print(" In plot section ")
    current_dir = os.getcwd()
    tar_dir=os.path.join(current_dir,"Synthetic_data_outputs_L2")
    os.makedirs(tar_dir,exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.plot(n_row, err1, marker='*',mec='black',linestyle='-',color='blue',lw=1.75, label='Original RACE')
    plt.plot(n_row, err2, marker='*',mec='black',linestyle='-',color='green',lw=1.75,label='Sliding window AKDE')
    plt.xlabel('Number of Rows in RACE Sketch')
    plt.ylabel('Log(Mean Relative Error)')
    plt.title('Mean Relative Error vs Number of Rows')
    plt.legend()
    plt.grid(True)
    # plt.show()
    f_n=tar_dir+"/MC_compare.pdf"
    plt.savefig(f_n)
    plt.close()
    """
    gc.collect()
