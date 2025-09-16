from angular_hash import Angular_Hash
import numpy as np
import time
import random
from tqdm import tqdm
import math
import gc

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


class RACE:
    def __init__(self,rows:int,hash_range:int,k:int,dim:int):
        self.L=rows # number of ACE repetitions
        self.R=hash_range # range of the Hash function
        self.k=k # number of concatenations of the hash function i.e. number of columns is R^k
        self.sparse_dic={}
        self.hash_list=[] # storing the list of L hash functions
        for _ in range(self.L):
            h=[] # hash function concatenated k times
            for _ in range(k):
                h.append(Angular_Hash(dim)) # dim is the dimension of input data
            self.hash_list.append(h)

    def update_counter(self,data): # updates the RACE structure when a data comes at timestamp t
        for k,i in enumerate(self.hash_list):
            r=0
            for j in i:
                r=(1 if j.eval(data)==1 else 0)*2+r
            if (k,r) not in self.sparse_dic:
                self.sparse_dic[(k,r)]=1 # create a new dictionary key for the sparse representation of RACE sketch
            else:
                self.sparse_dic[(k,r)]+=1 # increment existing cell count
    
    # def query(self,data): # return the KDE for the query data
    #     val=0
    #     for k,i in enumerate(self.hash_list):
    #         r=0
    #         for j in i:
    #             r=(1 if j.eval(data)==1 else 0)*2+r
    #         if (k,r) in self.sparse_dic:
    #             val+=self.sparse_dic[(k,r)]

    #     return val/self.L # return the mean
    
    def query(self,data): # return the KDE for the query data
        N=self.L
        m=5 # chunk size
        arr=np.zeros(N)
        for k,i in enumerate(self.hash_list):
            r=0
            for j in i:
                r=(1 if j.eval(data)==1 else 0)*2+r
            if (k,r) in self.sparse_dic:
                arr[k]=self.sparse_dic[(k,r)]
        num_chunks= N//m
        chunks=[arr[i*m:(i+1)*m] for i in range(num_chunks)]
        if N % m !=0:
            chunks.append(arr[num_chunks*m:N])
        means=[np.mean(i) for i in chunks]
        return np.median(means)
    
if __name__=="__main__":
    files = ["data/encodings.npy", "data/encodings_2.npy", "data/encodings_3.npy", "data/encodings_4.npy"]
    arrays = [np.load(f) for f in files]
    data = np.vstack(arrays)
    dim=data.shape[1]
    print(f"Data shape: {data.shape}")


    k=1 # the bandwidth parameter of hash function
    num_data=10000 # number of streaming data
    n_query=1000 # number of queries
    print('---------Parameters----------')
    print(f'Bandwidth parameter={k},  Data dimension {dim}')
    print(f'Number of streaming data={num_data}, Number of queries={n_query}')
    print('----------------------------')
    random.seed(124)
    query=data[random.sample(range(71000,80000),n_query)] # n_query random queries from the data points

    # query=np.random.randn(dim) # random query
    n_row=2560 # number of rows initialized to 1

    true_kde=np.zeros(n_query) # true kde for the n_query query points
    st_time=time.time()
    for j in tqdm(range(n_query), desc="Traversing the data for true KDE"):
        for i in range(num_data):
            true_kde[j]+=math.pow(1-1/np.pi*angle_between_vectors(data[i,:],query[j]),k) # calculating the collision probability

    end_time=time.time()
    print(f'Time taken to compute true KDE for {n_query} queries is {end_time-st_time:.2f} seconds')
    print(f'Mean True KDE={np.mean(true_kde):.6f}')
    for i in range(1,100):
        r_sketch=RACE(n_row,2,k,dim)# creating an instance of a RACE sketch
        app_kde=np.zeros(n_query) # approximate kde
        
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
        rel_err=np.mean(abs((app_kde-true_kde)/true_kde)) # calculate relative error
        print(f'R={n_row} Mean A-KDE={np.mean(app_kde):.6f} Mean Relative error={rel_err:.6f}\n')
        del r_sketch # delete the sketch to free memory

        n_row=2*n_row # doubling the number of rows
        break
    gc.collect()