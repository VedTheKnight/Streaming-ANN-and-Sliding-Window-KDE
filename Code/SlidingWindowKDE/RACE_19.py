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


class RACE_1:
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
    
    def query(self,data): # return the KDE for the query data as the median of means
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
