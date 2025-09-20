from Exponential_Histogram import ExpHst
from angular_hash import Angular_Hash
import math
import numpy as np
class RACE_Ah:
    def __init__(self,rows:int,hash_range:int,k:int,dim:int,N:int,eps=0.5):
        self.L=rows # number of ACE repetitions
        self.R=hash_range # range of the Hash function
        self.k=k # number of concatenations of the hash function i.e. number of columns is R^k
        self.winsize=N # window size
        self.err=eps # relative error of Exponential Histogram
        self.sparse_dic={}
        self.hash_list=[] # storing the list of L hash functions
        for _ in range(self.L):
            h=[] # hash function concatenated k times
            for _ in range(k):
                h.append(Angular_Hash(dim)) # dim is the dimension of input data
            self.hash_list.append(h)

    def update_counter(self,data,t): # updates the RACE structure when a data comes at timestamp t
        for k,i in enumerate(self.hash_list):
            r=0
            for j in i:
                r=(1 if j.eval(data)==1 else 0)*2+r
            if (k,r) not in self.sparse_dic:
                self.sparse_dic[(k,r)]=ExpHst(self.winsize,math.ceil(1/self.err)) # create a new Exponential histogram if not present)
            else:
                self.sparse_dic[(k,r)].new_bucket(t) # add a new element to the Exponential histogram at timestamp t
    
    def query1(self,data): # return the KDE for the query data as the mean of the RACE cells
        val=0
        for k,i in enumerate(self.hash_list):
            r=0
            for j in i:
                r=(1 if j.eval(data)==1 else 0)*2+r
            if (k,r) in self.sparse_dic:
                val+=self.sparse_dic[(k,r)].count_est()

        return val/self.L # return the mean
    
    def query2(self,data): # return the KDE for the query data as the median of means of RACE cells
        N=self.L
        m=5 # chunk size
        arr=np.zeros(N)
        for k,i in enumerate(self.hash_list):
            r=0
            for j in i:
                r=(1 if j.eval(data)==1 else 0)*2+r
            if (k,r) in self.sparse_dic:
                arr[k]=self.sparse_dic[(k,r)].count_est()
        num_chunks= N//m
        chunks=[arr[i*m:(i+1)*m] for i in range(num_chunks)]
        if N % m !=0:
            chunks.append(arr[num_chunks*m:N])
        means=[np.mean(i) for i in chunks]
        return np.median(means)






            



