from Exponential_Histogram import ExpHst
from angular_hash import Angular_Hash
import math

class RACE:
    def __init__(self,rows,range,k,dim):
        self.L=rows # number of ACE repetitions
        self.R=range # range of the Hash function
        self.k=k # number of concatenations of the hash function i.e. number of columns is R^k
        self.sparse_dic={}
        self.hash_list=[] # storing the list of L hash functions
        for _ in range(self.L):
            h=[] # hash function concatenated k times
            for _ in range(k):
                h.append(Angular_Hash(dim)) # dim is the dimension of input data
            self.hash_list.append(h)

    def update_counter(self,data,t): # updates the RACE structure when a data comes at timestamp t
        for i in self.hash_list:
            r=0
            for j in i:
                r=(1 if j.eval(data)==1 else 0)*2+r
            if (i,r) not in self.sparse_dic:
                self.sparse_dic[(i,r)]=ExpHst(10,2)
            else:
                self.sparse_dic[(i,r)].new_bucket(t) # add a new element to the Exponential histogram at timestamp t
    
    def query(self,data): # return the KDE for the query data
        val=0
        for k in self.hash_list:
            r=0
            for j in k:
                r=(1 if j.eval(data)==1 else 0)*2+r
            if (k,r) in self.sparse_dic:
                val+=self.sparse_dic[(k,r)].count_est()

        return val/self.L # return the mean






            



