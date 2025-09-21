import torch
import numpy as np
import random, math, time, gc,os
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from Exponential_Histogram import ExpHst
from p_stable import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RACE_L2:
    def __init__(self, rows:int, hash_range:int, k:int, dim:int, N:int, width:int, eps=0.5):
        self.L = rows
        self.R = hash_range
        self.k = k
        self.winsize = N
        self.err = eps
        self.sparse_dic = {}
        self.hasher = UniversalHasher(self.R, seed=42)
        self.hash_list = []
        for _ in range(self.L):
            h = []
            for _ in range(k):
                h.append(PStableLSH(dim, width, 2))  # leave logic unchanged
            self.hash_list.append(h)

    def update_counter(self, data, t):
        for k, i in enumerate(self.hash_list):
            r = []
            for j in i:
                r.append(self.hasher.hash(j.hash(data)))
                s = sum(r)
            if (k, s) not in self.sparse_dic:
                self.sparse_dic[(k, s)] = ExpHst(self.winsize, math.ceil(1/self.err))
            else:
                self.sparse_dic[(k, s)].new_bucket(t)

    def query1(self, data):
        val = 0
        for k, i in enumerate(self.hash_list):
            r = []
            for j in i:
                r.append(self.hasher.hash(j.hash(data)))
                s = sum(r)
            if (k, s) in self.sparse_dic:
                val += self.sparse_dic[(k, s)].count_est()
        return val / self.L


def l2_lsh_collision_probability(d, w):
    if d == 0:
        return 1.0
    term1 = 1 - 2 * norm.cdf(-w / d)
    term2 = (2 * d / (np.sqrt(2 * np.pi) * w)) * (1 - np.exp(-(w**2) / (2 * d**2)))
    return term1 - term2