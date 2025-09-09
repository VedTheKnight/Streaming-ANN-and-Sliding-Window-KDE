import numpy as np
from RACE_DS import *
import random

data=np.load('encodings.npy')
# data=np.random.normal(25,10,(10000,500)) # for random data

dim=data.shape[1]


eps=0.25 # relative error for Exponential Histogram
k=1 # the bandwidth parameter of hash function
N=20 # window size
num_data=5000 # number of streaming data
eps_=2*eps+eps*eps # relative error of A-KDE
n_query=500 # number of queries

print('---------Parameters----------')
print(f'Bandwidth parameter={k}, Window size={N}, Relative error of EH={eps}, Data dimension {dim}')
print(f'Relative error of A-KDE ={eps_}, Number of streaming data={num_data}, Number of queries={n_query}')
print('----------------------------')
query=data[random.sample(range(6000,10000),n_query)] # n_query random queries from the data points

# query=np.random.randn(dim) # random query
n_row=1 # number of rows initialized to 1

true_kde=np.zeros(n_query) # true kde for the last N data points
for j in range(n_query):
    for i in range(num_data-N,num_data):
        true_kde[j]+=math.pow(1-1/np.pi*angle_between_vectors(data[i,:],query[j]),k) # calculating the collision probability
# print(f'True KDE={true_kde:.6f}')
true_kde=true_kde/N # true kde
print(f'Mean True KDE={np.mean(true_kde):.6f}')
for i in range(1,100):
    r_sketch=RACE(n_row,2,k,dim,N,eps)# creating an instance of a RACE sketch
    app_kde=np.zeros(n_query) # approximate kde
    
    for j in range(num_data):
        r_sketch.update_counter(data[j,:],j+1) # updating the sketch with the streaming data   
    for j in range(n_query):
        app_kde[j]=r_sketch.query(query[j]) # calculate the approximate kde from the race sketch
    rel_err=np.mean(abs((app_kde-true_kde)/true_kde)) # calculate relative error
    # print(f'R={n_row} A-KDE={app_kde:.6f} Relative error={rel_err:.6f}')
    print(f'R={n_row} Mean A-KDE={np.mean(app_kde):.6f} Mean Relative error={rel_err:.6f}')
    del r_sketch # delete the sketch to free memory
    if rel_err<= eps_:
        print('Relative error below threshold')
        break

    n_row=2*n_row # doubling the number of rows
    # break # for debugging purpose only one iteration is done
