import numpy as np
import pickle

num_batch=100 # number of batches
num_test=500 # number of test points in each batch
np.random.seed(236)
batch_sizes=a=np.random.randint(5,21,size=num_batch)
# print(batch_sizes)
np.random.seed(438)
dis_mean=np.random.choice(int(num_batch/10),num_batch,replace=True) # mean of a probability distribution
batch_data=[]
batch_queries=[]
batch_densities=[]
for i in range(num_batch):
    batch_data.append(np.random.normal(loc=dis_mean[i],scale=5,size=batch_sizes[i])) # generating batch data from a normal distribution
    batch_queries.append(np.random.normal(loc=dis_mean[i],scale=5,size=num_test)) # generating query data from a normal distribution
    batch_densities.append([dis_mean[i],5])

with open('Streaming-ANN-and-Sliding-Window-KDE-1/Code/SlidingWindowKDE/TAKDE/batch_data.pkl','wb') as file:
    pickle.dump(batch_data,file)

with open('Streaming-ANN-and-Sliding-Window-KDE-1/Code/SlidingWindowKDE/TAKDE/batch_queries.pkl','wb') as file:
    pickle.dump(batch_queries,file)

with open('Streaming-ANN-and-Sliding-Window-KDE-1/Code/SlidingWindowKDE/TAKDE/batch_densities.pkl','wb') as file:
    pickle.dump(batch_densities,file)
