import pickle
from takde_alg import add_batches


if __name__=="__main__":

    data=[]
    queries=[]
    densities=[]
    with open('Streaming-ANN-and-Sliding-Window-KDE-1/Code/SlidingWindowKDE/TAKDE/batch_data.pkl','rb') as file:
        data=pickle.load(file)

    with open('Streaming-ANN-and-Sliding-Window-KDE-1/Code/SlidingWindowKDE/TAKDE/batch_queries.pkl','rb') as file:
        queries=pickle.load(file)

    with open('Streaming-ANN-and-Sliding-Window-KDE-1/Code/SlidingWindowKDE/TAKDE/batch_densities.pkl','rb') as file:
        densities=pickle.load(file)

    print("Data loaded successfully")
    add_batches(data,queries,densities,1,16,0.15)