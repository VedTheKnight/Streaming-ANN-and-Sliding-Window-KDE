import numpy as np
from tqdm import tqdm
import time

def gauss_kernel(x, y, h):
    """Compute the Gaussian kernel between two data points."""
    diff = x - y
    return np.exp(-np.dot(diff, diff) / (2 * h * h))

def add_batches(batches,queries,dst,s:int,w:int,c):
    """
    Args:
    batches: stores the data for each batch
    queries: stores the queries for each batch
    dst: list of (mean,std) tuples representing true density of each batch
    s: cutoff value
    w: hard cap on window size
    c: smoothness parameter 
    """

    curr_batches={}
    R=0.2821   # For a gaussian kernel with \mu=0,\sigma=1 this is \frac{1}{\sqrt{4\pi}} \approx 0.2821
    
    for t,batch in enumerate(batches):

        print(f"In batch {t}")
        curr_batches[t]=np.array(batch)
        if t<w:    
            continue
        try:
            del curr_batches[t-w]
        except KeyError:
            print('Key not found in dictionary')

        # window generator
        m=int(1+3.322*np.log(np.min([len(i) for i in curr_batches.values()])))
        print(f'No of bins = {m}')
        dist=0
        batch_window={}
        count=0
        R_hat=[]
        hist,_=np.histogram(np.array(batch),bins=m,density=True)
        while count<w:
            hist_t,_=np.histogram(np.array(curr_batches[t-count]),bins=m,density=True)
            tmp=np.sum((hist-hist_t)**2)
            dist+=tmp
            if dist>s:
                break
            R_hat.append(m*tmp)
            batch_window[count]=curr_batches[t-count]
            count+=1
        print(f"Generated window of size {count}")

        # bandwidth generator
        h=np.zeros(count)
        S=np.copy(h)
        for i in range(count):
            h[i]=c*np.std(batch_window[i])/((2*count-1)*len(batch_window[i]))**(1/5)
            S[i]=5*R/(4*len(batch_window[i])*h[i])+(2*count-1)*R_hat[i]
        print("Generated bandwidths")
        # weight generator
        weights=np.zeros(count)
        a=1/S
        weights=a/np.sum(a)

        # address query 
        mean_kde=0 # average approximate kde value over the queries for batch t
        true_kde=0 
        mean_error=0
        st_time=time.time() # start time
        for j in queries[t]:
            tmp=np.exp(-(j-dst[t][0])**2 / (2 * dst[t][1]**2)) # computes the true kde for query j in batch t
            kde_res=0 # store the kde approx for a single query
            for i in range(count):
                p=0
                for x in batch_window[i]:
                    p+=gauss_kernel(x,j,h[i])
                p=p/len(batch_window[i])
                kde_res+=weights[i]*p # computing the approximate kde in sliding window setup
            mean_kde+=kde_res
            mean_error+=abs(tmp-kde_res)/tmp # computes the absolute relative error

        end_time=time.time() # end time
        print(f'Time taken for servicing {len(queries[t])} query points is {(end_time-st_time):.4f} seconds')
        mean_kde/=len(queries[t]) # computes the mean KDE for the batch t
        mean_error/=len(queries[t]) # computes the mean absolute relative error
        print(f'Approx Mean KDE for batch {t} : {mean_kde:.6f}, Mean relative error : {mean_error :.6f}\n')            


