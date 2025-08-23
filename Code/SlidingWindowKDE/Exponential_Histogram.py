from buckets_DS import Bucket 
import math

class ExpHst: # define the class for Exponential Histogram
    def __init__(self,N,k):
        self.bucket_list=[] # maintains the list of active buckets
        self.window_size=N # initializes the sliding window size
        self.k=k # stores the parameter k for the exponential histogram
        self.last=0 # stores size of last bucket
        self.total=0 # stores total size of all buckets 

    def del_last(self): # delete the last bucket
        # print('Delete last bucket')
        # store the size of last bucket in a
        a=self.bucket_list[0].get_size()
        del self.bucket_list[0] # delete the last bucket from the list
        # update the counters total and last
        self.last=self.bucket_list[0].get_size()
        self.total-=a
    
    def new_bucket(self,t): # new element arriving at time t
        # check if the time stamp of last bucket has expired
        if len(self.bucket_list)!=0:
            if self.bucket_list[0].get_timestamp()<(t-self.window_size):
                self.del_last() # delete last bucket

        self.bucket_list.append(Bucket(t))
        self.total+=1
        cap=math.ceil(self.k/2)+1
        i=len(self.bucket_list)-1
        while i>0:
            j=i-1
            rep=1
            s=self.bucket_list[i].get_size()
            while j>=0 and self.bucket_list[j].get_size()==s:
                rep+=1
                j-=1
            if rep>cap: # check if number of similar sized buckets exceed k/2+1
                del self.bucket_list[j+1] # deleting the oldest similar bucket
                self.bucket_list[j+1].set_size(2*s)
                if j+1==0:
                    self.last=2*s
                i=j+1
            else:
                break
        # print("New bucket added successfully")

    def print_buckets(self):
        print('Bucket list :',end='')
        for i in range(0,len(self.bucket_list)):
            print(self.bucket_list[i].get_size(),end=',')
    
    def count_est(self):
        return self.total-1/2.0*self.last

