import numpy as np

class Angular_Hash:
    def __init__(self,dim):
        self.dim=dim # dimension of the input to the hash function
        self.w=np.random.normal(0,1,(dim,1))

    def eval(self,x): # evaluate h(x)
        z=1 if np.dot(np.squeeze(self.w),x)>=0 else -1
        return z
    
