import numpy as np

orig_data=np.load('data/hsi.npy')
mask=np.load('data/hsi_gt.npy')
a=orig_data[mask!=0]
a=a/np.max(a,axis=1,keepdims=True)
np.save('data/hsi_data_points.npy',a)
