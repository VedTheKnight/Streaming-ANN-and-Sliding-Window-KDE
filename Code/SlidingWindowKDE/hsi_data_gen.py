import numpy as np

orig_data=np.load('data/hsi.npy')
mask=np.load('data/hsi_gt.npy')
ext_data=orig_data[mask!=0]
np.save('data/hsi_data_points.npy',ext_data)