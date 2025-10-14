import numpy as np
import matplotlib.pyplot as plt


num=50
x_cum1=np.zeros((50,6),dtype=np.float32)
x_cum2=np.zeros((50,6),dtype=np.float32)

for itr in range(num):
    z=np.load(f'Synthetic_data_outputs/results_{itr+1}.npz')
    x_cum1[itr]=z['race']
    x_cum2[itr]=z['akde']

x_mean1=np.mean(x_cum1,axis=0)
x_mean2=np.mean(x_cum2,axis=0)
n_row=[100,200,400,800,1600,3200]

plt.figure(figsize=(10,6),dpi=100)
plt.plot(n_row,x_mean1,marker='*',mec='black',linestyle='-',color='blue',lw=1.75, label='RACE')
plt.plot(n_row,x_mean2,marker='*',mec='black',linestyle='-',color='green',lw=1.75, label='AKDE')
plt.xlabel('Number of rows')
plt.ylabel('Log mean relative error')
plt.legend()
plt.title('Mean relative error vs Number of rows')
plt.grid(True)
f_n="Synthetic_data_outputs/MC_compare.pdf"
plt.savefig(f_n)
plt.close()
print("Plotting completed")
