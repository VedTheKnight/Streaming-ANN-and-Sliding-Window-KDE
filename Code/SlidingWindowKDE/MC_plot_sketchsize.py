import numpy as np

import matplotlib.pyplot as plt

num=50
y_cum=np.zeros((50,6),dtype=np.float32)
x_cum=np.zeros((50,6),dtype=np.float32)

for itr in range(num):
    f_n=f'Synthetic_data_outputs/err_values_{itr+1}.npy'
    tmp=np.load(f_n)
    y_cum[itr,:]=tmp.reshape(1,-1)
    f_n=f'Synthetic_data_outputs/sketch_sizes_{itr+1}.npy'
    tmp=np.load(f_n)
    x_cum[itr,:]=tmp.reshape(1,-1)

x_mean1=np.mean(x_cum,axis=0)
y_mean1=np.mean(y_cum,axis=0)

plt.figure(figsize=(10,6),dpi=100)
plt.plot(x_mean1,y_mean1,marker='*',mec='black',linestyle='-',color='blue',lw=1.75, label='Angular Hash')
plt.xlabel('Sketch size in KB')
plt.ylabel('Log mean relative error')
plt.legend()
# plt.title('Mean relative error vs Sketch size')
plt.grid(True)
f_n="Synthetic_data_outputs/MC_AH.pdf"
plt.savefig(f_n)
plt.close()

y_cum=np.zeros((50,6),dtype=np.float32)
x_cum=np.zeros((50,6),dtype=np.float32)
for itr in range(num):
    f_n=f'Synthetic_data_outputs_L2/err_values_{itr+1}.npy'
    tmp=np.load(f_n)
    y_cum[itr,:]=tmp.reshape(1,-1)
    f_n=f'Synthetic_data_outputs_L2/sketch_sizes_{itr+1}.npy'
    tmp=np.load(f_n)
    x_cum[itr,:]=tmp.reshape(1,-1)

x_mean2=np.mean(x_cum,axis=0)
y_mean2=np.mean(y_cum,axis=0)

plt.figure(figsize=(10,6),dpi=100)
plt.plot(x_mean2,y_mean2,marker='*',mec='black',linestyle='-',color='blue',lw=1.75, label='L2 Hash')
plt.xlabel('Sketch size in KB')
# plt.xlabel('Number of rows in the sketch')
plt.ylabel('Log mean relative error')
plt.legend()
# plt.title('Mean relative error vs sketch size')
plt.grid(True)
f_n="Synthetic_data_outputs_L2/MC_L2.pdf"
plt.savefig(f_n)
plt.close()
print("Plotting completed")
