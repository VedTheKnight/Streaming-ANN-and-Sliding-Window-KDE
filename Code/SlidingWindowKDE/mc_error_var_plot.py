import numpy as np
import matplotlib.pyplot as plt


num=50
x_cum=np.zeros((50,5),dtype=np.float32)

for itr in range(num):
    z=np.load(f'Synthetic_data_outputs/eps_v_err_val_{itr+1}.npy')
    x_cum[itr]=z

x_mean=np.mean(x_cum,axis=0)
eps_values = [0.05,0.1,0.2,0.5,1]

plt.figure(figsize=(10,6),dpi=100)
plt.plot(eps_values,x_mean,marker='*',mec='black',linestyle='-',color='blue',lw=1.75, label='AKDE with 200 rows')
plt.xlabel('Relative error of EH')
plt.ylabel('Log mean relative error')
plt.legend()
plt.title('Mean relative error vs EH error')
plt.grid(True)
f_n="Synthetic_data_outputs/MC_eps_v_err.pdf"
plt.savefig(f_n)
plt.close()
print("Plotting completed")
