import matplotlib.pyplot as plt
import os

# Example usage:
# plot_from_textfile("obs_mean_error_vs_sketch_sz_text.txt")


def plot_from_textfile(file_path,):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]  # remove empty lines

    # N = int(lines[0])                   # number of data points
    # ylabel = lines[1]                   # y-axis label
    # xlabel = lines[2]                   # x-axis label
    # title=lines[4]

    # Read x, y data
    # x_vals = []
    # y_vals = []
    # for line in lines[6:6+N]:
    #     x, y = map(float, line.split(",") if "," in line else line.split())
    #     x_vals.append(x)
    #     y_vals.append(y)
    color_list=['b','g','r','c','m','y']

    # Read x, y data
    for i in range(6):
        lb=int(lines[7*i].split(' ')[-1])
        x_vals = []
        y_vals = []
        for line in lines[7*i+1:7*i+7]:
            x, y = map(float, line.split(",") if "," in line else line.split())
            x_vals.append(x)
            y_vals.append(y)

        # Plotting
        plt.plot(x_vals, y_vals, marker='*',mec='black',linestyle='-',color=color_list[i],lw=1.75,label=lb)

if __name__=="__main__":
    current_dir = os.getcwd()
    dir_name = os.path.join(current_dir, "Outputs","text")
    os.makedirs(dir_name, exist_ok=True)
    f_n=f"{dir_name}/window_variation_text.png"
    plt.figure(figsize=(10,6))
    plot_from_textfile('Window_data_text_L2 Hash.txt')
    # plot_from_textfile("error_vs_sz_text_L2.txt",'blue','text')
    # plot_from_textfile("error_vs_sz_image_L2.txt",'green','image')
    plt.xlabel('Sketch size in KB')
    plt.ylabel('Log(Mean Relative Error)')
    plt.title('Mean Relative Error vs Number of rows L2 hash')
    plt.legend()
    plt.grid(True)
    plt.savefig(f_n)
    plt.close()
    print(f"Plot saved to {f_n}")