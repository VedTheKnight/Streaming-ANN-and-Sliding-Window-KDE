import matplotlib.pyplot as plt
import os

# Example usage:
# plot_from_textfile("obs_mean_error_vs_sketch_sz_text.txt")


def plot_from_textfile(file_path,legend_label,col):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]  # remove empty lines

    # Read header information
    N = int(lines[0])                   # number of data points
    ylabel = lines[1]                   # y-axis label
    xlabel = lines[2]                   # x-axis label
    title=lines[4]

    # Read x, y data
    x_vals = []
    y_vals = []
    for line in lines[6:6+N]:
        x, y = map(float, line.split(",") if "," in line else line.split())
        x_vals.append(x)
        y_vals.append(y)

    # Plotting

    plt.plot(x_vals, y_vals, marker='*',mec='black',linestyle='-',color=col,lw=1.75,label=legend_label)
    return xlabel,ylabel,title

if __name__=="__main__":
    current_dir = os.getcwd()
    dir_name = os.path.join(current_dir, "Outputs")
    os.makedirs(dir_name, exist_ok=True)
    f_n=f"{dir_name}/error_vs_sz_L2.pdf"
    plt.figure(figsize=(8, 5))
    xlabel,ylabel,title=plot_from_textfile("error_vs_sz_text_L2.txt",'text','blue')
    _,_,_=plot_from_textfile("error_vs_sz_image_L2.txt",'image','green')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f_n)
    plt.close()
    print(f"Plot saved to {f_n}")