import matplotlib.pyplot as plt


# Example usage:
# plot_from_textfile("obs_mean_error_vs_sketch_sz_text.txt")


def plot_from_textfile(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]  # remove empty lines

    # Read header information
    N = int(lines[0])                   # number of data points
    ylabel = lines[1]                   # y-axis label
    xlabel = lines[2]                   # x-axis label
    legend_label = lines[3]             # legend text
    title = lines[4]                    # plot title
    output_path = lines[5]              # output pdf path

    # Read x, y data
    x_vals = []
    y_vals = []
    for line in lines[6:6+N]:
        x, y = map(float, line.split(",") if "," in line else line.split())
        x_vals.append(x)
        y_vals.append(y)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, marker='+',mec='blue',linestyle='-',color='red',lw=1.75,label=legend_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__=="__main__":
    plot_from_textfile("obs_mean_error_vs_sketch_sz_text.txt")