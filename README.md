# Streaming ANN and AKDE
This repository contains codes of the proposed `ANN` and `AKDE` algorithms. The code is tested in Linux platform (python 3.11).

## Run the codes
The codes for AKDE are present in `Code/SlidingWindowKDE` whereas that of ANN are in `Code/StreamingANN`.

## AKDE
Change the current directory to `/Code/SlidingWindowKDE/`.

First of all you need to generate the daatsets. 

* **Synthetic data:** To generate synthetic data, run
```python data_generate.py
```
The generated data will be saved as *data_1.npy,data_2.npy,...,dat_50.npy* in the directory `/synthetic_data`. We have generated 50 datasets of size 10000 each consisting of 200-dimensional vectors. We will use these data for monte carlo simulations to demonstrate the performance of our algorithm.
* **Real world data:**
  * For the `News headlines` data, we have generated the encodings as 384-dimensional vectors and saved them as `.npy` files in `/data`.
  * For the `ROSIS Hyperspectral Images`, we have the image and binary mask saved as `data/hsi.npy` and `data/hsi_gt.npy` respectively. Run `hsi_data_gen.py` for preprocessing the data. The preprocessed HSI data will be stored as `data/hsi_data_points.npy`.


Now we will enumerate the files for data structures and the algorithm implementation.


**Utilities:** The necessary data structures for the `AKDE` algorithm are implemented in the files `angular_hash.py`, `p_stable.py`, `buckets_DS.py`, `Exponential_Histogram.py`. The **AKDE** algorithm is implemented in:
* `Ang_hash_AKDE.py`: using angular kernel.
* `L2_hash_AKDE.py`: using Euclidean p-stable(p=2) kernel.

### Variation of mean relative error with sketch size
* **Real-world dataset** Run the following command in terminal to compute the log of mean relative errors for different sketch sizes corresponding to number of rows=100,200,400,800,1600,3200. We take the window size as 450.
```
python3 sketch_size.py --file_name text --n 10000 --n_query 1000 --lsh 1 --b 1 --eps 0.1
```
Explanations for options:
 * `file_name`: specifies the type of real-world dataset, text for **News headlines** or image for **ROSIS HSI**.
 * `n`: specifies number of streaming data (taken as 10000).
 * `n_query`: number of queries (taken as 1000).
 * `lsh`: specifies the type of LSH kernel, 1 for Angular and 2 for Euclidean.
 * `w`: specifies the width of the euclidean kernel.
 * `r`: specifies the range of the euclidean hash.
 * `b`: specifies the bandwidth of the hash function (taken as 1).
 * `eps`: relative error of the exponential histogram (taken as 0.1)

* **Synthetic dataset** We use similar parameters like real-world dataset. Run the command
  ```
  python simulate_AH.py --lsh 1 --w 4 --r 1000 --b 1 --eps 0.1
  ```
  
### Effect of window size on the mean relative error
Run the following command to plot the log of mean relative errors versus number of rows for different values of window sizes (64,128,256,512,1024,2048). We have used L2 hash for the text data and Angular hash for the image data.
```
python window_size.py --file_name text --n 10000 --n_query 1000 --lsh 2 --w 4 --r 1000 --b 1 --eps 0.1
```
The options have the same explanation as before. The plot is saved as `Window_variation_text.pdf` in the `Outputs` directory.

### Comparison of AKDE with RACE
* **Real-world dataset** Run the following command to plot the performance of our algorithm **AKDE** with **RACE**. Note that AKDE works in the sliding window model(window size taken as 260) whereas RACE works in the general streaming setup. We have used *Angular LSH* for both the algorithms.
```
python compare2.py --data_type text --n 10000 --n_query 1000 --b 1 --eps 0.1
```
The option `data_type` specifies the type of dataset, text or image. The plot is saved as `mean_relative_error_vs_rows.pdf` in the `Outputs` directory.
* **Synthetic dataset**  Run the following command for synthetic dataset. Here also we have used Angular Hash and window size as 260.
```
python mc_compare.py --b 1 --eps 0.1
```
Here, the options have same explanation as before. The relative errors for RACE and AKDE are saved in *results_i.npy* for *i=1,2...,50* in `Synthetic_data_outputs_L2` directory. To plot the graph, run
```
python mc_plot_compare.py
```
The plot will be saved as `MC_compare.pdf` in `Synthetic_data_outputs` directory.

## Streaming ANN
