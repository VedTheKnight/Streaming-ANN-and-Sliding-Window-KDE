# Streaming ANN and AKDE
This repository contains codes of the proposed `ANN` and `AKDE` algorithms. The code is tested in Linux platform (python 3.11).

## Run the codes
The codes for AKDE are present in `Code/SlidingWindowKDE` whereas that of ANN are in `Code/StreamingANN`.

## AKDE
Change the current directory to `/Code/SlidingWindowKDE/`.

First of all you need to generate the daatsets. 

* **Synthetic data:** To generate synthetic data, run `data_generate.py`. The generated data will be saved in `/synthetic_data`.
* **Real world data:**
  * For the `News headlines` data, we have generated the encodings as 384-dimensional vectors and saved them as `.npy` files in `/data`.
  * For the `ROSIS Hyperspectral Images`, we have the image and binary mask saved as `data/hsi.npy` and `data/hsi_gt.npy` respectively. Run `hsi_data_gen.py` for preprocessing the data. The preprocessed HSI data will be stored as `data/hsi_data_points.npy`. 
