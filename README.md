# Overview
This repository contains implementations of convolution operations in two different data formats: NCHW (Batch size, Channels, Height, Width) and NHWC (Batch size, Height, Width, Channels). The code is primarily written in C++, with Python scripts provided for output verification.
## Requirements
1. C++ compiler 
2. Python 
3. cnpy library to load .npy files
4. Required Python libraries (numpy, torch, torchvision)
## Cloning the repo
Use the command below to clone the repo
```
    https://github.com/TirumalaReddyVenkataChetan/Convolution_Assignment.git
```
## How to Run Convolution with NCHW data format

```
    source run_conv_nchw.sh
```
1. Input image shape [1x3x224x224], Weights shape [64x3x7x7]
2. expected output 
```
Output shape: [1, 64, 220, 220]
First output value: 0.542983
Python computation
        3D Convolution complete. Output written to 'py_conv_nchw_output.bin'.
        3D Convolution complete. Output written to 'py_conv_nhwc_output.bin'.
Min difference:  0.0
Max difference:  5.722046e-06
Mean difference:  2.3915592e-07
Files are identical
```

## How to Run Convolution with NHWC data format

```
    source run_conv_nhwc.sh
```
1. Input image shape [1x224x224x3], Weights shape [7x7x3x64]
2. expected output 
```
Output shape: [1, 220, 220, 64]
First output value: 0.542983
Python computation
        3D Convolution complete. Output written to 'py_conv_nchw_output.bin'.
        3D Convolution complete. Output written to 'py_conv_nhwc_output.bin'.
Min difference:  0.0
Max difference:  1.9073486e-06
Mean difference:  1.2412875e-07
Files are identical
```
