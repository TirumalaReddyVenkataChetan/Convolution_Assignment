#!/bin/bash
cd conv_with_pad_stride
# Compile the C++ code
g++ conv_nchw.cpp ../utils/utils.cpp -o conv3d_nchw -lcnpy

# Execute the compiled program
./conv3d_nchw

cd ../python
# Run the Python script for convolution
python3 conv3d.py

# Compare the output binary files
python3 validate.py ../outputs/conv_nchw_cpp.bin ../outputs/py_conv_nchw_output.bin
cd ..
