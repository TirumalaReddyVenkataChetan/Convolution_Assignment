#!/bin/bash
cd conv_with_pad_stride
# Compile the C++ code
g++ conv_nhwc.cpp ../utils/utils.cpp -o conv_nhwc -lcnpy

# Execute the compiled program
./conv_nhwc

cd ../python
# Run the Python script for convolution
python3 conv3d.py

# Compare the output binary files
python3 validate.py ../outputs/conv_nhwc_cpp.bin ../outputs/py_conv_nhwc_output.bin
cd ..
