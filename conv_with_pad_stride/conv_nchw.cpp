#include <iostream>
#include <vector>
#include "../utils/utils.hpp"

using namespace std;
typedef vector<vector<vector<vector<float>>>> Matrix;

// Defining Convolution function
void convnchw(const Matrix &input,const Matrix &kernel, Matrix &output,const int &stride,const int &padding){
    // Intializing Dimensions
    int batch = input.size();
    int in_channels = input[0].size();
    int in_height = input[0][0].size();
    int in_width = input[0][0][0].size();
    int no_kernels = kernel.size();
    int k_channels = kernel[0].size();
    int k_height = kernel[0][0].size();
    int k_width = kernel[0][0][0].size();

    // Intializing Padded_input
    Matrix padded_input(batch, 
                        vector<vector<vector<float>>>(in_channels,
                        vector<vector<float>>(in_height+2*padding,
                        vector<float>(in_width+2*padding,0))));
    // Adding Padding to the input  
    for(int b=0;b<batch;b++){
        for(int c=0;c<in_channels;c++){
            for(int h=0;h<in_height;h++){
                for(int w=0;w<in_width;w++){
                    padded_input[b][c][h+padding][w+padding]=input[b][c][h][w];
                }
            }
        }
    }

    int ou_height = (in_height+2*padding-k_height)/stride + 1;
    int ou_width = (in_width+2*padding-k_width)/stride + 1;
    int ou_batches = input.size();
    int ou_channels = kernel.size();
    // Intializing Output
    output.assign(ou_batches,
                  vector<vector<vector<float>>>(ou_channels,
                  vector<vector<float>>(ou_height,
                  vector<float>(ou_width,0))));

    // Performing the convolution
    for (int b=0;b<batch;b++){
        for (int o_c=0;o_c<ou_channels;o_c++){
            for (int i_h=0;i_h<ou_height;i_h++){
                for (int i_w=0;i_w<ou_width;i_w++){
                    float sum = 0;
                    for (int i_c=0;i_c<in_channels;i_c++){
                        for (int k_h=0;k_h<k_height;k_h++){
                            for (int k_w=0;k_w<k_width;k_w++){
                                sum+=padded_input[b][i_c][i_h*stride+k_h][i_w*stride+k_w]*kernel[o_c][i_c][k_h][k_w];
                            }
                        }
                    }
                    output[b][o_c][i_h][i_w]=sum;
                }
            }
        }
    }
    // Writing the output to 1D vector
    vector<float> output_1d;
    for (int b = 0; b < batch; b++){
        for (int i = 0; i < ou_channels; i++){
            for (int j = 0; j < ou_height; j++){
                for (int k = 0; k < ou_width; k++){
                    output_1d.push_back(output[b][i][j][k]);
                }
            }
        }
    }
    write_to_binary("../outputs/conv_nchw_cpp.bin", output_1d);
}
int main() {

    vector<int> input_dims = {1,3,224,224};
    vector<int> kernel_dims = {64,3,7,7};
    auto input = read_npy_file("../inputs/py_input.npy", input_dims);
    auto kernel = read_npy_file("../weights/conv1_wt.npy", kernel_dims);
    // Output tensor
    vector<vector<vector<vector<float>>>> output;

    // Perform the convolution
    convnchw(input, kernel, output, 1, 1);

    //Print the output dimensions and the first value for verification
    cout << "Output shape: [" << output.size() << ", "
         << output[0].size() << ", "
         << output[0][0].size() << ", "
         << output[0][0][0].size() << "]" << endl;
    cout << "First output value: " << output[0][0][0][0] << endl;

    return 0;
}