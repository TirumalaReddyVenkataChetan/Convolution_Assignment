#include <iostream>
#include <vector>
#include "../utils/utils.hpp"

using namespace std;
typedef vector<vector<vector<vector<float>>>> Matrix;

// Defining Convolution function
void convnhwc(const Matrix &input,const Matrix &kernel, Matrix &output,const int &stride, const int &padding){
    
    // Intializing Dimensions
    int batch = input.size();
    int in_height = input[0].size();
    int in_width = input[0][0].size();
    int in_channels = input[0][0][0].size();

    int ke_height = kernel.size();
    int ke_width = kernel[0].size();

    int ou_channels = kernel[0][0][0].size();
    int ou_height = (in_height+2*padding-ke_height)/stride + 1;
    int ou_width = (in_width+2*padding-ke_width)/stride + 1;

    // Intializing Output 
    output.assign(batch,
                  vector<vector<vector<float>>>(ou_height,
                  vector<vector<float>>(ou_width,
                  vector<float>(ou_channels,0))));
    
    // Intializing Padded_input
    Matrix padded_input(batch,
                        vector<vector<vector<float>>>(in_height+2*padding,
                        vector<vector<float>>(in_width+2*padding,
                        vector<float>(in_channels,0))));
    // Adding Padding to the input  
    for(int b=0;b<batch;b++){
        for(int h=0;h<in_height;h++){
            for(int w=0;w<in_width;w++){
                for(int c=0;c<in_channels;c++){
                    padded_input[b][h+padding][w+padding][c] = input[b][h][w][c];
                }
            }
        }
    }

    // Performing the convolution
    for(int b=0;b<batch;b++){
        for(int i_h=0;i_h<ou_height;i_h++){
            for(int i_w=0;i_w<ou_width;i_w++){
                for(int o_c=0;o_c<ou_channels;o_c++){
                    float sum = 0;
                    for(int k_h=0;k_h<ke_height;k_h++){
                        for(int k_w=0;k_w<ke_width;k_w++){
                            for(int i_c=0;i_c<in_channels;i_c++){
                                int h = i_h*stride+k_h;
                                int w = i_w*stride+k_w;
                                if (h>=0 && h<padded_input[b].size() && 
                                    w>=0 && w<padded_input[b][0].size() && i_c<in_channels){
                                        sum+=padded_input[b][h][w][i_c]*kernel[k_h][k_w][i_c][o_c];
                                }
                            }
                        }
                    }
                    output[b][i_h][i_w][o_c] = sum;
                }
            }
        }
    }
    // Writing the output to 1D vector
    vector<float> output_1d;
    for(int b=0;b<batch;b++){
        for(int h=0;h<ou_height;h++){
            for(int w=0;w<ou_width;w++){
                for(int c=0;c<ou_channels;c++){
                    output_1d.push_back(output[b][h][w][c]);
                }
            }
        }
    }
    write_to_binary("../outputs/conv_nhwc_cpp.bin",output_1d);
}

int main() {

    vector<int> input_dims = {1,224,224,3};
    vector<int> kernel_dims = {7,7,3,64};
    auto input = read_npy_file("../inputs/py_input_nhwc.npy", input_dims);
    auto kernel = read_npy_file("../weights/conv1_wt_nhwc.npy", kernel_dims);
   
    Matrix output;
    // Perform the convolution
    convnhwc(input, kernel, output, 1, 1);

    // Print the output dimensions and the first value for verification
    cout << "Output shape: [" << output.size() << ", "
         << output[0].size() << ", "
         << output[0][0].size() << ", "
         << output[0][0][0].size() << "]" << endl;
    cout << "First output value: " << output[0][0][0][0] << endl;
    return 0;
}