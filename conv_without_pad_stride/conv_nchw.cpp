#include <iostream>
#include <vector>

using namespace std;

typedef vector<vector<vector<vector<int>>>> Matrix;

Matrix conv3D(const Matrix &input, const Matrix &kernel){
    //input parameters
    int in_batches = input.size();
    int in_channels = input[0].size();
    int in_height = input[0][0].size();
    int in_width = input[0][0][0].size();
    //kernel parameters
    int no_kernels = kernel.size();
    int ke_channels = kernel[0].size();
    int ke_height = kernel[0][0].size();
    int ke_width = kernel[0][0][0].size();
    //output parameters
    int ou_height = in_height - ke_height + 1;
    int ou_width = in_width - ke_width + 1;
    int ou_batches = input.size();
    int ou_channels = kernel.size();
    //output intialization
    Matrix output(ou_batches,
                    vector<vector<vector<int>>>(ou_channels,
                    vector<vector<int>>(ou_height,
                    vector<int>(ou_width,0)))); 
    for (int batch=0;batch<in_batches;batch++){
        for (int o_c=0;o_c<ou_channels;o_c++){
            for (int i_h=0;i_h<ou_height;i_h++){
                for (int i_w=0;i_w<ou_width;i_w++){
                    int sum = 0;
                    for (int k_c=0;k_c<ke_channels;k_c++){
                        for (int k_h=0;k_h<ke_height;k_h++){
                            for (int k_w=0;k_w<ke_width;k_w++){
                                sum+=input[batch][k_c][i_h+k_h][i_w+k_w]*kernel[o_c][k_c][k_h][k_w];
                            }
                        }
                    }
                    output[batch][o_c][i_h][i_w]=sum;
                }
            }
        }
    }
    return output;
}

int main(){
        Matrix input = {
        //1st image
        {{{1,2,3,4,5},{2,3,4,5,1},{3,4,5,1,2},{4,5,1,2,3},{5,1,2,3,4}},  //1st channel
         {{2,3,4,5,1},{3,4,5,1,2},{4,5,1,2,3},{5,1,2,3,4},{1,2,3,4,5}},  //2nd channel
         {{3,4,5,1,2},{4,5,1,2,3},{5,1,2,3,4},{1,2,3,4,5},{2,3,4,5,1}}}, //3rd channel
        //2nd image
        {{{4,5,1,2,3},{5,1,2,3,4},{1,2,3,4,5},{2,3,4,5,1},{3,4,5,1,2}},  //1st channel
         {{5,1,2,3,4},{1,2,3,4,5},{2,3,4,5,1},{3,4,5,1,2},{4,5,1,2,3}},  //2nd channel
         {{1,2,3,4,5},{2,3,4,5,1},{3,4,5,1,2},{4,5,1,2,3},{5,1,2,3,4}}}, //3rd channel
        //3rd image
        {{{2,3,4,5,1},{3,4,5,1,2},{4,5,1,2,3},{5,1,2,3,4},{1,2,3,4,5}},  //1st channel
         {{3,4,5,1,2},{4,5,1,2,3},{5,1,2,3,4},{1,2,3,4,5},{2,3,4,5,1}},  //2nd channel
         {{4,5,1,2,3},{5,1,2,3,4},{1,2,3,4,5},{2,3,4,5,1},{3,4,5,1,2}}}  //3rd channel
    };
    //kernel_size: 3x3x3x3 - Number of filters, channels, height, width
    Matrix kernel = {
        {{{1,2,3},{2,3,1},{3,2,1}},
         {{2,3,1},{3,2,1},{1,2,3}},
         {{3,2,1},{1,2,3},{2,3,1}}},
        {{{1,0,-1},{1,0,-1},{1,0,-1}},
         {{1,0,-1},{1,0,-1},{1,0,-1}},
         {{1,0,-1},{1,0,-1},{1,0,-1}}},
        {{{-1,0,1},{-1,0,1},{-1,0,1}},
         {{-1,0,1},{-1,0,1},{-1,0,1}},
         {{-1,0,1},{-1,0,1},{-1,0,1}}}
    };
    
    Matrix conv = conv3D(input,kernel);
    for (int b = 0; b < conv.size(); ++b) {
        cout << "Batch " << b << ":" << endl;
        for (int c = 0; c < conv[0].size(); ++c) {
            cout << "  Channel " << c << ":" << endl;
            for (int h = 0; h < conv[0][0].size(); ++h) {
                for (int w = 0; w < conv[0][0][0].size(); ++w) {
                    cout << conv[b][c][h][w] << " ";
                }
                cout << endl; 
            }
            cout << endl; 
        }
        cout << endl; 
    }
    return 0;
}