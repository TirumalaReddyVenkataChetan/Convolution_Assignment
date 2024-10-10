#include <iostream>
#include <vector>

using namespace std;

typedef vector<vector<vector<vector<int>>>> Matrix;

Matrix conv3D(const Matrix &input, const Matrix &kernel) {
    // Input parameters
    int in_batches = input.size();
    int in_height = input[0].size();
    int in_width = input[0][0].size();
    int in_channels = input[0][0][0].size();

    // Kernel parameters
    int no_kernels = kernel.size();
    int ke_height = kernel[0].size();
    int ke_width = kernel[0][0].size();
    int ke_channels = kernel[0][0][0].size();

    // Output parameters
    int ou_height = in_height - ke_height + 1;
    int ou_width = in_width - ke_width + 1;
    int ou_batches = input.size();
    int ou_channels = kernel.size();

    // Output initialization
    Matrix output(ou_batches,
                  vector<vector<vector<int>>>(ou_height,
                  vector<vector<int>>(ou_width,
                  vector<int>(ou_channels, 0))));

    for (int batch = 0; batch < in_batches; batch++) {
        for (int o_h = 0; o_h < ou_height; o_h++) {
            for (int o_w = 0; o_w < ou_width; o_w++) {
                for (int o_c = 0; o_c < ou_channels; o_c++) {
                    int sum = 0;
                    for (int k_h = 0; k_h < ke_height; k_h++) {
                        for (int k_w = 0; k_w < ke_width; k_w++) {
                            for (int i_c = 0; i_c < in_channels; i_c++) {
                                sum += input[batch][o_h + k_h][o_w + k_w][i_c] * kernel[o_c][k_h][k_w][i_c];
                            }
                        }
                    }
                    output[batch][o_h][o_w][o_c] = sum;
                }
            }
        }
    }
    return output;
}

int main(){
    //input_size: 3x3x5x5 - batch_size, channels, height, width
    Matrix input = {
    {{{1, 2, 3},{2, 3, 4},{3, 4, 5},{4, 5, 1},{5, 1, 2}},
     {{2, 3, 4},{3, 4, 5},{4, 5, 1},{5, 1, 2},{1, 2, 3}},
     {{3, 4, 5},{4, 5, 1},{5, 1, 2},{1, 2, 3},{2, 3, 4}},
     {{4, 5, 1},{5, 1, 2},{1, 2, 3},{2, 3, 4},{3, 4, 5}},
     {{5, 1, 2},{1, 2, 3},{2, 3, 4},{3, 4, 5},{4, 5, 1}}},

    {{{4, 5, 1},{5, 1, 2},{1, 2, 3},{2, 3, 4},{3, 4, 5}},
     {{5, 1, 2},{1, 2, 3},{2, 3, 4},{3, 4, 5},{4, 5, 1}},
     {{1, 2, 3},{2, 3, 4},{3, 4, 5},{4, 5, 1},{5, 1, 2}},
     {{2, 3, 4},{3, 4, 5},{4, 5, 1},{5, 1, 2},{1, 2, 3}},
     {{3, 4, 5},{4, 5, 1},{5, 1, 2},{1, 2, 3},{2, 3, 4}}},

    {{{2, 3, 4},{3, 4, 5},{4, 5, 1},{5, 1, 2},{1, 2, 3}},
     {{3, 4, 5},{4, 5, 1},{5, 1, 2},{1, 2, 3},{2, 3, 4}},
     {{4, 5, 1},{5, 1, 2},{1, 2, 3},{2, 3, 4},{3, 4, 5}},
     {{5, 1, 2},{1, 2, 3},{2, 3, 4},{3, 4, 5},{4, 5, 1}},
     {{1, 2, 3},{2, 3, 4},{3, 4, 5},{4, 5, 1},{5, 1, 2}}}
    };

    //kernel_size: 3x3x3x3 - Number of filters, channels, height, width
    Matrix kernel = {
    {{{ 1,  2,  3},{ 2,  3,  2},{ 3,  1,  1}},
     {{ 2,  3,  1},{ 3,  2,  2},{ 1,  1,  3}},
     {{ 3,  1,  2},{ 2,  2,  3},{ 1,  3,  1}}},
    {{{ 1,  1,  1},{ 0,  0,  0},{-1, -1, -1}},
     {{ 1,  1,  1},{ 0,  0,  0},{-1, -1, -1}},
     {{ 1,  1,  1},{ 0,  0,  0},{-1, -1, -1}}},
    {{{-1, -1, -1},{ 0,  0,  0},{ 1,  1,  1}},
     {{-1, -1, -1},{ 0,  0,  0},{ 1,  1,  1}},
     {{-1, -1, -1},{ 0,  0,  0},{ 1,  1,  1}}}
    };


    Matrix conv = conv3D(input, kernel);
    for (int b = 0; b < conv.size(); ++b) {
        cout << "Batch " << b << ":" << endl;
        for (int h = 0; h < conv[0].size(); ++h) {
            for (int w = 0; w < conv[0][0].size(); ++w) {
                for (int c = 0; c < conv[0][0][0].size(); ++c) {
                    cout << conv[b][h][w][c] << " ";
                }
                cout << endl; 
            }
            cout << endl; 
        }
        cout << endl; 
    }
    return 0;
}