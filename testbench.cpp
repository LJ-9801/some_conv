#include "naive_convs.h"


int main(){
  int in_channel = 3;
  int out_channel = 32;
  int input_width = 224;
  int input_height = 224;
  int group = 1;
  vector<uint32_t> kernels = {3,3};
  vector<uint32_t> padding = {1,1};
  vector<uint32_t> dilation = {1,1};
  vector<uint32_t> stride = {1,1};

  Conv_params_t params = Conv_params(
                            in_channel, out_channel, 
                            input_width, input_height,
                            kernels, padding,
                            dilation, stride, group
                            );

  float* input = new float[params.input_size];
  float* weight = new float[params.weight_size];
  float* output = new float[params.output_size];

  for(int i = 0; i<params.input_size; i++){
    input[i] = 1;
  }

  for(int i = 0; i<params.weight_size; i++){
    weight[i] = 1;
  }
  
  conv<float>(input, weight, output, params);

  delete[] input;
  delete[] output;
  delete[] weight;

  return 0;
}