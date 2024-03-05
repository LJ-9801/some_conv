#include <iostream>
#include <cstdint>
#include "common.h"

typedef void (*conv_t)(const float** input, const float** weight, float** output, const Conv_params_t params);


template<typename T>
void naive_conv(const T** input,  const T** weight, T** output, const Conv_params_t params){
  int out_c = params.out_channel;
  int in_c = params.in_channel;
  int in_h = params.input_height;
  int in_w = params.input_width;
  int p_w = params.paddings[0];
  int p_h = params.paddings[1]; 
  int kernel_h = params.kernels[0];
  int kernel_w = params.kernels[1];
  int dh = params.dilation[0];
  int dw = params.dilation[1]; 
  int out_h = params.output_height;
  int out_w = params.output_width;
  int stride_h = params.stride[0];
  int stride_w = params.stride[1];

  #pragma omp parallel for collapse(3)
  for(int oc = 0; oc < out_c; oc++){
    for(int oh = 0; oh < out_h; oh++){
      for(int ow = 0; ow < out_w; ow++){
        T sum = 0;
        for(int ic = 0; ic < in_c; ic++){
          for(int kh = 0; kh < kernel_h; kh++){
            for(int kw = 0; kw < kernel_w; kw++){
              int ih = oh * stride_h + kh * dh - p_h;
              int iw = ow * stride_w + kw * dw - p_w;
              if(ih >= 0 && ih < in_h && iw >= 0 && iw < in_w){
                int input_index = ic * in_h * in_w + ih * in_w + iw;
                int weight_index = oc * in_c * kh * kw + ic * kh * kw + kh * kw;
                sum += (*input)[input_index] * (*weight)[weight_index];
              }
            }
          }
        }
        int output_index = oc * out_h * out_w + oh * out_w + ow;
        (*output)[output_index] = sum;
      }
    }
  }
}

template<typename T>
void awesome_conv(const T** input,  const T** weight, T** output, const Conv_params_t params){
 
}

template<typename T, conv_t F = naive_conv>
void conv(const T* input, const T* weight, T* output, const Conv_params_t& params){
  // do convolution
  F(&input, &weight, &output, params);
  
}