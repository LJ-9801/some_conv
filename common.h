#include <vector>
#include <assert.h>
#include <iostream>

using namespace std;

typedef struct Conv_params{ 
  int in_channel;
  int out_channel;
  int input_width;
  int input_height;
  vector<uint32_t> kernels;
  vector<uint32_t> paddings;
  vector<uint32_t> dilation;
  vector<uint32_t> stride;
  int group;


  int output_width;
  int output_height;

  int input_size;
  int weight_size;
  int output_size;

  Conv_params(int in_c, int out_c, int in_w, int in_h,
              vector<uint32_t> k, vector<uint32_t> p,
              vector<uint32_t> dil, vector<uint32_t> stri, int group):
              in_channel(in_c), out_channel(out_c), input_width(in_w), input_height(in_h),
              kernels(k), paddings(p),
              dilation(dil), stride(stri), group(group){
    
    assert(dilation[0] >= 1 && dilation[1] >= 1);

    output_height = ((input_height + 2*paddings[0] - dilation[0] * (kernels[0] - 1)-1) / stride[0]) + 1;
    output_width  = ((input_width + 2*paddings[1] - dilation[1] * (kernels[1] - 1)-1) / stride[1]) + 1;

    input_size = in_channel * input_height * input_width;
    weight_size = out_channel * in_channel * kernels[0] * kernels[1];
    output_size = out_channel * output_height * output_width;
  }

  Conv_params(int in_c, int out_c, int in_w, int in_h, vector<uint32_t> kernels):
              in_channel(in_c), out_channel(out_c), input_width(in_w), input_height(in_h),
              kernels(kernels), paddings({0,0}),
              dilation({1,1}), stride({1,1}), group(1){
    
    assert(dilation[0] >= 1 && dilation[1] >= 1);

    output_height = ((input_height + 2*paddings[0] - dilation[0] * (kernels[0] - 1)-1) / stride[0]) + 1;
    output_width  = ((input_width + 2*paddings[1] - dilation[1] * (kernels[1] - 1)-1) / stride[1]) + 1;

    input_size = in_channel * input_height * input_width;
    weight_size = out_channel * in_channel * kernels[0] * kernels[1];
    output_size = out_channel * output_height * output_width;
  }

} Conv_params_t;

template<typename T>
void im2col(const T* input, const T* weight, T* A, T* B, const Conv_params_t& params){
  int in_c = params.in_channel;
  int in_h = params.input_height;
  int in_w = params.input_width;
  int kernel_h = params.kernels[0];
  int kernel_w = params.kernels[1];
  int p_w = params.paddings[0];
  int p_h = params.paddings[1];
  int dh = params.dilation[0];
  int dw = params.dilation[1];
  int out_h = params.output_height;
  int out_w = params.output_width;
  int stride_h = params.stride[0];
  int stride_w = params.stride[1];


  assert(A != nullptr && B != nullptr);

  #pragma omp parallel for
  for(int oc = 0; oc < params.out_channel; oc++){
    for(int kh = 0; kh < kernel_h; kh++){
      for(int kw = 0; kw < kernel_w; kw++){
        for(int oh = 0; oh < out_h; oh++){
          for(int ow = 0; ow < out_w; ow++){
            int ih = oh * stride_h + kh * dh - p_h;
            int iw = ow * stride_w + kw * dw - p_w;
            if(ih >= 0 && ih < in_h && iw >= 0 && iw < in_w){
              int input_index = oc * in_h * in_w + ih * in_w + iw;
              int weight_index = oc * in_c * kernel_h * kernel_w + kh * kernel_w + kw;
              int a_index = oc * in_c * kernel_h * kernel_w * out_h * out_w + kh * kernel_w * out_h * out_w + kw * out_h * out_w + oh * out_w + ow;
              A[a_index] = input[input_index];
              B[a_index] = weight[weight_index];
            }
          }
        }
      }
    }
  }
}


template<typename T>
void gemm(const T* A, const T* B, T* C, uint32_t M, uint32_t N, uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc){

  #pragma omp parallel for collapse(2)
  for(int m = 0; m < M; m++){
    for(int n = 0; n < N; n++){
      T sum = 0;
      for(int k = 0; k < K; k++){
        sum += A[m * lda + k] * B[k * ldb + n];
      }
      C[m * ldc + n] = sum;
    }
  }
}