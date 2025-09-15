#include "include/cuda_lib.h"
#include <cuda_runtime.h>

__global__ void sub_kernel(float *a_ptr, float *b_ptr, float *d_ptr, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    d_ptr[i] = a_ptr[i] - b_ptr[i];
  }
}

extern "C" void *sub(void *a_ptr, void *b_ptr, void *d_ptr, int n) {
  dim3 block(16, 16);
  dim3 grid((unsigned)n / 16 + 1);

  float *x = static_cast<float *>(a_ptr);
  float *y = static_cast<float *>(b_ptr);
  float *z = static_cast<float *>(d_ptr);

  sub_kernel<<<grid, block>>>(x, y, z, n);
  cudaDeviceSynchronize();

  return d_ptr;
}
