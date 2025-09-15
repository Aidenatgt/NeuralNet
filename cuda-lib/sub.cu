#include "include/cuda_lib.h"
#include <cuda_runtime.h>

__global__ void esub_kernel(void *a_ptr, void *b_ptr, void *d_ptr, int n) {
  float *x = static_cast<float *>(a_ptr);
  float *y = static_cast<float *>(b_ptr);
  float *z = static_cast<float *>(d_ptr);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    z[i] = y[i] - x[i];
  }
}

extern "C" void *sub(void *a_ptr, void *b_ptr, void *d_ptr, int n) {
  dim3 block(16, 16);
  dim3 grid((unsigned)n);

  esub_kernel<<<grid, block>>>(a_ptr, b_ptr, d_ptr, n);
  cudaDeviceSynchronize();

  return d_ptr;
}
