#include "include/cuda_lib.h"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_kernel(float *x, float *y, float *z, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    z[i] = y[i] + x[i];
  }
}

extern "C" void *add(void *a_ptr, void *b_ptr, void *d_ptr, int n) {
  dim3 block(16, 16);
  dim3 grid((unsigned)n / 16 + 1);

  float *x = static_cast<float *>(a_ptr);
  float *y = static_cast<float *>(b_ptr);
  float *z = static_cast<float *>(d_ptr);

  add_kernel<<<grid, block>>>(x, y, z, n);
  cudaDeviceSynchronize();

  return d_ptr;
}
