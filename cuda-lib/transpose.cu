#include "include/cuda_lib.h"
#include <cuda_runtime.h>

__global__ void transpose_kernel(void *a, void *b, int r, int c) {
  float *x = static_cast<float *>(a);
  float *y = static_cast<float *>(b);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < r * c; i += stride) {
    int R = i / r;
    int C = i % c;

    int N = C * c + R;

    y[N] = x[i];
  }
}

extern "C" void *t(void *a_ptr, int r, int c) {
  void *d_ptr = nullptr;
  cudaMalloc(&d_ptr, r * c * sizeof(float));
  dim3 block(16, 16);
  dim3 grid((unsigned)((c + block.x - 1) / block.x),
            (unsigned)((r + block.y - 1) / block.y));

  transpose_kernel<<<grid, block>>>(a_ptr, d_ptr, r, c);
  cudaDeviceSynchronize();

  return d_ptr;
}
