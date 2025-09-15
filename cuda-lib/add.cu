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
  int blockSize = 256;
  int gridSize = static_cast<int>((n + blockSize - 1) / blockSize);
  if (gridSize <= 0)
    gridSize = 1;

  float *x = static_cast<float *>(a_ptr);
  float *y = static_cast<float *>(b_ptr);
  float *z = static_cast<float *>(d_ptr);

  printf("Starting add\n");

  add_kernel<<<gridSize, blockSize>>>(x, y, z, n);
  auto st = cudaGetLastError();
  if (st != cudaSuccess)
    fprintf(stderr, "eadd launch error: %s\n", cudaGetErrorString(st));
  cudaDeviceSynchronize();
  printf("Ending add\n");

  return d_ptr;
}
