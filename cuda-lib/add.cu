#include "include/cuda_lib.h"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_kernel(float *a_ptr, float *b_ptr, float *d_ptr, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    d_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

extern "C" float *add_f32(float *a_ptr, float *b_ptr, float *d_ptr, int n) {
  (void)cudaFree(0);

  auto chk = [](const char *tag) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
      fprintf(stderr, "[CUDA] %s: %s\n", tag, cudaGetErrorString(e));
    return e;
  };

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  // Clear any sticky error first
  (void)cudaGetLastError();

  add_kernel<<<blocks, threads>>>(a_ptr, b_ptr, d_ptr, n);

  // Catch immediate launch errors
  if (chk("launch(gemm_tiled)") != cudaSuccess)
    return d_ptr;

  // Catch async errors
  auto e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    fprintf(stderr, "sync(gemm_tiled): %s\n", cudaGetErrorString(e));
    (void)cudaGetLastError(); // clear sticky
  }

  return d_ptr;
}
