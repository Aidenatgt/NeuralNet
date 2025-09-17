#include "include/cuda_lib.h"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void transpose_kernel(float *a, float *b, int r, int c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < r * c; i += stride) {
    int R = i / r;
    int C = i % c;

    int N = C * c + R;

    b[N] = a[i];
  }
}

extern "C" void *t_f32(float *a_ptr, int r, int c) {
  float *C = nullptr;

  // Initialize runtime on primary context (helps when mixing with Driver API)
  (void)cudaFree(0);

  auto chk = [](const char *tag) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
      fprintf(stderr, "[CUDA] %s: %s\n", tag, cudaGetErrorString(e));
    return e;
  };

  cudaError_t e;
  e = cudaMalloc(&C, (size_t)c * (size_t)r * sizeof(float));
  if (e != cudaSuccess) {
    fprintf(stderr, "cudaMalloc C: %s\n", cudaGetErrorString(e));
    return nullptr;
  }

  // Fill with 0xFF so if kernel doesn't run you won't get silent 0s
  cudaMemset(C, 0xFF, (size_t)c * (size_t)r * sizeof(float));
  dim3 block(16, 16);
  dim3 grid((unsigned)((c + block.x - 1) / block.x),
            (unsigned)((r + block.y - 1) / block.y));

  transpose_kernel<<<grid, block>>>(a_ptr, C, r, c);

  // Catch immediate launch errors
  if (chk("launch(gemm_tiled)") != cudaSuccess)
    return C;

  // Catch async errors
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    fprintf(stderr, "sync(gemm_tiled): %s\n", cudaGetErrorString(e));
    (void)cudaGetLastError(); // clear sticky
  }

  return C;
}
