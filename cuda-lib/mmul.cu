#include "include/cuda_lib.h"
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16
#endif

__global__ void gemm_tiled(float *A, float *B, float *C, int M, int K, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;

  int tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < tiles; ++t) {
    int a_col = t * TILE + threadIdx.x;
    int b_row = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = acc;
}

extern "C" void *mmul(void *a_ptr, void *b_ptr, int M, int K, int N) {
  float *A = static_cast<float *>(a_ptr);
  float *B = static_cast<float *>(b_ptr);

  float *C = nullptr;
  cudaMalloc(&C, M * N * sizeof(float));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  gemm_tiled<<<grid, block>>>(A, B, C, M, K, N);
  cudaDeviceSynchronize();

  return C;
}
