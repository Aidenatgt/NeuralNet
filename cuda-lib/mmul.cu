#include "include/cuda_lib.h"
#include <cstdio>
#include <cuda_runtime.h>

#ifndef TILE
#define TILE 16
#endif

__global__ void gemm_naive(float *A, float *B, float *C, int M, int K, int N) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = tmp + C[x * N + y];
  }
}
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

extern "C" float *mmul_f32(float *a_ptr, float *b_ptr, int M, int K, int N) {
  (void)cudaFree(0);
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
  e = cudaMalloc(&C, (size_t)M * (size_t)N * sizeof(float));
  if (e != cudaSuccess) {
    fprintf(stderr, "cudaMalloc C: %s\n", cudaGetErrorString(e));
    return nullptr;
  }

  // Fill with 0xFF so if kernel doesn't run you won't get silent 0s
  cudaMemset(C, 0xFF, (size_t)M * (size_t)N * sizeof(float));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  // Clear any sticky error first
  (void)cudaGetLastError();

  gemm_tiled<<<grid, block>>>(a_ptr, b_ptr, C, M, K, N);

  // Catch immediate launch errors
  if (chk("launch(gemm_tiled)") != cudaSuccess) {
    printf("Here's trouble\n");
    return C;
  }

  // Catch async errors
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    fprintf(stderr, "sync(gemm_tiled): %s\n", cudaGetErrorString(e));
    (void)cudaGetLastError(); // clear sticky
  }

  return C;
}
