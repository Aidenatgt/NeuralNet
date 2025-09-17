#include "include/cuda_lib.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <math.h>

// Make sure your enum values match your Rust #[repr(i32)]
enum UnaryOp : int {
  OP_RELU = 0,
  OP_LEAKY_RELU = 1,
  OP_SILU = 2,
  OP_GELU = 3,
  OP_TANH = 4,
  OP_SIGMOID = 5,
  OP_SOFTPLUS = 6
};

__device__ __forceinline__ float sigmoidf_fast(float x) {
  // stable sigmoid
  if (x >= 0.f) {
    float e = __expf(-x);
    return 1.f / (1.f + e);
  } else {
    float e = __expf(x);
    return e / (1.f + e);
  }
}

__device__ __forceinline__ float apply_op(float x, int op) {
  switch (op) {
  case OP_RELU:
    return x > 0.f ? x : 0.f;
  case OP_LEAKY_RELU:
    return x > 0.f ? x : 0.1f * x;

  case OP_SILU: {
    float s = sigmoidf_fast(x);
    return x * s;
  }

  case OP_GELU: {
    // tanh approximation
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x2 = x * x;
    float u = k0 * (x + k1 * x * x2);
    return 0.5f * x * (1.f + tanhf(u));
  }

  case OP_TANH:
    return tanhf(x);

  case OP_SIGMOID:
    return sigmoidf_fast(x);

  case OP_SOFTPLUS: {
    // softplus(x) = max(x,0) + log1p(exp(-|x|))
    float ax = fabsf(x);
    return fmaxf(x, 0.f) + log1pf(__expf(-ax));
  }

  default:
    return x; // identity
  }
}

__device__ __forceinline__ float apply_op_grad(float x, int op) {
  switch (op) {
  case OP_RELU:
    // choose subgradient at 0 as 0
    return x > 0.f ? 1.f : 0.f;

  case OP_LEAKY_RELU:
    return x > 0.f ? 1.f : 0.1f;

  case OP_SILU: {
    // d/dx x*s = s + x*s*(1-s) = s*(1 + x*(1-s))
    float s = sigmoidf_fast(x);
    return s * (1.f + x * (1.f - s));
  }

  case OP_GELU: {
    // d/dx 0.5*x*(1+tanh(u)), u=k0*(x+k1*x^3)
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x2 = x * x;
    float u = k0 * (x + k1 * x * x2);
    float t = tanhf(u);
    float sech2 = 1.f - t * t;               // sech^2(u)
    float dudx = k0 * (1.f + 3.f * k1 * x2); // k0*(1 + 0.134145*x^2)
    return 0.5f * (1.f + t) + 0.5f * x * sech2 * dudx;
  }

  case OP_TANH: {
    float t = tanhf(x);
    return 1.f - t * t;
  }

  case OP_SIGMOID: {
    float s = sigmoidf_fast(x);
    return s * (1.f - s);
  }

  case OP_SOFTPLUS:
    // derivative is sigmoid(x)
    return sigmoidf_fast(x);

  default:
    return 1.f; // derivative of identity
  }
}

__global__ void unary_op_kernel(void *in, void *out, int n, int op) {
  float *x = static_cast<float *>(in);
  float *y = static_cast<float *>(out);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    y[i] = apply_op(x[i], op);
  }
}
__global__ void unary_op_grad_kernel(void *in, void *out, int n, int op) {
  float *x = static_cast<float *>(in);
  float *y = static_cast<float *>(out);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    y[i] = apply_op_grad(x[i], op);
  }
}

extern "C" void *unary_op_f32(float *a_ptr, float *d_ptr, int op, int n) {
  // Initialize runtime on primary context (helps when mixing with Driver API)
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

  unary_op_kernel<<<blocks, threads>>>(a_ptr, d_ptr, n, op);

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

extern "C" void *unary_op_grad_f32(float *a_ptr, float *d_ptr, int op, int n) {
  // Initialize runtime on primary context (helps when mixing with Driver API)
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

  unary_op_kernel<<<blocks, threads>>>(a_ptr, d_ptr, n, op);

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
