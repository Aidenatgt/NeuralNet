#pragma once
#include <cuda_runtime.h>
#include <math.h>

enum UnaryOp : int {
  OP_RELU = 0,
  OP_LEAKY_RELU = 1, // p0 = alpha
  OP_SILU = 2,
  OP_GELU = 3,
  OP_TANH = 4,
  OP_SIGMOID = 5,
  OP_SOFTPLUS = 6 // optional; stable form
};

__device__ __forceinline__ float apply_op(float x, int op, float p0) {
  switch (op) {
  case OP_RELU:
    return x > 0.f ? x : 0.f;
  case OP_LEAKY_RELU:
    return x > 0.f ? x : p0 * x;
  case OP_SILU: {
    float s = 1.f / (1.f + __expf(-x));
    return x * s;
  }
  case OP_GELU: {
    // tanh approx
    float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + tanhf(u));
  }
  case OP_TANH:
    return tanhf(x);
  case OP_SIGMOID: {
    // stable sigmoid
    if (x >= 0.f) {
      float e = __expf(-x);
      return 1.f / (1.f + e);
    } else {
      float e = __expf(x);
      return e / (1.f + e);
    }
  }
  case OP_SOFTPLUS: {
    // log1p(exp(x)) in a stable form
    float ax = fabsf(x);
    return fmaxf(x, 0.f) + log1pf(__expf(-ax));
  }
  default:
    return x; // no-op fallback
  }
}

__device__ __forceinline__ float apply_op_grad(float x, int op, float p0) {
  switch (op) {
  case OP_RELU:
    return x > 0.f ? 1.f : 0.f;
  case OP_LEAKY_RELU:
    return x > 0.f ? 1.f : p0;
  // TODO: Replace everything below this with a derivative
  case OP_SILU: {
    float s = 1.f / (1.f + __expf(-x));
    return x * s;
  }
  case OP_GELU: {
    // tanh approx
    float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + tanhf(u));
  }
  case OP_TANH:
    return tanhf(x);
  case OP_SIGMOID: {
    // stable sigmoid
    if (x >= 0.f) {
      float e = __expf(-x);
      return 1.f / (1.f + e);
    } else {
      float e = __expf(x);
      return e / (1.f + e);
    }
  }
  case OP_SOFTPLUS: {
    // log1p(exp(x)) in a stable form
    float ax = fabsf(x);
    return fmaxf(x, 0.f) + log1pf(__expf(-ax));
  }
  default:
    return x; // no-op fallback
  }
}

__global__ void map_unary_inplace(float *in, float *out, size_t n, int op,
                                  float p0) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += stride) {
    out[i] = apply_op(in[i], op, p0);
  }
}
__global__ void map_unary_grad_inplace(float *in, float *out, size_t n, int op,
                                       float p0) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += stride) {
    out[i] = apply_op_grad(in[i], op, p0);
  }
}
