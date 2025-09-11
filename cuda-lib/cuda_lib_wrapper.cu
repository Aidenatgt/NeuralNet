#include "cuda_lib.cuh"
#include <cstdio>

extern "C" void t(void *a_ptr, void *d_ptr, size_t r, size_t c) {
  dim3 block(16, 16);
  dim3 grid((unsigned)((c + block.x - 1) / block.x),
            (unsigned)((r + block.y - 1) / block.y));

  transpose_kernel<<<grid, block>>>(a_ptr, d_ptr, r, c);
  cudaDeviceSynchronize();
}
extern "C" void *mmul(void *a_ptr, void *b_ptr, size_t r, size_t c) {
  void *d_ptr = nullptr;
  cudaMalloc(&d_ptr, r * c * sizeof(float));
  dim3 block(16, 16);
  dim3 grid((unsigned)((c + block.x - 1) / block.x),
            (unsigned)((r + block.y - 1) / block.y));

  mmul_kernel<<<grid, block>>>(a_ptr, b_ptr, d_ptr, r, c);
  cudaDeviceSynchronize();
}
extern "C" void *emul(void *a_ptr, void *b_ptr, void *d_ptr, size_t n) {
  dim3 block(16, 16);
  dim3 grid((unsigned)n);

  emul_kernel<<<grid, block>>>(a_ptr, b_ptr, d_ptr, n);
  cudaDeviceSynchronize();
}
extern "C" void *eadd(void *a_ptr, void *b_ptr, void *d_ptr, size_t n) {
  dim3 block(16, 16);
  dim3 grid((unsigned)n);

  eadd_kernel<<<grid, block>>>(a_ptr, b_ptr, d_ptr, n);
  cudaDeviceSynchronize();
}
extern "C" void *esub(void *a_ptr, void *b_ptr, void *d_ptr, size_t n) {
  dim3 block(16, 16);
  dim3 grid((unsigned)n);

  esub_kernel<<<grid, block>>>(a_ptr, b_ptr, d_ptr, n);
  cudaDeviceSynchronize();
}
extern "C" void *unary_op(void *a_ptr, void *d_ptr, int op, size_t n) {
  dim3 block(16, 16);
  dim3 grid((unsigned)n);

  unary_op_kernel<<<grid, block>>>(a_ptr, d_ptr, n, op);
  cudaDeviceSynchronize();
}
extern "C" void *unary_op_grad(void *a_ptr, void *d_ptr, int op, size_t n) {
  dim3 block(16, 16);
  dim3 grid((unsigned)n);

  unary_op_grad_kernel<<<grid, block>>>(a_ptr, d_ptr, n, op);
  cudaDeviceSynchronize();
}
