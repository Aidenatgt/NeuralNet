#include "unary.cuh"
#include <cstdio>

extern "C" void map_unary_f32(float *i_ptr, float *o_ptr, size_t n, int op,
                              float p0) {
  if (!i_ptr || !o_ptr || n == 0)
    return;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  grid = grid > 65535 ? 65535 : grid;
  map_unary_inplace<<<grid, block>>>(i_ptr, o_ptr, n, op, p0);
  cudaDeviceSynchronize(); // simple: synchronous version
}
extern "C" void map_unary_grad_f32(float *i_ptr, float *o_ptr, size_t n, int op,
                                   float p0) {
  if (!i_ptr || !o_ptr || n == 0)
    return;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  grid = grid > 65535 ? 65535 : grid;
  map_unary_grad_inplace<<<grid, block>>>(i_ptr, o_ptr, n, op, p0);
  cudaDeviceSynchronize(); // simple: synchronous version
}
