#pragma once
#include <cstddef>
extern "C" void map_unary_f32(float *i_ptr, float *o_ptr, size_t n, int op,
                              float p0);
extern "C" void map_unary_grad_f32(float *i_ptr, float *o_ptr, size_t n, int op,
                                   float p0);
