#pragma once
#include <stddef.h>

extern "C" {
float *t_32(float *a_ptr, int r, int c);
float *mmul_32(float *a_ptr, float *b_ptr, int r, int c, int n);
float *emul_32(float *a_ptr, float *b_ptr, float *d_ptr, int n);
float *add_32(float *a_ptr, float *b_ptr, float *d_ptr, int n);
float *sub_32(float *a_ptr, float *b_ptr, float *d_ptr, int n);
float *unary_op_32(float *a_ptr, float *d_ptr, int op, int n);
float *unary_op_grad_32(float *a_ptr, float *d_ptr, int op, int n);

void cuda_init_runtime_on_primary();
int bind_primary_ctx(int device_ordinal);
}
