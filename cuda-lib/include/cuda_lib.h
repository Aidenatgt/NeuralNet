#pragma once
#include <stddef.h>

extern "C" {
void *t(void *a_ptr, int r, int c);
void *mmul(void *a_ptr, void *b_ptr, int r, int c, int n);
void *emul(void *a_ptr, void *b_ptr, void *d_ptr, int n);
void *add(void *a_ptr, void *b_ptr, void *d_ptr, int n);
void *sub(void *a_ptr, void *b_ptr, void *d_ptr, int n);
void *unary_op(void *a_ptr, void *d_ptr, int op, int n);
void *unary_op_grad(void *a_ptr, void *d_ptr, int op, int n);
}
