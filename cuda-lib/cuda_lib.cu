#pragma once
#include <cstddef>
extern "C" void t(void *a_ptr, void *d_ptr, int r, int c);
extern "C" void *mmul(void *a_ptr, void *b_ptr, int r, int c);
extern "C" void *emul(void *a_ptr, void *b_ptr, void *d_ptr, int n);
extern "C" void *eadd(void *a_ptr, void *b_ptr, void *d_ptr, int n);
extern "C" void *esub(void *a_ptr, void *b_ptr, void *d_ptr, int n);
extern "C" void *unary_op(void *a_ptr, void *d_ptr, int op, int n);
extern "C" void *unary_op_grad(void *a_ptr, void *d_ptr, int op, int n);
