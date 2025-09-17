#include "include/cuda_lib.h"
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" void cuda_init_runtime_on_primary() { (void)cudaFree(0); }

extern "C" int bind_primary_ctx(int dev_ord) {
  CUdevice d;
  CUcontext ctx;
  if (cuInit(0))
    return 1;
  if (cuDeviceGet(&d, dev_ord))
    return 2;
  if (cuDevicePrimaryCtxRetain(&ctx, d))
    return 3;
  if (cuCtxSetCurrent(ctx))
    return 4;
  (void)cudaFree(0); // attach runtime to this context on this thread
  return 0;
}
