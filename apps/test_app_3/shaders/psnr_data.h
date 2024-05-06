
#ifndef PSNR_DATA_H
#define PSNR_DATA_H

#include <cuda.h>

struct PsnrData {
  CUdeviceptr outputBuffer;
  CUdeviceptr outputBuffer_ref;
  CUdeviceptr workspace;

  uint32_t num_pixels;
  uint32_t gridDimX_start;
};

#endif // PSNR_DATA_H
