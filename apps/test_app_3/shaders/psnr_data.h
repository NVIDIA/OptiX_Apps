
#ifndef PSNR_DATA_H
#define PSNR_DATA_H

#include <cuda.h>

struct PsnrData {
  CUdeviceptr outputBuffer;
  CUdeviceptr outputBuffer_ref;
  CUdeviceptr workspace;

  uint32_t num_pixels;
};

#endif // PSNR_DATA_H
