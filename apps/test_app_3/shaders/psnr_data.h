
#ifndef PSNR_DATA_H
#define PSNR_DATA_H

#include <cuda.h>

struct PsnrData {
  CUdeviceptr outputBuffer;
  CUdeviceptr outputBuffer_ref;

  int2 resolution;
};

#endif // PSNR_DATA_H
