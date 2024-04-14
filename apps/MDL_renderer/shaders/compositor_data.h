/* 
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef COMPOSITOR_DATA_H
#define COMPOSITOR_DATA_H

#include <cuda.h>

struct CompositorData
{
  // 8 byte alignment
  CUdeviceptr outputBuffer;
  CUdeviceptr tileBuffer;

  int2 resolution;  // The actual rendering resolution. Independent from the launch dimensions for some rendering strategies.
  int2 tileSize;    // Example: make_int2(8, 4) for 8x4 tiles. Must be a power of two to make the division a right-shift.
  int2 tileShift;   // Example: make_int2(3, 2) for the integer division by tile size. That actually makes the tileSize redundant. 

  // 4 byte alignment
  int launchWidth;  // The orignal launch width. Needed to calculate the source data index. The compositor launch gridDim.x * blockDim.x might be different!
  int deviceCount;  // Number of devices doing the rendering.
  int deviceIndex;  // Device index to be able to distinguish the individual devices in a multi-GPU environment.
};

#endif // COMPOSITOR_DATA_H
