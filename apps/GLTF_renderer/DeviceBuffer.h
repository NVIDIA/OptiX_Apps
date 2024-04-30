//
// Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <cuda.h>

#include "CheckMacros.h"

struct DeviceBuffer
{
  DeviceBuffer()
    : d_ptr(0)
    , h_ptr(nullptr)
    , size(0)
    , count(0)
  {
  }

  void free()
  {
    if (d_ptr)
    {
      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_ptr)) );
      d_ptr = 0;
    }
    if (h_ptr)
    {
      delete [] h_ptr;
      h_ptr = nullptr;
    }
    size = 0;
    count = 0;
  }

  CUdeviceptr    d_ptr; // Device pointer of the array of the target type. The type is implicitly defined by the usage in this implementation.
  unsigned char* h_ptr; // Temporary host copy to be able to implement all algorithms on the CPU first.
  size_t         size;  // Size in bytes of the host and device buffers.
  size_t         count; // Number of elements in this DeviceBuffer, same as Accessor.count.
};
