//
// Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <utility>

#include "CheckMacros.h"

class DeviceBuffer
{
public:
  DeviceBuffer()
    : d_ptr(0)
    , size(0)
    , count(0)
  {
  }

  ~DeviceBuffer()
  {
    if (d_ptr)
    {
      CUDA_CHECK_NO_THROW( cudaFree(reinterpret_cast<void*>(d_ptr)) );
      d_ptr = 0;
    }
    size = 0;
    count = 0;
  }

  // Move constructor from another DeviceBuffer.
  DeviceBuffer(DeviceBuffer&& that) noexcept
  {
    operator=(std::move(that));
  }

  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(DeviceBuffer&)       = delete;

  // Move operator (preventing that the destructor of "that" is called on the copied pointers).
  DeviceBuffer& operator=(DeviceBuffer&& that) noexcept
  {
    d_ptr = that.d_ptr;
    size  = that.size;
    count = that.count;

    that.d_ptr = 0;
    that.size  = 0;
    that.count = 0;
    
    return *this;
  }

  // These two helper functions allow using the DeviceBuffer as "grow" buffer
  // for persistent data with a maximum size holding device allocations 
  // which are reused for skinning or IAS build and update.
  
  // Explicit deallocation call, needed on grow buffers when switching scenes etc.
  void clear()
  {
    if (d_ptr)
    {
      CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_ptr)) );
      d_ptr = 0;
    }
    size = 0;
    count = 0;
  }

  // This is destructive, the data in d_ptr is discarded when a grow happens.
  void grow(const size_t sizeGrow)
  {
    if (size < sizeGrow)
    {
      if (d_ptr)
      {
        CUDA_CHECK( cudaFree(reinterpret_cast<void*>(d_ptr)) ); // This discards the current data.
        d_ptr = 0;
      }
      CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&d_ptr), sizeGrow) );
      size = sizeGrow;
      count = 0; // HACK The grow buffers do not make use of the count member.
    }
  }

public:
  CUdeviceptr d_ptr; // Device pointer of the array of the target type. The type is implicitly defined by the usage in this implementation.
  size_t      size;  // Size in bytes of the device buffer.
  size_t      count; // Number of elements in this DeviceBuffer, same as Accessor.count. Unused (== 0) when used as grow buffer.
};
