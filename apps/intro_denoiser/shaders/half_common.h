/* 
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef HALF_COMMON_H
#define HALF_COMMON_H

#include "app_config.h"

#if !USE_FP32_OUTPUT

#include <cuda_fp16.h>

// CUDA doesn't implement half4? Run my own class.
// Align the struct to 8 bytes to get vectorized ld.v4.u16 and st.v4.u16 instructions.
struct __align__(8) Half4
{
  half x;
  half y;
  half z;
  half w;
};

__forceinline__ __host__ __device__ Half4 make_Half4(const half x, const half y, const half z, const half w)
{
  Half4 h4;

  h4.x = x;
  h4.y = y;
  h4.z = z;
  h4.w = w;
  
  return h4;
}

__forceinline__ __host__ __device__ Half4 make_Half4(const float x, const float y, const float z, float w)
{
  Half4 h4;

  h4.x = __float2half(x);
  h4.y = __float2half(y);
  h4.z = __float2half(z);
  h4.w = __float2half(w);
  
  return h4;
}

__forceinline__ __host__ __device__ Half4 make_Half4(float3 const& v, float w)
{
  Half4 h4;

  h4.x = __float2half(v.x);
  h4.y = __float2half(v.y);
  h4.z = __float2half(v.z);
  h4.w = __float2half(w);
  
  return h4;
}

#endif

#endif // HALF_COMMON_H
