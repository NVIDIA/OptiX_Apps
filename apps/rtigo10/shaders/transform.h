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

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "config.h"

#include <optix.h>

#include "vector_math.h"

// Get the 3x4 object to world transform and its inverse.
__forceinline__ __device__ void getTransforms(const OptixTraversableHandle handle, float4* mW, float4* mO) 
{
  const float4* tW = optixGetInstanceTransformFromHandle(handle);
  const float4* tO = optixGetInstanceInverseTransformFromHandle(handle);

  mW[0] = tW[0];
  mW[1] = tW[1];
  mW[2] = tW[2];

  mO[0] = tO[0];
  mO[1] = tO[1];
  mO[2] = tO[2];
}

// Functions to get the individual transforms in case only one of them is needed.
__forceinline__ __device__ void getTransformObjectToWorld(const OptixTraversableHandle handle, float4* mW) 
{
  const float4* tW = optixGetInstanceTransformFromHandle(handle);

  mW[0] = tW[0];
  mW[1] = tW[1];
  mW[2] = tW[2];
}

__forceinline__ __device__ void getTransformWorldToObject(const OptixTraversableHandle handle, float4* mO) 
{
  const float4* tO = optixGetInstanceInverseTransformFromHandle(handle);

  mO[0] = tO[0];
  mO[1] = tO[1];
  mO[2] = tO[2];
}


// Matrix3x4 * point. v.w == 1.0f
__forceinline__ __device__ float3 transformPoint(const float4* m, const float3& v)
{
  float3 r;

  r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z + m[0].w;
  r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z + m[1].w;
  r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z + m[2].w;

  return r;
}

// Matrix3x4 * vector. v.w == 0.0f
__forceinline__ __device__ float3 transformVector(const float4* m, const float3& v)
{
  float3 r;

  r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
  r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
  r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;

  return r;
}

// (Matrix3x4^-1)^T * normal. v.w == 0.0f
// Takes the inverse matrix as input and applies it transposed.
__forceinline__ __device__ float3 transformNormal(const float4* m, const float3& v)
{
  float3 r;

  r.x = m[0].x * v.x + m[1].x * v.y + m[2].x * v.z;
  r.y = m[0].y * v.x + m[1].y * v.y + m[2].y * v.z;
  r.z = m[0].z * v.x + m[1].z * v.y + m[2].z * v.z;

  return r;
}

// Matrix3x3 * vector.
// Used with light orientation matrices.
__forceinline__ __device__ float3 transformVector(const float3* m, const float3& v)
{
  float3 r;

  r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
  r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
  r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;

  return r;
}

#endif // TRANSFORM_H
