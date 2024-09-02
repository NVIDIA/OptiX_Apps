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

#ifndef SHADER_COMMON_H
#define SHADER_COMMON_H

#include "config.h"

#include "vector_math.h"


/**
*  Calculates refraction direction
*  r   : refraction vector
*  i   : incident vector
*  n   : surface normal
*  ior : index of refraction ( n2 / n1 )
*  returns false in case of total internal reflection, in that case r is initialized to (0,0,0).
*/
__forceinline__ __host__ __device__ bool refract(float3& r, const float3& i, const float3& n, const float ior)
{
  float3 nn = n;
  float negNdotV = dot(i, nn);
  float eta;

  if (negNdotV > 0.0f)
  {
    eta = ior;
    nn = -n;
    negNdotV = -negNdotV;
  }
  else
  {
    eta = 1.f / ior;
  }

  const float k = 1.f - eta * eta * (1.f - negNdotV * negNdotV);

  if (k < 0.0f)
  {
    // Initialize this value, so that r always leaves this function initialized.
    r = make_float3(0.f);
    return false;
  }
  else
  {
    r = normalize(eta * i - (eta * negNdotV + sqrtf(k)) * nn);
    return true;
  }
}



// Tangent-Bitangent-Normal orthonormal space.
struct TBN
{
  // Default constructor to be able to include it into other structures when needed.
  __forceinline__ __host__ __device__ TBN()
  {
  }

  __forceinline__ __host__ __device__ TBN(const float3& n)
  : normal(n)
  {
    if (fabsf(normal.z) < fabsf(normal.x))
    {
      tangent.x =  normal.z;
      tangent.y =  0.0f;
      tangent.z = -normal.x;
    }
    else
    {
      tangent.x =  0.0f;
      tangent.y =  normal.z;
      tangent.z = -normal.y;
    }
    tangent   = normalize(tangent);
    bitangent = cross(normal, tangent);
  }

  // Constructor for cases where tangent, bitangent, and normal are given as ortho-normal basis.
  __forceinline__ __host__ __device__ TBN(const float3& t, const float3& b, const float3& n)
  : tangent(t)
  , bitangent(b)
  , normal(n)
  {
  }

  // Normal is kept, tangent and bitangent are calculated.
  // Normal must be normalized.
  // Must not be used with degenerated vectors!
  __forceinline__ __host__ __device__ TBN(const float3& tangent_reference, const float3& n)
  : normal(n)
  {
    bitangent = normalize(cross(normal, tangent_reference));
    tangent   = cross(bitangent, normal);
  }

  __forceinline__ __host__ __device__ void negate()
  {
    tangent   = -tangent;
    bitangent = -bitangent;
    normal    = -normal;
  }

  __forceinline__ __host__ __device__ float3 transformToLocal(const float3& p) const
  {
    return make_float3(dot(p, tangent),
                       dot(p, bitangent),
                       dot(p, normal));
  }

  __forceinline__ __host__ __device__ float3 transformToWorld(const float3& p) const
  {
    return p.x * tangent + p.y * bitangent + p.z * normal;
  }

  float3 tangent;
  float3 bitangent;
  float3 normal;
};

// FBN (Fiber, Bitangent, Normal) 
// Special version of TBN (Tangent, Bitangent, Normal) ortho-normal basis generation for fiber geometry.
// The difference is that the TBN keeps the normal intact where the FBN keeps the tangent intact, which is the fiber orientation!
struct FBN
{
  // Default constructor to be able to include it into State.
  __forceinline__ __host__ __device__ FBN()
  {
  }

  // Do not use these single argument constructors for anisotropic materials!
  // There will be discontinuities on round objects when the FBN orientation flips.
  // Tangent is kept, bitangent and normal are calculated.
  // Tangent t must be normalized.
  __forceinline__ __host__ __device__ FBN(const float3& t) // t == fiber orientation
  : tangent(t)
  {
    if (fabsf(tangent.z) < fabsf(tangent.x))
    {
      bitangent.x = -tangent.y;
      bitangent.y =  tangent.x;
      bitangent.z =  0.0f;
    }
    else
    {
      bitangent.x =  0.0f;
      bitangent.y = -tangent.z;
      bitangent.z =  tangent.y;
    }

    bitangent = normalize(bitangent);
    normal    = cross(tangent, bitangent);
  }

  // Constructor for cases where tangent, bitangent, and normal are given as ortho-normal basis.
  __forceinline__ __host__ __device__ FBN(const float3& t, const float3& b, const float3& n) // t == fiber orientation
  : tangent(t)
  , bitangent(b)
  , normal(n)
  {
  }

  // Tangent is kept, bitangent and normal are calculated.
  // Tangent t must be normalized.
  // Must not be used with degenerated vectors!
  __forceinline__ __host__ __device__ FBN(const float3& t, const float3& n)
  : tangent(t)
  {
    bitangent = normalize(cross(n, tangent));
    normal    = cross(tangent, bitangent);
  }

  __forceinline__ __host__ __device__ void negate()
  {
    tangent   = -tangent;
    bitangent = -bitangent;
    normal    = -normal;
  }

  __forceinline__ __host__ __device__ float3 transformToLocal(const float3& p) const
  {
    return make_float3(dot(p, tangent),
                       dot(p, bitangent),
                       dot(p, normal));
  }

  __forceinline__ __host__ __device__ float3 transformToWorld(const float3& p) const
  {
    return p.x * tangent + p.y * bitangent + p.z * normal;
  }

  float3 tangent;
  float3 bitangent;
  float3 normal;
};



__forceinline__ __host__ __device__ float luminance(const float3& rgb)
{
  const float3 ntsc_luminance = { 0.30f, 0.59f, 0.11f };
  return dot(rgb, ntsc_luminance);
}

__forceinline__ __host__ __device__ float intensity(const float3& rgb)
{
  return (rgb.x + rgb.y + rgb.z) * 0.3333333333f;
}

__forceinline__ __host__ __device__ float cube(const float x)
{
  return x * x * x;
}

__forceinline__ __host__ __device__ bool isNull(const float3& v)
{
  return (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f);
}

__forceinline__ __host__ __device__ bool isNotNull(const float3& v)
{
  return (v.x != 0.0f || v.y != 0.0f || v.z != 0.0f);
}

// Used for Multiple Importance Sampling.
__forceinline__ __host__ __device__ float powerHeuristic(const float a, const float b)
{
  const float t = a * a;
  return t / (t + b * b);
}

__forceinline__ __host__ __device__ float balanceHeuristic(const float a, const float b)
{
  return a / (a + b);
}


#endif // SHADER_COMMON_H
