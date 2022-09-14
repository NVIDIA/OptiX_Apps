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

  if (0.0f < negNdotV)
  {
    eta = ior;
    nn = -n;
    negNdotV = -negNdotV;
  }
  else
  {
    eta = 1.0f / ior;
  }

  const float k = 1.0f - eta * eta * (1.0f - negNdotV * negNdotV);

  if (k < 0.0f)
  {
    // Initialize this value, so that r always leaves this function initialized.
    r = make_float3(0.0f);
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

__forceinline__ __device__ void alignVector(const float3& axis, float3& w)
{
  // Align w with axis.
  const float s = copysignf(1.0f, axis.z);
  w.z *= s;
  const float3 h = make_float3(axis.x, axis.y, axis.z + s);
  const float  k = dot(w, h) / (1.0f + fabsf(axis.z));
  w = k * h - w;
}

__forceinline__ __device__ void unitSquareToCosineHemisphere(const float2 sample, const float3& axis, float3& w, float& pdf)
{
  // Choose a point on the local hemisphere coordinates about +z.
  const float theta = 2.0f * M_PIf * sample.x;
  const float r = sqrtf(sample.y);
  w.x = r * cosf(theta);
  w.y = r * sinf(theta);
  w.z = 1.0f - w.x * w.x - w.y * w.y;
  w.z = (0.0f < w.z) ? sqrtf(w.z) : 0.0f;
 
  pdf = w.z * M_1_PIf;

  // Align with axis.
  alignVector(axis, w);
}

__forceinline__ __device__ void unitSquareToSphere(const float u, const float v, float3& p, float& pdf)
{
  p.z = 1.0f - 2.0f * u;
  float r = 1.0f - p.z * p.z;
  r = (0.0f < r) ? sqrtf(r) : 0.0f;
  
  const float phi = v * 2.0f * M_PIf;
  p.x = r * cosf(phi);
  p.y = r * sinf(phi);

  pdf = 0.25f * M_1_PIf;  // == 1.0f / (4.0f * M_PIf)
}

// Binary-search and return the highest cell index with CDF value <= sample.
// Arguments are the CDF values array pointer, the index of the last element and the random sample in the range [0.0f, 1.0f).
__forceinline__ __device__ unsigned int binarySearchCDF(const float* cdf, const unsigned int last, const float sample)
{
  unsigned int ilo = 0;
  unsigned int ihi = last; // Index on the last entry containing 1.0f. Can never be reached with the sample in the range [0.0f, 1.0f).

  while (ilo + 1 != ihi) // When a pair of limits have been found, the lower index indicates the cell to use.
  {
    const unsigned int i = (ilo + ihi) >> 1;
    
    if (sample < cdf[i]) // If the CDF value is greater than the sample, use that as new higher limit.
    {
      ihi = i;
    }
    else // If the sample is greater than or equal to the CDF value, use that as new lower limit.
    {
      ilo = i; 
    }
  }

  return ilo;
}


// This function evaluates a Fresnel dielectric function when the transmitting cosine ("cost")
// is unknown and the incident index of refraction is assumed to be 1.0f.
// \param et     The transmitted index of refraction.
// \param costIn The cosine of the angle between the incident direction and normal direction.
__forceinline__ __device__ float evaluateFresnelDielectric(const float et, const float cosIn)
{
  const float cosi = fabsf(cosIn);

  float sint = 1.0f - cosi * cosi;
  sint = (0.0f < sint) ? sqrtf(sint) / et : 0.0f;

  // Handle total internal reflection.
  if (1.0f < sint)
  {
    return 1.0f;
  }

  float cost = 1.0f - sint * sint;
  cost = (0.0f < cost) ? sqrtf(cost) : 0.0f;

  const float et_cosi = et * cosi;
  const float et_cost = et * cost;

  const float rPerpendicular = (cosi - et_cost) / (cosi + et_cost);
  const float rParallel      = (et_cosi - cost) / (et_cosi + cost);

  const float result = (rParallel * rParallel + rPerpendicular * rPerpendicular) * 0.5f;

  return (result <= 1.0f) ? result : 1.0f;
}


// Optimized version to calculate D and pdf reusing shared calculations.
__forceinline__ __device__ float2 ggx_D_pdf(const float2 a, const float3& wm)
{
  if (DENOMINATOR_EPSILON < wm.z) // Heaviside function: X_plus(wm * wg). (wm is in tangent space.)
  {
    const float cosThetaSqr = wm.z * wm.z;
    const float tanThetaSqr = (1.0f - cosThetaSqr) / cosThetaSqr;

    const float phiM    = atan2f(wm.y, wm.x);
    const float cosPhiM = cosf(phiM);
    const float sinPhiM = sinf(phiM);

    const float term = 1.0f + tanThetaSqr * ((cosPhiM * cosPhiM) / (a.x * a.x) + (sinPhiM * sinPhiM) / (a.y * a.y));

    const float d   = 1.0f / (M_PIf * a.x * a.y * cosThetaSqr * cosThetaSqr * term * term); // Heitz, Formula (85)
    const float pdf = d * wm.z; // PDF with respect to the half-direction.
      
    return make_float2(d, pdf);
  }
  return make_float2(0.0f);
}

// Return a sample direction in local tangent space coordinates.
__forceinline__ __device__ float3 ggx_sample(const float2 a, const float2 xi)
{
  // Made isotropic to a.y. Output vector scales .x accordingly.
  const float theta    = atanf(a.y * sqrtf(xi.x) / sqrtf(1.0f - xi.x)); // Walter, Formula (35).
  const float phi      = 2.0f * M_PIf * xi.y;                           // Walter, Formula (36).
  const float sinTheta = sinf(theta);
  return normalize(make_float3(cosf(phi) * sinTheta * a.x / a.y,        // Heitz, Formula (77)
                               sinf(phi) * sinTheta,
                               cosf(theta)));
}

// "Microfacet Models for Refraction through Rough Surfaces" - Walter, Marschner, Li, Torrance.
// PERF Using this because it's faster than the approximation below.
__forceinline__ __device__ float smith_G1(const float alpha, const float3& w, const float3& wm)
{
  const float w_wm = dot(w, wm);
  if (w_wm * w.z <= 0.0f) // X_plus(v * m / v * n) from Walter, Formula (34). // PERF Checking the sign with a multiplication here.
  {
    return 0.0f;
  }
  const float cosThetaSqr = w.z * w.z;
  const float sinThetaSqr = 1.0f - cosThetaSqr;
  //const float tanTheta = (0.0f < sinThetaSqr) ? sqrtf(sinThetaSqr) / w.z : 0.0f; // PERF Remove the sqrtf() by calculating tanThetaSqr here
  //const float invA = alpha * tanTheta;                                           // because this is squared below: invASqr = alpha * alpha * tanThetaSqr;
  //const float lambda = (-1.0f + sqrtf(1.0f + invA * invA)) * 0.5f; // Heitz, Formula (86)
  //return 1.0f / (1.0f + lambda);                                   // Heitz, below Formula (69)
  const float tanThetaSqr = (0.0f < sinThetaSqr) ? sinThetaSqr / cosThetaSqr : 0.0f;
  const float invASqr = alpha * alpha * tanThetaSqr;                                           
  return 2.0f / (1.0f + sqrtf(1.0f + invASqr));                     // Optimized version is Walter, Formula (34)
}

// Approximation from "Microfacet Models for Refraction through Rough Surfaces" - Walter, Marschner, Li, Torrance.
//__forceinline__ __device__ float smith_G1(const float alpha, const float3& w, const float3& wm)
//{
//  const float w_wm = optix::dot(w, wm);
//  if (w_wm * w.z <= 0.0f) // X_plus(v * m / v * n) from Walter, Formula (34). // PERF Checking the sign with a multiplication here.
//  {
//    return 0.0f;
//  }
//  const float t        = 1.0f - w.z * w.z; 
//  const float tanTheta = (0.0f < t) ? sqrtf(t) / w.z : 0.0f;
//  if (tanTheta == 0.0f)
//  {
//    return 1.0f;
//  }
//  const float a = 1.0f / (tanTheta * alpha);
//  if (1.6f <= a)
//  {
//    return 1.0f;
//  }
//  const float aSqr = a * a;
//  return (3.535f * a + 2.181f * aSqr) / (1.0f + 2.276f * a + 2.577f * aSqr); // Walter, Formula (27) used for Heitz, Formula (83)
//}

__forceinline__ __device__ float ggx_G(const float2 a, const float3& wo, const float3& wi, const float3& wm)
{
  float phi   = atan2f(wo.y, wo.x);
  float c     = cosf(phi);
  float s     = sinf(phi);
  float alpha = sqrtf(c * c * a.x * a.x + s * s * a.y * a.y); // Heitz, Formula (80) for wo

  const float g = smith_G1(alpha, wo, wm);

  phi   = atan2f(wi.y, wi.x);
  c     = cosf(phi);
  s     = sinf(phi);
  alpha = sqrtf(c * c * a.x * a.x + s * s * a.y * a.y); // Heitz, Formula (80) for wi.

  return g * smith_G1(alpha, wi, wm);
}


#endif // SHADER_COMMON_H
