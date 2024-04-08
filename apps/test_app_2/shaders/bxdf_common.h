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

#ifndef GGX_SMITH_COMMON_H
#define GGX_SMITH_COMMON_H

#include "config.h"

#include "vector_math.h"

// For the updateMaterialStack() function.
#include "per_ray_data.h"
#include "material_definition.h"

// Fresnel equation for an equal mix of polarization.
__forceinline__ __device__ float ior_fresnel(
  const float eta, // refracted / reflected ior
  const float kh)  // cosine of angle between normal/half-vector and direction
{
  float costheta = 1.0f - (1.0f - kh * kh) / (eta * eta);
  
  if (costheta <= 0.0f)
  {
    return 1.0f;
  }

  costheta = sqrtf(costheta); // refracted angle cosine

  const float n1t1 = kh;
  const float n1t2 = costheta;
  const float n2t1 = kh * eta;
  const float n2t2 = costheta * eta;
  const float r_p = (n1t2 - n2t1) / (n1t2 + n2t1);
  const float r_o = (n1t1 - n2t2) / (n1t1 + n2t2);
  
  const float fres = 0.5f * (r_p * r_p + r_o * r_o);

  return clamp(fres, 0.0f, 1.0f);
}


// Compute refraction direction.
__forceinline__ __device__ float3 refract(
  const float3 k,  // direction (pointing from surface)
  const float3 n,  // normal
  const float  b,  // (reflected side IOR) / (transmitted side IOR)
  const float  nk, // dot(n, k)
  bool& tir)       // total internal reflection
{
  const float refraction = b * b * (1.0f - nk * nk);

  tir = (1.0f <= refraction);

  return (tir) ? (n * (nk + nk) - k) : normalize((-k * b + n * (b * nk - sqrtf(1.0f - refraction))));
}


// Check for total internal reflection.
__forceinline__ __device__ bool isTIR(const float2 ior, const float kh)
{
  const float b = ior.x / ior.y;

  return (1.0f < (b * b * (1.0f - kh * kh)));
}


// Compute half vector (convention: pointing to outgoing direction, like shading normal)
__forceinline__ __device__ float3 compute_half_vector(
  const float3 k1,
  const float3 k2,
  const float3 normal,
  const float2 ior,
  const float  nk2,
  const bool   transmission,
  const bool   thinwalled)
{
  float3 h;

  if (transmission)
  {
    if (thinwalled) // No refraction!
    {
      h = k1 + (normal * (nk2 + nk2) + k2); // Use corresponding reflection direction.
    }
    else
    {
      h = k2 * ior.y + k1 * ior.x; // Points into thicker medium.

      if (ior.y > ior.x)
      {
        h *= -1.0f; // Make pointing to outgoing direction's medium.
      }
    }
  }
  else
  {
    h = k1 + k2; // unnormalized half-vector
  }
  
  return normalize(h);
}


// Smith-masking for anisotropic GGX
__forceinline__ __device__ float smith_shadow_mask(
  const float3 k,
  const float2 roughness)
{
  const float ax = roughness.x * k.x;
  const float ay = roughness.y * k.y;

  const float inv_a_2 = (ax * ax + ay * ay) / (k.z * k.z);

  return 2.0f / (1.0f + sqrtf(1.0f + inv_a_2));
}


__forceinline__ __device__ float ggx_smith_shadow_mask(
  float& G1,
  float& G2,
  const float3 k1,
  const float3 k2,
  const float2 roughness)
{
  G1 = smith_shadow_mask(k1, roughness);
  G2 = smith_shadow_mask(k2, roughness);

  return G1 * G2;
}

// Sample visible (Smith-masked) half-vector according to the anisotropic GGX distribution.
// (See Eric Heitz - A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals)
__forceinline__ __device__ float3 hvd_ggx_sample_vndf(
  const float3 k,
  const float2 roughness,
  const float2 xi)
{ 
  // Stretch view.
  const float3 v = normalize(make_float3(k.x * roughness.x,
                                         k.y * roughness.y,
                                         k.z));
  // Orthonormal basis.
  const float3 t1 = (v.z < 0.99999f)
                  ? normalize(cross(v, make_float3(0.0f, 0.0f, 1.0f))) 
                  : make_float3(1.0f, 0.0f, 0.0f); // tangent
  const float3 t2 = cross(t1, v);                  // bitangent

  // Sample point with polar coordinates (r, phi)
  const float a = 1.0f / (1.0f + v.z);

  const float r   = sqrtf(xi.x);
  const float phi = (xi.y < a) ? xi.y / a * M_PIf : M_PIf + (xi.y - a) / (1.0f - a) * M_PIf;
  
  const float p1 = r * cosf(phi);
  const float p2 = r * sinf(phi) * ((xi.y < a) ? 1.0f : v.z);

  // Compute normal.
  float3 h = p1 * t1 + p2 * t2 + sqrtf(fmaxf(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

  // unstretch
  h.x *= roughness.x;
  h.y *= roughness.y;
  h.z  = fmaxf(0.0f, h.z);

  return normalize(h);
}


// Evaluate anisotropic GGX distribution on the non-projected hemisphere
__forceinline__ __device__ float hvd_ggx_eval(
  const float2 invAlpha,
  const float3 h) // == make_float3(dot(tangent, h), dot(bitangent, h), dot(normal, h))
 {
  const float x = h.x * invAlpha.x;
  const float y = h.y * invAlpha.y;
  
  const float aniso = x * x + y * y;

  const float f = aniso + h.z * h.z;

  return M_1_PIf * invAlpha.x * invAlpha.y * h.z / (f * f);
}


__forceinline__ __device__ void updateMaterialStack(PerRayData* prd, const MaterialDefinition& material, const bool isFrontFace)
{
  int idx;

  if (isFrontFace) // Entered a volume. 
  {
    idx = min(prd->idxStack + 1, MATERIAL_STACK_LAST); // Push current medium parameters.

    prd->idxStack = idx;

    prd->stack[idx].absorption_ior  = material.absorption_ior;
    prd->stack[idx].scattering_bias = material.scattering_bias;
  }
  else // if !isFrontFace. Left a volume.
  {
    idx = max(0, prd->idxStack - 1); // Pop current medium parameters.

    prd->idxStack = idx;
  }

  // Update the extinction coefficient sigma_t.
  prd->sigma_t = make_float3(prd->stack[idx].absorption_ior) + // sigma_a +
                 make_float3(prd->stack[idx].scattering_bias); // sigma_s

  prd->walk = 0; // Reset the number of random walk steps taken when crossing any volume boundary.
}

#endif // GGX_SMITH_COMMON_H
