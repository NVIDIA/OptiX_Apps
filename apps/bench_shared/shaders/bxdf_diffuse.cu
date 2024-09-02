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

#include "config.h"

#include <optix.h>

#include "per_ray_data.h"
#include "material_definition.h"
#include "shader_common.h"
#include "random_number_generators.h"


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

// BRDF Diffuse (Lambert)

extern "C" __device__ void __direct_callable__sample_brdf_diffuse(const MaterialDefinition& material, const State& state, PerRayData* prd)
{
  // Cosine weighted hemisphere sampling for Lambert material.
  unitSquareToCosineHemisphere(rng2(prd->seed), state.normal, prd->wi, prd->pdf);

  if (prd->pdf <= 0.0f || dot(prd->wi, state.normalGeo) <= 0.0f)
  {
    prd->flags |= FLAG_TERMINATE;
    return;
  }

  // This would be the universal implementation for an arbitrary sampling of a diffuse surface.
  // prd->f_over_pdf = state.albedo * (M_1_PIf * fabsf(dot(prd->wi, state.normal)) / prd->pdf); 
  
  // PERF Since the cosine-weighted hemisphere distribution is a perfect importance-sampling of the Lambert material,
  // the whole term ((M_1_PIf * fabsf(dot(prd->wi, state.normal)) / prd->pdf) is always 1.0f here!
  prd->f_over_pdf = state.albedo;

  prd->flags |= FLAG_DIFFUSE; // Direct lighting will be done with multiple importance sampling.
}

// The parameter wiL is the lightSample.direction (direct lighting), not the next ray segment's direction prd.wi (indirect lighting).
extern "C" __device__ float4 __direct_callable__eval_brdf_diffuse(const MaterialDefinition& material, const State& state, const PerRayData* const prd, const float3 wiL)
{
  const float3 f   = state.albedo * M_1_PIf;
  const float  pdf = fmaxf(0.0f, dot(wiL, state.normal) * M_1_PIf);

  return make_float4(f, pdf);
}

