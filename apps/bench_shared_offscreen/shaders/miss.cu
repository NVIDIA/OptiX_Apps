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
#include "light_definition.h"
#include "shader_common.h"
#include "system_data.h"

extern "C" __constant__ SystemData sysData;
 
// Not actually a light. Never appears inside the sysLightDefinitions.
extern "C" __global__ void __miss__env_null()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->radiance = make_float3(0.0f);
  thePrd->flags |= FLAG_TERMINATE;
}


extern "C" __global__ void __miss__env_constant()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

#if USE_NEXT_EVENT_ESTIMATION
  // If the last surface intersection was a diffuse which was directly lit with multiple importance sampling,
  // then calculate light emission with multiple importance sampling as well.
  const float weightMIS = (thePrd->flags & FLAG_DIFFUSE) ? powerHeuristic(thePrd->pdf, 0.25f * M_1_PIf) : 1.0f;
  thePrd->radiance = make_float3(weightMIS); // Constant white emission multiplied by MIS weight.
#else
  thePrd->radiance = make_float3(1.0f); // Constant white emission.
#endif

  thePrd->flags |= FLAG_TERMINATE;
}


extern "C" __global__ void __miss__env_sphere()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  const float3 R = thePrd->wi; // theRay.direction;
  // The seam u == 0.0 == 1.0 is in positive z-axis direction.
  // Compensate for the environment rotation done inside the direct lighting.
  const float u     = (atan2f(R.x, -R.z) + M_PIf) * 0.5f * M_1_PIf + sysData.envRotation; // DAR FIXME Use a light.matrix to rotate the environment.
  const float theta = acosf(-R.y);     // theta == 0.0f is south pole, theta == M_PIf is north pole.
  const float v     = theta * M_1_PIf; // Texture is with origin at lower left, v == 0.0f is south pole.

  const float3 emission = make_float3(tex2D<float4>(sysData.envTexture, u, v));

#if USE_NEXT_EVENT_ESTIMATION
  float weightMIS = 1.0f;
  // If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
  // then calculate light emission with multiple importance sampling for this implicit light hit as well.
  if (thePrd->flags & FLAG_DIFFUSE)
  {
    // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
    // and not the Gaussian smoothed one used to actually generate the CDFs.
    const float pdfLight = intensity(emission) / sysData.envIntegral;
    weightMIS = powerHeuristic(thePrd->pdf, pdfLight);
  }
  thePrd->radiance = emission * weightMIS;
#else
  thePrd->radiance = emission;
#endif

  thePrd->flags |= FLAG_TERMINATE;
}
