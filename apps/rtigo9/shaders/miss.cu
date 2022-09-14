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
#include "transform.h"

extern "C" __constant__ SystemData sysData;
 
// Not actually a light. Never appears inside the sysLightDefinitions.
extern "C" __global__ void __miss__env_null()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->radiance = make_float3(0.0f);
  thePrd->flags   |= FLAG_TERMINATE;
}


extern "C" __global__ void __miss__env_constant()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  // The environment light is always in the first element.
  float3 emission = sysData.lightDefinitions[0].emission; // Constant emission.

  if (sysData.directLighting)
  {
    // If the last surface intersection was a diffuse which was directly lit with multiple importance sampling,
    // then calculate light emission with multiple importance sampling as well.
    const float weightMIS = (thePrd->flags & FLAG_DIFFUSE) ? balanceHeuristic(thePrd->pdf, 0.25f * M_1_PIf) : 1.0f;

    emission *= weightMIS;
  }

  thePrd->radiance = emission; 
  thePrd->flags   |= FLAG_TERMINATE;
}


extern "C" __global__ void __miss__env_sphere()
{
  // The environment light is always in the first element.
  const LightDefinition& light = sysData.lightDefinitions[0];

  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
  
  const float3 R = transformVector(light.oriInv, thePrd->wi); // Transform the ray.direction from world space to light object space.

  // All lights shine down the positive z-axis in this renderer.
  const float u = (atan2f(-R.x, R.z) + M_PIf) * 0.5f * M_1_PIf;
  const float v = acosf(-R.y) * M_1_PIf; // Texture is with origin at lower left, v == 0.0f is south pole.

  float3 emission = make_float3(tex2D<float4>(light.textureEmission, u, v));

  if (sysData.directLighting)
  {
    // If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
    // then calculate light emission with multiple importance sampling for this implicit light hit as well.
    if (thePrd->flags & FLAG_DIFFUSE)
    {
      // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
      // and not the Gaussian smoothed one used to actually generate the CDFs.
      const float pdfLight = intensity(emission) / light.integral;
      
      emission *= balanceHeuristic(thePrd->pdf, pdfLight);
    }
  }
  
  thePrd->radiance = emission * light.emission;
  thePrd->flags   |= FLAG_TERMINATE;
}
