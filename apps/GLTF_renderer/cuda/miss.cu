//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "config.h"

#include <optix.h>

#include "launch_parameters.h"
#include "light_definition.h"
#include "per_ray_data.h"
#include "shader_common.h"
#include "transform.h"
#include "vector_math.h"

extern "C" {
  __constant__ LaunchParameters theLaunchParameters;
}

extern "C" __global__ void __miss__shadow()
{
  optixSetPayload_0(1); // isVisible != 0
}

#if 0 // HACK Implement volume scattering
// Take the step along the volume scattering random walk.
__forceinline__ __device__ void stepVolume(PerRayData* thePrd)
{
  // Calculate the new position at the end of the random walk ray segment.
  thePrd->pos += thePrd->wi * thePrd->distance;

  // Change the throughput along the random walk according to the current extinction and the sampled density.
  const float3 transmittance = expf(thePrd->sigma_t * -thePrd->distance);
  
  const float pdf = dot(thePrd->pdfVolume, thePrd->sigma_t * transmittance);
  
  thePrd->throughput *= make_float3(thePrd->stack[thePrd->idxStack].scattering_bias) * transmittance / pdf;

  // Indicate that the random walk missed.
  thePrd->flags |= FLAG_VOLUME_SCATTERING_MISS;

  // Increment the number of steps done for the random walk
  ++thePrd->walk;
}
#endif // FIXME Implement volume scattering.

// Not actually a light. Never appears inside the sysLightDefinitions.
extern "C" __global__ void __miss__env_null()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  //if (thePrd->flags & FLAG_VOLUME_SCATTERING)
  //{
  //  stepVolume(thePrd);
  //  return; // Continue the random walk.
  //}

  // The null environment adds nothing to the radiance.
  thePrd->typeEvent = BSDF_EVENT_ABSORB;
}


extern "C" __global__ void __miss__env_constant()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  //if (thePrd->flags & FLAG_VOLUME_SCATTERING)
  //{
  //  stepVolume(thePrd);
  //  return; // Continue the random walk.
  //}

  // The environment light is always in the first element.
  float3 emission = theLaunchParameters.lightDefinitions[0].emission; // Constant emission.

  if (theLaunchParameters.directLighting)
  {
    // If the last surface intersection was diffuse or glossy which was directly lit with multiple importance sampling,
    // then calculate implicit light emission with multiple importance sampling as well.
    const float weightMIS = (thePrd->typeEvent & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY))
                          ? balanceHeuristic(thePrd->pdf, 0.25f * M_1_PIf)
                          : 1.0f;

    emission *= weightMIS;
  }
  // Potential ambient occlusion is only applied for environment lights.
  thePrd->radiance += thePrd->throughput * emission * thePrd->occlusion;
  thePrd->typeEvent = BSDF_EVENT_ABSORB;
}


extern "C" __global__ void __miss__env_sphere()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  //if (thePrd->flags & FLAG_VOLUME_SCATTERING)
  //{
  //  stepVolume(thePrd);
  //  return; // Continue the random walk.
  //}

  // The environment light is always in the first element.
  const LightDefinition& light = theLaunchParameters.lightDefinitions[0];

  // Transform the ray.direction from world space to light object space.
  // (This is only using the upper 3x3 matrix which contains a rotation for spherical environment lights.)
  const float3 R = transformVector(light.matrixInv, thePrd->wi);

  // This environment light has the seam u == 0.0f == 1.0f on the negative z-axis.
  // Means the center of the image is on the positive z-axis behind the camera with default settings.
  const float u = (atan2f(-R.x, R.z) + M_PIf) * 0.5f * M_1_PIf;
  // Texture is with origin at lower left, v == 0.0f is south pole.
  const float v = acosf(-R.y) * M_1_PIf;

  float3 emission = make_float3(tex2D<float4>(light.textureEmission, u, v));

  if (theLaunchParameters.directLighting)
  {
    // If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
    // then calculate implicit light emission with multiple importance sampling as well.
    if (thePrd->typeEvent & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY))
    {
      // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
      // and not the Gaussian smoothed one used to actually generate the CDFs.
      const float pdfLight = intensity(emission) * light.invIntegral;
      
      emission *= balanceHeuristic(thePrd->pdf, pdfLight);
    }
  }
  // Potential ambient occlusion is only applied for environment lights.
  thePrd->radiance += thePrd->throughput * emission * light.emission * thePrd->occlusion;
  thePrd->typeEvent = BSDF_EVENT_ABSORB;
}
