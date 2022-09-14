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

#include "system_data.h"
#include "per_ray_data.h"
#include "shader_common.h"
#include "random_number_generators.h"


extern "C" __constant__ SystemData sysData;


__forceinline__ __device__ float3 integrator(PerRayData& prd)
{
  // This renderer supports nested volumes. Four levels is plenty enough for most cases.
  // The absorption coefficient and IOR of the volume the ray is currently inside.
  float4 absorptionStack[MATERIAL_STACK_SIZE]; // .xyz == absorptionCoefficient (sigma_a), .w == index of refraction
  
  int stackIdx = MATERIAL_STACK_EMPTY; // Start with empty nested materials stack.

  // Russian Roulette path termination after a specified number of bounces needs the current depth.
  int depth = 0; // Path segment index. Primary ray is 0. 

  float3 radiance   = make_float3(0.0f); // Start with black.
  float3 throughput = make_float3(1.0f); // The throughput for the next radiance, starts with 1.0f.

  // Assumes that the primary ray starts in vacuum.
  prd.absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // No absorption and IOR == 1.0f
  prd.sigma_t        = make_float3(0.0f);                   // No extinction.
  prd.flags          = 0;

  while (depth < sysData.pathLengths.y)
  {
    prd.wo        = -prd.wi;            // Direction to observer.
    prd.ior       = make_float2(1.0f);  // Reset the volume IORs.
    prd.distance  = RT_DEFAULT_MAX;     // Shoot the next ray with maximum length.
    prd.flags    &= FLAG_CLEAR_MASK;    // Clear all non-persistent flags. In this demo only the last diffuse surface interaction stays.

    // Handle volume absorption of nested materials.
    if (MATERIAL_STACK_FIRST <= stackIdx) // Inside a volume?
    {
      prd.flags  |= FLAG_VOLUME;                            // Indicate that we're inside a volume. => At least absorption calculation needs to happen.
      prd.sigma_t = make_float3(absorptionStack[stackIdx]); // There is only volume absorption in this demo, no volume scattering.
      prd.ior.x   = absorptionStack[stackIdx].w;            // The IOR of the volume we're inside. Needed for eta calculations in transparent materials.
      if (MATERIAL_STACK_FIRST <= stackIdx - 1)
      {
        prd.ior.y = absorptionStack[stackIdx - 1].w; // The IOR of the surrounding volume. Needed when potentially leaving a volume to calculate eta in transparent materials.
      }
    }

    // Put payload pointer into two unsigned integers. Actually const, but that's not what optixTrace() expects.
    uint2 payload = splitPointer(&prd);

    // Note that the primary rays (or volume scattering miss cases) wouldn't normally offset the ray t_min by sysSceneEpsilon. Keep it simple here.
    optixTrace(sysData.topObject,
               prd.pos, prd.wi, // origin, direction
               sysData.sceneEpsilon, prd.distance, 0.0f, // tmin, tmax, time
               OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_NONE, 
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    // This renderer supports nested volumes.
    if (prd.flags & FLAG_VOLUME) // We're inside a volume?
    {
      // The transmittance along the current path segment inside a volume needs to attenuate 
      // the ray throughput with the extinction before it modulates the radiance of the hitpoint.
      throughput *= expf(-prd.distance * prd.sigma_t);
    }

    radiance += throughput * prd.radiance;

    // Path termination by miss shader or sample() routines.
    // If terminate is true, f_over_pdf and pdf might be undefined.
    if ((prd.flags & FLAG_TERMINATE) || prd.pdf <= 0.0f || isNull(prd.f_over_pdf))
    {
      break;
    }

    // PERF f_over_pdf already contains the proper throughput adjustment for diffuse materials: f * (fabsf(dot(prd.wi, state.normal)) / prd.pdf);
    throughput *= prd.f_over_pdf;

    // Unbiased Russian Roulette path termination.
    if (sysData.pathLengths.x <= depth) // Start termination after a minimum number of bounces.
    {
      const float probability = fmaxf(throughput); // DEBUG Other options: // intensity(throughput); // fminf(0.5f, intensity(throughput));
      if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
      {
        break;
      }
      throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
    }

    // Adjust the material volume stack if the geometry is not thin-walled but a border between two volumes and
    // the outgoing ray direction was a transmission.
    if ((prd.flags & (FLAG_THINWALLED | FLAG_TRANSMISSION)) == FLAG_TRANSMISSION)
    {
      // Transmission.
      if (prd.flags & FLAG_FRONTFACE) // Entered a new volume?
      {
        // Push the entered material's volume properties onto the volume stack.
        //rtAssert((stackIdx < MATERIAL_STACK_LAST), 1); // Overflow?
        stackIdx = min(stackIdx + 1, MATERIAL_STACK_LAST);
        absorptionStack[stackIdx] = prd.absorption_ior;
      }
      else // Exited the current volume?
      {
        // Pop the top of stack material volume.
        // This assert fires and is intended because I tuned the frontface checks so that there are more exits than enters at silhouettes.
        //rtAssert((MATERIAL_STACK_EMPTY < stackIdx), 0); // Underflow?
        stackIdx = max(stackIdx - 1, MATERIAL_STACK_EMPTY);
      }
    }

    ++depth; // Next path segment.
  }
  
  return radiance;
}


__forceinline__ __device__ unsigned int distribute(const uint2 launchIndex)
{
  // First calculate block coordinates of this launch index.
  // That is the launch index divided by the tile dimensions. (No operator>>() on vectors?)
  const unsigned int xBlock = launchIndex.x >> sysData.tileShift.x;
  const unsigned int yBlock = launchIndex.y >> sysData.tileShift.y;
  
  // Each device needs to start at a different column and each row should start with a different device.
  const unsigned int xTile = xBlock * sysData.deviceCount + ((sysData.deviceIndex + yBlock) % sysData.deviceCount);

  // The horizontal pixel coordinate is: tile coordinate * tile width + launch index % tile width.
  return xTile * sysData.tileSize.x + (launchIndex.x & (sysData.tileSize.x - 1)); // tileSize needs to be power-of-two for this modulo operation.
}


extern "C" __global__ void __raygen__path_tracer_local_copy()
{
#if USE_TIME_VIEW
  clock_t clockBegin = clock();
#endif

  const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());
  
  unsigned int launchColumn = theLaunchIndex.x;

  if (1 < sysData.deviceCount) // Multi-GPU distribution required?
  {
    launchColumn = distribute(theLaunchIndex); // Calculate mapping from launch index to pixel index.
    if (sysData.resolution.x <= launchColumn)  // Check if the launchColumn is outside the resolution.
    {
      return;
    }
  }

  PerRayData prd;

  const uint2 theLaunchDim = make_uint2(optixGetLaunchDimensions()); // For multi-GPU tiling this is (resolution + deviceCount - 1) / deviceCount.

  // Initialize the random number generator seed from the linear pixel index and the iteration index.
  prd.seed = tea<4>(theLaunchIndex.y * theLaunchDim.x + launchColumn, sysData.iterationIndex); // PERF This template really generates a lot of instructions.

  // Decoupling the pixel coordinates from the screen size will allow for partial rendering algorithms.
  // Resolution is the actual full rendering resolution and for the single GPU strategy, theLaunchDim == resolution.
  const float2 screen = make_float2(sysData.resolution); // == theLaunchDim for rendering strategy RS_SINGLE_GPU.
  const float2 pixel  = make_float2(launchColumn, theLaunchIndex.y);
  const float2 sample = rng2(prd.seed);

  // Lens shaders
  const LensRay ray = optixDirectCall<LensRay, const float2, const float2, const float2>(sysData.typeLens, screen, pixel, sample);

  prd.pos = ray.org;
  prd.wi  = ray.dir;

  float3 radiance = integrator(prd);

#if USE_DEBUG_EXCEPTIONS
  // DEBUG Highlight numerical errors.
  if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
  {
    radiance = make_float3(1000000.0f, 0.0f, 0.0f); // super red
  }
  else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z))
  {
    radiance = make_float3(0.0f, 1000000.0f, 0.0f); // super green
  }
  else if (radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f)
  {
    radiance = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
  }
#else
  // NaN values will never go away. Filter them out before they can arrive in the output buffer.
  // This only has an effect if the debug coloring above is off!
  if (!(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z)))
#endif
  {
    // The texelBuffer is a CUdeviceptr to allow different formats.
    float4* buffer = reinterpret_cast<float4*>(sysData.texelBuffer); // This is a per device launch sized buffer in this renderer strategy.

    // This renderer write the results into individual launch sized local buffers and composites them in a separate native CUDA kernel.
    const unsigned int index = theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x;

#if USE_TIME_VIEW
    clock_t clockEnd = clock(); 
    const float alpha = (clockEnd - clockBegin) * sysData.clockScale;

    float4 result = make_float4(radiance, alpha);

    if (0 < sysData.iterationIndex)
    {
      const float4 dst = buffer[index]; // RGBA32F
      result = lerp(dst, result, 1.0f / float(sysData.iterationIndex + 1)); // Accumulate the alpha as well.
    }
    buffer[index] = result;
#else
    if (0 < sysData.iterationIndex)
    {
      const float4 dst = buffer[index]; // RGBA32F
      radiance = lerp(make_float3(dst), radiance, 1.0f / float(sysData.iterationIndex + 1)); // Only accumulate the radiance, alpha stays 1.0f.
    }
    buffer[index] = make_float4(radiance, 1.0f);
#endif
  }
}


extern "C" __global__ void __raygen__path_tracer()
{
#if USE_TIME_VIEW
  clock_t clockBegin = clock();
#endif

  const uint2 theLaunchDim   = make_uint2(optixGetLaunchDimensions()); // For multi-GPU tiling this is (resolution + deviceCount - 1) / deviceCount.
  const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());

  PerRayData prd;

  // Initialize the random number generator seed from the linear pixel index and the iteration index.
  prd.seed = tea<4>( theLaunchDim.x * theLaunchIndex.y + theLaunchIndex.x, sysData.iterationIndex); // PERF This template really generates a lot of instructions.

  // Decoupling the pixel coordinates from the screen size will allow for partial rendering algorithms.
  // Resolution is the actual full rendering resolution and for the single GPU strategy, theLaunchDim == resolution.
  const float2 screen = make_float2(sysData.resolution); // == theLaunchDim for rendering strategy RS_SINGLE_GPU.
  const float2 pixel  = make_float2(theLaunchIndex);
  const float2 sample = rng2(prd.seed);

  // Lens shaders
  const LensRay ray = optixDirectCall<LensRay, const float2, const float2, const float2>(sysData.typeLens, screen, pixel, sample);

  prd.pos = ray.org;
  prd.wi  = ray.dir;

  float3 radiance = integrator(prd);

#if USE_DEBUG_EXCEPTIONS
  // DEBUG Highlight numerical errors.
  if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
  {
    radiance = make_float3(1000000.0f, 0.0f, 0.0f); // super red
  }
  else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z))
  {
    radiance = make_float3(0.0f, 1000000.0f, 0.0f); // super green
  }
  else if (radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f)
  {
    radiance = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
  }
#else
  // NaN values will never go away. Filter them out before they can arrive in the output buffer.
  // This only has an effect if the debug coloring above is off!
  if (!(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z)))
#endif
  {
    float4* buffer = reinterpret_cast<float4*>(sysData.outputBuffer);

    const unsigned int index = theLaunchDim.x * theLaunchIndex.y + theLaunchIndex.x;

#if USE_TIME_VIEW
    clock_t clockEnd = clock(); 
    const float alpha = (clockEnd - clockBegin) * sysData.clockScale;

    float4 result = make_float4(radiance, alpha);

    if (0 < sysData.iterationIndex)
    {
      const float4 dst = buffer[index]; // RGBA32F
      result = lerp(dst, result, 1.0f / float(sysData.iterationIndex + 1)); // Accumulate the alpha as well.
    }
    buffer[index] = result;
#else
    if (0 < sysData.iterationIndex)
    {
      const float4 dst = buffer[index]; // RGBA32F
      radiance = lerp(make_float3(dst), radiance, 1.0f / float(sysData.iterationIndex + 1)); // Only accumulate the radiance, alpha stays 1.0f.
    }
    buffer[index] = make_float4(radiance, 1.0f);
#endif
  }
}

