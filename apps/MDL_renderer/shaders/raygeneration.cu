/* 
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
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


__forceinline__ __device__ float3 safe_div(const float3& a, const float3& b)
{
  const float x = (b.x != 0.0f) ? a.x / b.x : 0.0f;
  const float y = (b.y != 0.0f) ? a.y / b.y : 0.0f;
  const float z = (b.z != 0.0f) ? a.z / b.z : 0.0f;

  return make_float3(x, y, z);
}

__forceinline__ __device__ float sampleDensity(const float3& albedo,
                                               const float3& throughput,
                                               const float3& sigma_t,
                                               const float   u,
                                               float3&       pdf)
{
  const float3 weights = throughput * albedo;

  const float sum = weights.x + weights.y + weights.z;
  
  pdf = (0.0f < sum) ? weights / sum : make_float3(1.0f / 3.0f);

  if (u < pdf.x)
  {
    return sigma_t.x;
  }
  if (u < pdf.x + pdf.y)
  {
    return sigma_t.y;
  }
  return sigma_t.z;
}

// Determine scatter reflection direction with Henyey-Greenstein phase function.
__forceinline__ __device__ void sampleHenyeyGreenstein(const float2 xi, const float g, float3& dir)
{
  float cost;

  // PBRT v3: Chapter 15.2.3
  if (fabsf(g) < 1e-3f) // Isotropic.
  {
    cost = 1.0f - 2.0f * xi.x;
  }
  else
  {
    const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi.x);
    cost = (1.0f + g * g - s * s) / (2.0f * g);
  }

  float sint = 1.0f - cost * cost;
  sint = (0.0f < sint) ? sqrtf(sint) : 0.0f;
 
  const float phi = 2.0f * M_PIf * xi.y;

  // This vector is oriented in its own local coordinate system:
  const float3 d = make_float3(cosf(phi) * sint, sinf(phi) * sint, cost); 

  // Align the vector with the incoming direction.
  const TBN tbn(dir); // Just some ortho-normal basis along dir as z-axis.
  
  dir = tbn.transformToWorld(d);
}


__forceinline__ __device__ float3 integrator(PerRayData& prd)
{
  // The integrator starts with black radiance and full path throughput.
  prd.radiance   = make_float3(0.0f);
  prd.pdf        = 0.0f;
  prd.throughput = make_float3(1.0f);
  prd.sigma_t    = make_float3(0.0f); // Extinction coefficient: sigma_a + sigma_s.
  prd.walk       = 0;                 // Number of random walk steps taken through volume scattering. 
  prd.eventType  = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
  // Nested material handling. 
  prd.idxStack   = 0;
  // Small stack of four entries of which the first is vacuum.
  prd.stack[0].ior     = make_float3(1.0f); // No effective IOR.
  prd.stack[0].sigma_a = make_float3(0.0f); // No volume absorption.
  prd.stack[0].sigma_s = make_float3(0.0f); // No volume scattering.
  prd.stack[0].bias    = 0.0f;              // Isotropic volume scattering.

  // Put payload pointer into two unsigned integers. Actually const, but that's not what optixTrace() expects.
  uint2 payload = splitPointer(&prd);

  // Russian Roulette path termination after a specified number of bounces needs the current depth.
  int depth = 0; // Path segment index. Primary ray is depth == 0. 

  while (depth < sysData.pathLengths.y)
  {
    // Hit condition needs to offset the next ray. Camera position and volume scattering miss events don't.
    const float epsilon = (prd.flags & FLAG_HIT) ? sysData.sceneEpsilon : 0.0f;

    prd.wo       = -prd.wi;        // Direction to observer.
    prd.distance = RT_DEFAULT_MAX; // Shoot the next ray with maximum length.
    prd.flags    = 0;

    // Special cases for volume scattering!
    if (0 < prd.idxStack) // Inside a volume?
    {
      // Note that this only supports homogeneous volumes so far! 
      // No change in sigma_s along the random walk here.
      const float3 sigma_s = prd.stack[prd.idxStack].sigma_s;

      if (isNotNull(sigma_s)) // We're inside a volume and it has volume scattering?
      {
        // Indicate that we're inside a random walk. This changes the behavior of the miss programs.
        prd.flags |= FLAG_VOLUME_SCATTERING;

        // Random walk through scattering volume, sampling the distance.
        // Note that the entry and exit of the volume is done according to the BSDF sampling.
        // Means glass with volume scattering will still do the proper refractions.
        // When the number of random walk steps has been exceeded, the next ray is shot with distance RT_DEFAULT_MAX
        // to hit something. If that results in a transmission the scattering volume is left.
        // If not, this continues until the maximum path length has been exceeded.
        if (prd.walk < sysData.walkLength)
        {
          const float3 albedo = safe_div(sigma_s, prd.sigma_t);
          const float2 xi     = rng2(prd.seed);
          
          const float s = sampleDensity(albedo, prd.throughput, prd.sigma_t, xi.x, prd.pdfVolume);

          // Prevent logf(0.0f) by sampling the inverse range (0.0f, 1.0f].
          prd.distance = -logf(1.0f - xi.y) / s;
        }
      }
    }

    // Note that the primary rays (or volume scattering miss cases) wouldn't normally offset the ray t_min by sysSceneEpsilon. Keep it simple here.
    optixTrace(sysData.topObject,
               prd.pos, prd.wi, // origin, direction
               epsilon, prd.distance, 0.0f, // tmin, tmax, time
               OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_NONE, 
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    // Path termination by miss shader or sample() routines.
    if ((prd.eventType == mi::neuraylib::BSDF_EVENT_ABSORB) || isNull(prd.throughput))
    {
      break;
    } 

    // Unbiased Russian Roulette path termination.
    if (sysData.pathLengths.x <= depth) // Start termination after a minimum number of bounces.
    {
      const float probability = fmaxf(prd.throughput);

      if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
      {
        break;
      }

      prd.throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
    }

    // We're inside a volume and the scatter ray missed.
    if (prd.flags & FLAG_VOLUME_SCATTERING_MISS) // This implies FLAG_VOLUME_SCATTERING.
    {
      // Random walk through scattering volume, sampling the direction according to the phase function.
      sampleHenyeyGreenstein(rng2(prd.seed), prd.stack[prd.idxStack].bias, prd.wi);
    }

    ++depth; // Next path segment.
  }
  
  return prd.radiance;
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

