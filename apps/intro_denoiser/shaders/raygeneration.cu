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

#include "app_config.h"

#include <optix.h>

#include "system_parameter.h"
#include "per_ray_data.h"
#include "shader_common.h"
#include "random_number_generators.h"
#include "half_common.h"

extern "C" __constant__ SystemParameter sysParameter;

extern "C" __global__ void __raygen__pathtracer()
{
  PerRayData prd;

  // This assumes that the launch dimensions are matching the size of the output buffer.
  const uint3 theLaunchDim   = optixGetLaunchDimensions();
  const uint3 theLaunchIndex = optixGetLaunchIndex();

  // Initialize the random number generator seed from the linear pixel index and the iteration index.
  prd.seed = tea<4>(theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x, sysParameter.iterationIndex);

  // Decoupling the pixel coordinates from the screen size will allow for partial rendering algorithms.
  // In this case theLaunchIndex is the pixel coordinate and theLaunchDim is sysOutputBuffer.size().
  const float2 screen = make_float2(theLaunchDim);
  const float2 pixel  = make_float2(theLaunchIndex);
  const float2 sample = rng2(prd.seed);

  // Lens shaders
  optixDirectCall<void, const float2, const float2, const float2, float3&, float3&>(sysParameter.cameraType, screen, pixel, sample, prd.pos, prd.wi);

  // This renderer supports nested volumes. Four levels is plenty enough for most cases.
  // The absorption coefficient and IOR of the volume the ray is currently inside.
  float4 absorptionStack[MATERIAL_STACK_SIZE]; // .xyz == absorptionCoefficient (sigma_a), .w == index of refraction
  
  float3 radiance   = make_float3(0.0f); // Start with black.
  float3 throughput = make_float3(1.0f); // The throughput for the next radiance, starts with 1.0f.

#if USE_DENOISER_ALBEDO
  float3 albedo = make_float3(0.0f); // Start with black.
#endif
#if USE_DENOISER_NORMAL
  float3 normal = make_float3(0.0f); // Start with null vector.
  prd.normal    = make_float3(0.0f); // Start with null vector. Important if nothing is hit!
#endif
  
  int stackIdx = MATERIAL_STACK_EMPTY; // Start with empty nested materials stack.

  // Russian Roulette path termination after a specified number of bounces needs the current depth.
  int depth = 0; // Path segment index. Primary ray is 0. 

  prd.absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // Assume primary ray starts in vacuum.
  prd.flags          = 0;

  while (depth < sysParameter.pathLengths.y)
  {
    prd.wo        = -prd.wi;            // Direction to observer.
    prd.ior       = make_float2(1.0f);  // Reset the volume IORs.
    prd.distance  = RT_DEFAULT_MAX;     // Shoot the next ray with maximum length.
    prd.flags    &= FLAG_CLEAR_MASK;    // Clear all non-persistent flags. In this demo only the last diffuse surface interaction stays.

    // Handle volume absorption of nested materials.
    if (MATERIAL_STACK_FIRST <= stackIdx) // Inside a volume?
    {
      prd.flags     |= FLAG_VOLUME;                            // Indicate that we're inside a volume. => At least absorption calculation needs to happen.
      prd.extinction = make_float3(absorptionStack[stackIdx]); // There is only volume absorption in this demo, no volume scattering.
      prd.ior.x      = absorptionStack[stackIdx].w;            // The IOR of the volume we're inside. Needed for eta calculations in transparent materials.
      if (MATERIAL_STACK_FIRST <= stackIdx - 1)
      {
        prd.ior.y = absorptionStack[stackIdx - 1].w; // The IOR of the surrounding volume. Needed when potentially leaving a volume to calculate eta in transparent materials.
      }
    }

    // Note that the primary rays (or volume scattering miss cases) wouldn't normally offset the ray t_min by sysSceneEpsilon. Keep it simple here.

    // Put radiance payload pointer into two unsigned integers.
    uint2 payload = splitPointer(&prd);

    optixTrace(sysParameter.topObject,
               prd.pos, prd.wi, // origin, direction
               sysParameter.sceneEpsilon, prd.distance, 0.0f, // tmin, tmax, time
               OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_NONE, 
               RAYTYPE_RADIANCE, NUM_RAYTYPES, RAYTYPE_RADIANCE,
               payload.x, payload.y);

    // This renderer supports nested volumes.
    if (prd.flags & FLAG_VOLUME)
    {
      // We're inside a volume. Calculate the extinction along the current path segment in any case.
      // The transmittance along the current path segment inside a volume needs to attenuate the ray throughput with the extinction
      // before it modulates the radiance of the hitpoint.
      throughput *= expf(-prd.distance * prd.extinction);
    }

    radiance += throughput * prd.radiance;

#if USE_DENOISER_ALBEDO
    // In physical terms, the albedo is a single color value approximating the ratio of radiant exitance to the irradiance under uniform lighting.
    // The albedo value can be approximated for simple materials by using the diffuse color of the first hit,
    // or for layered materials by using a weighted sum of the individual BRDFs albedo values.
    // For some objects such as perfect mirrors, the quality of the result might be improved by using the albedo value of a subsequent hit instead.

    // When no albedo has been written before and the hit was diffuse or a light, write the albedo.
    // DAR This makes glass materials and motion blur on specular surfaces in the demo a little noisier,
    // but should definitely be used with high frequency textures behind transparent or around reflective materials.
    if (!(prd.flags & FLAG_ALBEDO) && (prd.flags & (FLAG_DIFFUSE | FLAG_LIGHT)))
    {
      // The albedo buffer should contain the surface appearance under uniform lighting in linear color space in the range [0.0f, 1.0f].
      // Clamp the final albedo result to that range here, because it captured the radiance when hitting lights either directly or via specular events.
      albedo = clamp(throughput * prd.albedo, 0.0f, 1.0f);

      prd.flags |= FLAG_ALBEDO; // This flag is persistent along the path and prevents that the albedo is written more than once.
    }
#endif 

#if USE_DENOISER_NORMAL
    // The normal buffer is expected to contain the surface normals of the primary hit in camera space.
    // The camera space is assumed to be right handed such that the camera is looking down
    // the negative z-axis, and the up direction is along the y-axis. The x-axis points to the right.
    if (depth == 0 && (prd.flags & FLAG_HIT)) // Miss event keeps the null-vector.
    {
      // Note the input sysCameraU|V|W vectors are unnormalized and build a left-handed coordinate system.
      // They are also not necessarily perpendicular to each other, because the generic pinhole camera system
      // would allow sheared projections, but that's not used on these OptiX introduction examples. 
      
      // Project the world space normal into camera space.
      // Using the normalized camera basis vectors here as camera space to get consistent results
      // independently of the UVW vector lengths.
      // The end result looks like a normal map without scale and bias.
      // Normals pointing at the camera position will be blue.
      normal = make_float3( dot(prd.normal, normalize(sysParameter.cameraU)), 
                            dot(prd.normal, normalize(sysParameter.cameraV)), 
                           -dot(prd.normal, normalize(sysParameter.cameraW))); // Negative W to make it right-handed.
    }
#endif

    // Path termination by miss shader or sample() routines.
    // If terminate is true, f_over_pdf and pdf might be undefined.
    if ((prd.flags & FLAG_TERMINATE) || prd.pdf <= 0.0f || isNull(prd.f_over_pdf))
    {
      break;
    }

    // PERF f_over_pdf already contains the proper throughput adjustment for diffuse materials: f * (fabsf(optix::dot(prd.wi, state.normal)) / prd.pdf);
    throughput *= prd.f_over_pdf;

    // Unbiased Russian Roulette path termination.
    if (sysParameter.pathLengths.x <= depth) // Start termination after a minimum number of bounces.
    {
      const float probability = fmaxf(throughput); // Other options: // intensity(throughput); // fminf(0.5f, intensity(throughput));
      if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
      {
        break;
      }
      throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
    }

    // Adjust the material volume stack if the geometry is not thin-walled but a border between two volumes 
    // and the outgoing ray direction was a transmission.
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
    const unsigned int index = theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x;

#if USE_FP32_OUTPUT

    float4* bufferRGBA   = reinterpret_cast<float4*>(sysParameter.outputBuffer);
#if USE_DENOISER_ALBEDO
    float4* bufferAlbedo = reinterpret_cast<float4*>(sysParameter.albedoBuffer);
#endif
#if USE_DENOISER_NORMAL
    float4* bufferNormal = reinterpret_cast<float4*>(sysParameter.normalBuffer);
#endif
    if (0 < sysParameter.iterationIndex)
    {
      const float t = 1.0f / float(sysParameter.iterationIndex + 1);

      radiance = lerp(make_float3(bufferRGBA[index]),   radiance, t);
#if USE_DENOISER_ALBEDO
      albedo   = lerp(make_float3(bufferAlbedo[index]), albedo,   t);
#endif
#if USE_DENOISER_NORMAL
      normal   = lerp(make_float3(bufferNormal[index]), normal,   t);
      if (isNotNull(normal))
      {
        normal = normalize(normal);
      }
#endif
    }
    // sysIterationIndex 0 will fill the buffer.
    // If this isn't done separately, the result of the lerp() above is undefined, e.g. dst could be NaN.
    bufferRGBA[index]   = make_float4(radiance, 1.0f);
#if USE_DENOISER_ALBEDO
    bufferAlbedo[index] = make_float4(albedo,   1.0f);
#endif
#if USE_DENOISER_NORMAL
    bufferNormal[index] = make_float4(normal,   0.0f);
#endif

#else

    Half4* bufferRGBA   = reinterpret_cast<Half4*>(sysParameter.outputBuffer);
#if USE_DENOISER_ALBEDO
    Half4* bufferAlbedo = reinterpret_cast<Half4*>(sysParameter.albedoBuffer);
#endif
#if USE_DENOISER_NORMAL
    Half4* bufferNormal = reinterpret_cast<Half4*>(sysParameter.normalBuffer);
#endif
    
    if (0 < sysParameter.iterationIndex)
    {
      const float t = 1.0f / float(sysParameter.iterationIndex + 1);
      
      Half4 dst = bufferRGBA[index]; // RGBA16F

      radiance.x = lerp(__half2float(dst.x), radiance.x, t);
      radiance.y = lerp(__half2float(dst.y), radiance.y, t);
      radiance.z = lerp(__half2float(dst.z), radiance.z, t);

#if USE_DENOISER_ALBEDO
      dst = bufferAlbedo[index]; // RGBA16F

      albedo.x = lerp(__half2float(dst.x), albedo.x, t);
      albedo.y = lerp(__half2float(dst.y), albedo.y, t);
      albedo.z = lerp(__half2float(dst.z), albedo.z, t);
#endif
#if USE_DENOISER_NORMAL
      dst = bufferNormal[index]; // RGBA16F

      normal.x = lerp(__half2float(dst.x), normal.x, t);
      normal.y = lerp(__half2float(dst.y), normal.y, t);
      normal.z = lerp(__half2float(dst.z), normal.z, t);
      if (isNotNull(normal))
      {
        normal = normalize(normal);
      }
#endif
    }
    // sysIterationIndex 0 will fill the buffer.
    // If this isn't done separately, the result of the lerp() above is undefined, e.g. dst could be NaN.
    bufferRGBA[index]   = make_Half4(radiance, 1.0f);
#if USE_DENOISER_ALBEDO
    bufferAlbedo[index] = make_Half4(albedo,   1.0f);
#endif
#if USE_DENOISER_NORMAL
    bufferNormal[index] = make_Half4(normal,   0.0f);
#endif

#endif
  }
}
