//
// Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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
#include "per_ray_data.h"
#include "random.h"
#include "shader_common.h"
#include "vector_math.h"


extern "C" {
  __constant__ LaunchParameters theLaunchParameters;
}

#if 0 // FIXME Volume scattering is not implemented. No glTF extension for that.
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


// Determine Henyey-Greenstein phase function cos(theta) of scattering direction
__forceinline__ __device__ float sampleHenyeyGreensteinCos(const float xi, const float g)
{
  // PBRT v3: Chapter 15.2.3
  if (fabsf(g) < 1e-3f) // Isotropic.
  {
    return 1.0f - 2.0f * xi;
  }

  const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi);
  return (1.0f + g * g - s * s) / (2.0f * g);
}


// Determine scatter reflection direction with Henyey-Greenstein phase function.
__forceinline__ __device__ void sampleVolumeScattering(const float2 xi, const float g, float3& dir)
{
  const float cost = sampleHenyeyGreensteinCos(xi.x, g);
  
  float sint = 1.0f - cost * cost;
  sint = (0.0f < sint) ? sqrtf(sint) : 0.0f;
 
  const float phi = 2.0f * M_PIf * xi.y;

  // This vector is oriented in its own local coordinate system:
  const float3 d = make_float3(cosf(phi) * sint, sinf(phi) * sint, cost); 

  // Align the vector with the incoming direction.
  const TBN tbn(dir); // Just some ortho-normal basis along dir as z-axis.
  
  dir = tbn.transformToWorld(d);
}
#endif // FIXME Not implementing volume scattering.


// Called by raygen.
// @return radiance
__forceinline__ __device__ float3 integrator(PerRayData& prd)
{
  // The integrator starts with black radiance and full path throughput.
  prd.radiance   = make_float3(0.0f);
  prd.pdf        = 0.0f;
  prd.throughput = make_float3(1.0f);
  prd.occlusion  = 1.0f;
  prd.sigma_t    = make_float3(0.0f); // Extinction coefficient: sigma_a + sigma_s.
  //prd.walk       = 0;                 // Number of random walk steps taken through volume scattering. 
  prd.typeEvent  = BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
  
  // Nested material handling. 
  prd.idxStack   = 0;
  // Small stack of four entries of which the first is vacuum.
  prd.stack[0].absorption_ior  = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // No volume absorption, no effective IOR.
  //prd.stack[0].scattering_bias = make_float4(0.0f);  // No volume scattering, isotropic volume scattering.

  // Put payload pointer into two unsigned integers. Actually const, but that's not what optixTrace() expects.
  uint2 payload = splitPointer(&prd);

  // Russian Roulette path termination after a specified number of bounces needs the current depth.
  int depth = 0; // Path segment index. Primary ray is depth == 0. 

  while (depth < theLaunchParameters.pathLengths.y)
  {
    // Self-intersection avoidance:
    // Offset the ray t_min value by sysData.sceneEpsilon when a geometric primitive was hit by the previous ray.
    // Primary rays and volume scattering miss events will not offset the ray t_min.
    const float epsilon = (prd.flags & FLAG_HIT) ? theLaunchParameters.sceneEpsilon : 0.0f;

    prd.wo       = -prd.wi;        // Direction to observer.
    prd.distance = RT_DEFAULT_MAX; // Shoot the next ray with maximum length.
    prd.flags    = 0;

    // FIXME Not implementing volume scattering.
    // Special cases for volume scattering!
    //if (0 < prd.idxStack) // Inside a volume?
    //{
    //  // Note that this only supports homogeneous volumes so far! 
    //  // No change in sigma_s along the random walk here.
    //  const float3 sigma_s = make_float3(prd.stack[prd.idxStack].scattering_bias);

    //  if (isNotNull(sigma_s)) // We're inside a volume and it has volume scattering?
    //  {
    //    // Indicate that we're inside a random walk. This changes the behavior of the radiance ray miss programs.
    //    prd.flags |= FLAG_VOLUME_SCATTERING;

    //    // Random walk through scattering volume, sampling the distance.
    //    // Note that the entry and exit of the volume is done according to the BSDF sampling.
    //    // Means glass with volume scattering will still do the proper refractions.
    //    // When the number of random walk steps has been exceeded, the next ray is shot with distance RT_DEFAULT_MAX
    //    // to hit something. If that results in a transmission the scattering volume is left.
    //    // If not, this continues until the maximum path length has been exceeded.
    //    if (prd.walk < sysData.walkLength)
    //    {
    //      const float3 albedo = safe_div(sigma_s, prd.sigma_t);
    //      const float2 xi     = rng2(prd.seed);
    //      
    //      const float s = sampleDensity(albedo, prd.throughput, prd.sigma_t, xi.x, prd.pdfVolume);

    //      // Prevent logf(0.0f) by sampling the inverse range (0.0f, 1.0f].
    //      prd.distance = -logf(1.0f - xi.y) / s;
    //    }
    //  }
    //}

    // Note that the primary rays and volume scattering miss cases do not offset the ray t_min by sysData.sceneEpsilon.
    optixTrace(theLaunchParameters.handle,
               prd.pos, prd.wi, // origin, direction
               epsilon, prd.distance, 0.0f, // tmin, tmax, time
               OptixVisibilityMask(0xFF), 
               // GLTF default is backface culling enabled, but the GAS OptixGeometryFlags disable
               // culling depending on the material doubleSided condition.
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    // Path termination by miss shader or sample() routines.
    if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
    {
      if (theLaunchParameters.showEnvironment == 0 && // If the environment should not be shown in primary rays and 
          depth == 0 &&                               // it's a primary ray and 
          (prd.flags & FLAG_HIT) == 0)                // it did not hit anything then 
      {
        prd.radiance = make_float3(0.0f);             // return black instead of the miss program result.
      }
      break; // Path termination.
    } 

    // Unbiased Russian Roulette path termination.
    if (theLaunchParameters.pathLengths.x <= depth) // Start termination after a minimum number of bounces.
    {
      const float probability = fmaxf(prd.throughput);

      if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
      {
        break; // Path termination.
      }

      prd.throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
    }

    // We're inside a volume and the scatter ray missed.
    //if (prd.flags & FLAG_VOLUME_SCATTERING_MISS) // This implies FLAG_VOLUME_SCATTERING.
    //{
    //  // Random walk through scattering volume, sampling the direction according to the phase function.
    //  sampleVolumeScattering(rng2(prd.seed), prd.stack[prd.idxStack].scattering_bias.w, prd.wi);
    //}

    ++depth; // Next path segment.
  }
  return prd.radiance;
}


// Accumulates radiance into launch parameters' buffer.
extern "C" __global__ void __raygen__path_tracer()
{
  //const uint2 theLaunchDim   = make_uint2(optixGetLaunchDimensions());
  const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());

  PerRayData prd;
  
  float2 fragment;

  if (theLaunchParameters.picking.x < 0.0f) // Normal rendering, not picking.
  {
    // Initialize the random number generator seed from the linear pixel index and the iteration index.
    // PERF This template really generates a lot of instructions.
    prd.seed = tea<4>(theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x, theLaunchParameters.iteration); 

    // Calculate primary ray direction from perspective camera position into the scene, in world space.
    fragment = make_float2(theLaunchIndex) + rng2(prd.seed); // Jitter sub-pixel location.
  }
  else // Picking ray.
  {
    prd.indexMaterial = -1; // Default result when not hitting anything.

    fragment = theLaunchParameters.picking;
  }

  const float2 ndc = (fragment / make_float2(theLaunchParameters.resolution)) * 2.0f - 1.0f; // Normalized device coordinates in range [-1, 1].

  if (theLaunchParameters.cameraType) // perspective camera
  {
    // All primary rays start at the camera position and go through the camera plane at the end of the W vector.
    prd.pos = theLaunchParameters.cameraP;
    prd.wi  = normalize(theLaunchParameters.cameraU * ndc.x +
                        theLaunchParameters.cameraV * ndc.y + 
                        theLaunchParameters.cameraW);
  }
  else // if orthographic camera
  {
    // Primary rays start on a camera plane at the camera postion and all directions are parallel.
    prd.pos = theLaunchParameters.cameraP + 
              theLaunchParameters.cameraU * ndc.x +
              theLaunchParameters.cameraV * ndc.y;
    prd.wi  = theLaunchParameters.cameraW; // W is normalized for orthographic cameras.
  }

  float3 radiance = integrator(prd);

#if USE_DEBUG_EXCEPTIONS // DEBUG Highlight numerical errors.
  if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
  {
    radiance = make_float3(1000000.0f, 0.0f, 0.0f); // super red
    //printf("NAN at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
  else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z))
  {
    radiance = make_float3(0.0f, 1000000.0f, 0.0f); // super green
    //printf("INF at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
  else if (radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f)
  {
    radiance = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
    //printf("NEG at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
#endif

  if (theLaunchParameters.picking.x < 0.0f) // Rendering.
  {
    const unsigned int index = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x; // Linear image pixel index.

    if (0 < theLaunchParameters.iteration)
    {
      const float t = 1.0f / static_cast<float>(theLaunchParameters.iteration + 1);

      const float3 colorAccum = make_float3(theLaunchParameters.bufferAccum[index]);

      radiance = lerp(colorAccum, radiance, t);
    }

    theLaunchParameters.bufferAccum[index] = make_float4(radiance, 1.0f);
  }
  else // Picking.
  {
    theLaunchParameters.bufferPicking[0] = prd.indexMaterial;
  }
}


// This raygen program is used with m_interop == INTEROP_IMG and directly accumulates inside the registered OpenGL display texture.
extern "C" __global__ void __raygen__path_tracer_surface()
{
  //const uint2 theLaunchDim   = make_uint2(optixGetLaunchDimensions());
  const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());

  PerRayData prd;
  
  float2 fragment;

  if (theLaunchParameters.picking.x < 0.0f) // Normal rendering, not picking.
  {
    // Initialize the random number generator seed from the linear pixel index and the iteration index.
    // PERF This template really generates a lot of instructions.
    prd.seed = tea<4>(theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x, theLaunchParameters.iteration); 

    // Calculate primary ray direction from perspective camera position into the scene, in world space.
    fragment = make_float2(theLaunchIndex) + rng2(prd.seed); // Jitter sub-pixel location.
  }
  else // Picking ray.
  {
    prd.indexMaterial = -1; // Default result when not hitting anything.

    fragment = theLaunchParameters.picking;
  }

  const float2 ndc = (fragment / make_float2(theLaunchParameters.resolution)) * 2.0f - 1.0f; // Normalized device coordinates in range [-1, 1].

  if (theLaunchParameters.cameraType) // perspective camera
  {
    // All primary rays start at the camera position and go through the camera plane at the end of the W vector.
    prd.pos = theLaunchParameters.cameraP;
    prd.wi  = normalize(theLaunchParameters.cameraU * ndc.x +
                        theLaunchParameters.cameraV * ndc.y + 
                        theLaunchParameters.cameraW);
  }
  else // if orthographic camera
  {
    // Primary rays start on a camera plane at the camera postion and all directions are parallel.
    prd.pos = theLaunchParameters.cameraP + 
              theLaunchParameters.cameraU * ndc.x +
              theLaunchParameters.cameraV * ndc.y;
    prd.wi  = theLaunchParameters.cameraW; // W is normalized for orthographic cameras.
  }

  float3 radiance = integrator(prd);

#if USE_DEBUG_EXCEPTIONS // DEBUG Highlight numerical errors.
  if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
  {
    radiance = make_float3(1000000.0f, 0.0f, 0.0f); // super red
    //printf("NAN at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
  else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z))
  {
    radiance = make_float3(0.0f, 1000000.0f, 0.0f); // super green
    //printf("INF at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
  else if (radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f)
  {
    radiance = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
    //printf("NEG at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
#endif

  if (theLaunchParameters.picking.x < 0.0f) // Rendering.
  {
    if (0 < theLaunchParameters.iteration)
    {
      float4 colorAccum;
      surf2Dread(&colorAccum, theLaunchParameters.surface, theLaunchIndex.x * sizeof(float4), theLaunchIndex.y, cudaBoundaryModeZero); 

      const float t = 1.0f / static_cast<float>(theLaunchParameters.iteration + 1);
      radiance = lerp(make_float3(colorAccum), radiance, t);
    }
    surf2Dwrite(make_float4(radiance, 1.0f), theLaunchParameters.surface, theLaunchIndex.x * sizeof(float4), theLaunchIndex.y);
  }
  else // Picking.
  {
    theLaunchParameters.bufferPicking[0] = prd.indexMaterial;
  }
}

