/* 
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include "sdf_attributes.h"
#include "vector_math.h"

extern "C" __constant__ SystemData sysData;

// Compute the near and far intersections of the cube (stored in the x and y components) using the slab method.
// No intersection when tNear > tFar.
__forceinline__ __device__ float2 intersectAABB(float3 origin, float3 direction, float3 minAabb, float3 maxAabb)
{
  const float3 tMin = (minAabb - origin) / direction;
  const float3 tMax = (maxAabb - origin) / direction;
  
  const float3 t0 = fminf(tMin, tMax);
  const float3 t1 = fmaxf(tMin, tMax);

  const float tNear = fmaxf(t0);
  const float tFar  = fminf(t1);

  return make_float2(tNear, tFar);
}


// The SDF intersection function for single-component 3D texture data.
extern "C" __global__ void __intersection__sdf()
{
  const GeometryInstanceData theData = sysData.geometryInstanceData[optixGetInstanceId()];
  
  // Cast the CUdeviceptr to the actual format for SignedDistanceField geometry data.
  const SignedDistanceFieldAttributes* attributes = reinterpret_cast<const SignedDistanceFieldAttributes*>(theData.attributes);

  // There is only one primitive index inside the SDF attribute!
  // PERF Explicitly do not load the uint3 sdfTextureSize which is unused here.
  const float3        minimum   = attributes->minimum;
  const float3        maximum   = attributes->maximum;
  cudaTextureObject_t texture   = attributes->sdfTexture;
  const float         lipschitz = attributes->sdfLipschitz;

  const float3 theRayOrigin = optixGetObjectRayOrigin();

  // DANGER: The object space ray direction is not normalized if there is a scaling transform over the geometry!
  // Both the ray origin and ray direction are transformed into object space. The ray tmin and tmax values are not touched.
  // When the ray.direction is not a unit vector, the sphere tracing is not stepping with the expected SDF distances.
  // If the AABB was scaled bigger, it would need more steps because the object space direction is shorter.
  // And vice versa, which is even worse, when the AABB was scaled smaller, the ray direction vector is longer than 1.0
  // and the sphere tracing will actually overshoot and miss the surface. 
  // The invLength factor will correct all that!
  const float3 theRayDirection = optixGetObjectRayDirection();

  // At this point calculate start and end values from the intersections of the ray with the primitive's AABB.
  float2 t = intersectAABB(theRayOrigin, theRayDirection, minimum, maximum);

  const float theRayTmin = optixGetRayTmin();
  const float theRayTmax = optixGetRayTmax();

  // Factor scaling the SDF distance into object space.
  // Merge the Lipschitz distance scaling factor into that.
  const float invLength = 1.0f / (length(theRayDirection) * lipschitz);

  // Factor used during projection of the position into 3D texture space.
  const float3 invExtents = 1.0f / (maximum - minimum);
  
  // PERF Testing the intersection with the SDF only inside the enclosing AABB will make 
  // shadow rays to environment lights way faster because they do no iterate the full count
  // or until distance is infinity.

  // If tNear on the AABB is behind the ray.tmin, use the ray.tmin value as start value for the sphere tracing.
  t.x = (theRayTmin < t.x) ? t.x : theRayTmin;
  // If the tFar value on the AABB is farther than the ray.tmax, limit the sphere tracing end condition to the ray.tmax value.
  t.y = (t.y < theRayTmax) ? t.y : theRayTmax;

  const int   count      = sysData.sdfIterations; // Maximum number of sphere tracing iterations.
  const float sdfEpsilon = sysData.sdfEpsilon;    // Minimum distance to SDF iso-surface at 0.0f which counts as hit.

  // The four reported intersection attributes. Unused when there is no intersection.
  float  sign;
  float3 uvw;
  
  int iteration = 0;

  for (; iteration < count && t.x < t.y; ++iteration)
  {
    float3 position = theRayOrigin + theRayDirection * t.x;

    uvw = (position - minimum) * invExtents; // Position transformed into 3D texture coordinate space in range [0.0f, 1.0f].
   
    float distance = tex3D<float>(texture, uvw.x, uvw.y, uvw.z) * invLength; 

    if (iteration == 0)
    {
      // Positive started ouside the SDF, negative started inside.
      sign = copysignf(1.0f, distance);
    }
    else if (((__float_as_uint(sign) ^ __float_as_uint(distance)) & 0x80000000) != 0) // Detect overshoot (different sign bits).
    {
      // Adjust the intersection to lie exactly on the iso-surface.
      // (sign * distance) is always negative here.
      t.x += sign * distance;

      // Recalculate the uvw coordinate on the surface returned as intersection attibute.
      position = theRayOrigin + theRayDirection * t.x;
      uvw      = (position - minimum) * invExtents;
      break;
    }

    // Here (sign * distance) must be positive. 
    // Saves an fabsf() in the next condition and is needed for the next sphere tracing step anyway.
    distance *= sign; 

    // Near enough to the SDF iso-surface?
    // This must be on the same side as the start point because the distance's sign hasn't changed.
    if (distance < sdfEpsilon) 
    {
      break;
    }

    // Next sphere tracing step in positive ray direction, irrespective of the interval's start position due to the (sign * distance) above.
    t.x += distance; 
  }

  if (iteration < count && t.x < t.y)
  {
    optixReportIntersection(t.x, 1, 
                            __float_as_uint(sign),  // The sign attribute is the indicator for isFrontFace of an SDF iso-surface.
                            __float_as_uint(uvw.x), // Return the texture coordinate at which the iso-surface was found 
                            __float_as_uint(uvw.y), // to have the perfect value for the normal attribute calculation.
                            __float_as_uint(uvw.z));
  }
}
