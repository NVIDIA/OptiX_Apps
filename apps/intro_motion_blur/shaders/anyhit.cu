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
//#include "vertex_attributes.h"
#include "material_parameter.h"
#include "shader_common.h"
#include "random_number_generators.h"


extern "C" __constant__ SystemParameter sysParameter;


// One anyhit program for the radiance ray for all materials with cutout opacity!
extern "C" __global__ void __anyhit__radiance_cutout()
{
  GeometryInstanceData* theData = reinterpret_cast<GeometryInstanceData*>(optixGetSbtDataPointer());

  MaterialParameter const& parameters = sysParameter.materialParameters[theData->materialIndex];

  if (parameters.textureCutout != 0)
  {
    const unsigned int thePrimtiveIndex = optixGetPrimitiveIndex();

    const int3 tri = theData->indices[thePrimtiveIndex];

    const VertexAttributes& va0 = theData->attributes[tri.x];
    const VertexAttributes& va1 = theData->attributes[tri.y];
    const VertexAttributes& va2 = theData->attributes[tri.z];

    const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
    const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

    const float3 texcoord = va0.texcoord * alpha + va1.texcoord * theBarycentrics.x + va2.texcoord * theBarycentrics.y;

    const float opacity = intensity(make_float3(tex2D<float4>(parameters.textureCutout, texcoord.x, texcoord.y)));

    PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

    // Stochastic alpha test to get an alpha blend effect.
    if (opacity < 1.0f && opacity <= rng(thePrd->seed)) // No need to calculate an expensive random number if the test is going to fail anyway.
    {
      optixIgnoreIntersection();
    }
  }
}


// The shadow ray program for all materials with no cutout opacity.
extern "C" __global__ void __anyhit__shadow()
{
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->flags |= FLAG_SHADOW; // Visbility check failed.

  optixTerminateRay();
}


extern "C" __global__ void __anyhit__shadow_cutout()
{
  GeometryInstanceData* theData = reinterpret_cast<GeometryInstanceData*>(optixGetSbtDataPointer());

  MaterialParameter const& parameters = sysParameter.materialParameters[theData->materialIndex];

  float opacity = 1.0f;

  if (parameters.textureCutout != 0)
  {
    const unsigned int thePrimtiveIndex = optixGetPrimitiveIndex();

    const int3 tri = theData->indices[thePrimtiveIndex];

    const VertexAttributes& va0 = theData->attributes[tri.x];
    const VertexAttributes& va1 = theData->attributes[tri.y];
    const VertexAttributes& va2 = theData->attributes[tri.z];

    const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
    const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

    const float3 texcoord = va0.texcoord * alpha + va1.texcoord * theBarycentrics.x + va2.texcoord * theBarycentrics.y;

    opacity = intensity(make_float3(tex2D<float4>(parameters.textureCutout, texcoord.x, texcoord.y)));
  }

  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  // Stochastic alpha test to get an alpha blend effect.
  if (opacity < 1.0f && opacity <= rng(thePrd->seed)) // No need to calculate an expensive random number if the test is going to fail anyway.
  {
    optixIgnoreIntersection();
  }
  else
  {
    thePrd->flags |= FLAG_SHADOW;
    optixTerminateRay();
  }
}
