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
//#include "vertex_attributes.h"
#include "material_definition.h"
#include "shader_common.h"
#include "random_number_generators.h"


extern "C" __constant__ SystemData sysData;


// One anyhit program for the radiance ray for all materials with cutout opacity!
extern "C" __global__ void __anyhit__radiance_cutout()
{
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  //thePrd->flags |= FLAG_ANYHIT; // DEBUG Flag to visualize __anyhit__radiance_cutout invocations.

  GeometryInstanceData* theData = reinterpret_cast<GeometryInstanceData*>(optixGetSbtDataPointer());

  const MaterialDefinition& material = sysData.materialDefinitions[theData->idMaterial];

  if (material.textureCutout != 0)
  {
    // Cast the CUdeviceptr to the actual format for Triangles geometry.
    const uint3*              indices    = reinterpret_cast<uint3*>(theData->indices);
    const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(theData->attributes);

    const unsigned int thePrimitiveIndex = optixGetPrimitiveIndex();

    const uint3 tri = indices[thePrimitiveIndex];
    
    const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma

    const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

    const float3 texcoord = attributes[tri.x].texcoord * alpha +
                            attributes[tri.y].texcoord * theBarycentrics.x +
                            attributes[tri.z].texcoord * theBarycentrics.y;

    // This was originally designed with greyscale RGB in mind.
    // const float opacity = intensity(make_float3(tex2D<float4>(material.textureCutout, texcoord.x, texcoord.y)));
    // The OMM baker used in rtigo9_omm uses the alpha channel of the RGBA input. Match this example to enable performance comparisons.
    const float opacity = tex2D<float4>(material.textureCutout, texcoord.x, texcoord.y).w;
  
    // Stochastic alpha test to get an alpha blend effect.
    if (opacity < 1.0f && opacity <= rng(thePrd->seed)) // No need to calculate an expensive random number if the test is going to fail anyway.
    {
      optixIgnoreIntersection();
    }
  }
}


// The shadow ray program for all materials with no cutout opacity.
// With OMMs this is not needed anymore because that uses OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT and there is no hit program for shadow rays.
//extern "C" __global__ void __anyhit__shadow()
//{
//  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
//
//  // Always set payload values before calling optixIgnoreIntersection or optixTerminateRay because they return immediately!
//  thePrd->flags |= FLAG_SHADOW; // Visbility check failed. // With OMMs this is set by the caller. The miss program clears it.
//
//  optixTerminateRay();
//}


extern "C" __global__ void __anyhit__shadow_cutout() // For the radiance ray type.
{
  GeometryInstanceData* theData = reinterpret_cast<GeometryInstanceData*>(optixGetSbtDataPointer());

  const MaterialDefinition& material = sysData.materialDefinitions[theData->idMaterial];

  float opacity = 1.0f;

  if (material.textureCutout != 0)
  {
    const unsigned int thePrimitiveIndex = optixGetPrimitiveIndex();
    const uint3* indices = reinterpret_cast<uint3*>(theData->indices);
    const uint3  tri     = indices[thePrimitiveIndex];

    const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(theData->attributes);

    const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
    const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

    const float3 texcoord = attributes[tri.x].texcoord * alpha +
                            attributes[tri.y].texcoord * theBarycentrics.x +
                            attributes[tri.z].texcoord * theBarycentrics.y;

    // This was originally designed with greyscale RGB in mind.
    //opacity = intensity(make_float3(tex2D<float4>(material.textureCutout, texcoord.x, texcoord.y)));
    // The OMM baker used in rtigo9_omm uses the alpha channel of the RGBA input. Match this example to enable performance comparisons.
    opacity = tex2D<float4>(material.textureCutout, texcoord.x, texcoord.y).w;
  }

  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  // Stochastic alpha test to get an alpha blend effect.
  if (opacity < 1.0f && opacity <= rng(thePrd->seed)) // No need to calculate an expensive random number if the test is going to fail anyway.
  {
    optixIgnoreIntersection();
  }
  else
  {
    // Always set payload values before calling optixIgnoreIntersection or optixTerminateRay because they return immediately!
    //thePrd->flags |= FLAG_SHADOW; // With OMMs this is set by the caller. The miss program clears it.

    optixTerminateRay();
  }
}
