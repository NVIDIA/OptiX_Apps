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
#include "vertex_attributes.h"
#include "function_indices.h"
#include "material_definition.h"
#include "light_definition.h"
#include "shader_common.h"
#include "transform.h"
#include "random_number_generators.h"


extern "C" __constant__ SystemData sysData;


extern "C" __global__ void __closesthit__brdf_diffuse()
{
  GeometryInstanceData theData = sysData.geometryInstanceData[optixGetInstanceId()];

  // Cast the CUdeviceptr to the actual format for Triangles geometry.
  const unsigned int thePrimitiveIndex = optixGetPrimitiveIndex();

  const uint3* indices = reinterpret_cast<uint3*>(theData.indices);
  const uint3  tri     = indices[thePrimitiveIndex];

  const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(theData.attributes);

  const TriangleAttributes& attr0 = attributes[tri.x];
  const TriangleAttributes& attr1 = attributes[tri.y];
  const TriangleAttributes& attr2 = attributes[tri.z];

  const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
  const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;
  
  // PERF This State lies in memory. It's more efficient to hold the data in registers.
  //      Problem is that more advanced material systems need the State all the time.
  State state; // All in world space coordinates!

  state.normalGeo = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex);
  //state.tangent   = attr0.tangent  * alpha + attr1.tangent  * theBarycentrics.x + attr2.tangent  * theBarycentrics.y; // PERF tangent is not used in this shader.
  state.normal    = attr0.normal   * alpha + attr1.normal   * theBarycentrics.x + attr2.normal   * theBarycentrics.y;
  state.texcoord  = attr0.texcoord * alpha + attr1.texcoord * theBarycentrics.x + attr2.texcoord * theBarycentrics.y;
  
  float4 objectToWorld[3];
  float4 worldToObject[3];

  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject); // Single instance level transformation list only.
  
  state.normalGeo = normalize(transformNormal(worldToObject, state.normalGeo));
  //state.tangent   = normalize(transformVector(objectToWorld, state.tangent));
  state.normal    = normalize(transformNormal(worldToObject, state.normal));

  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->distance = optixGetRayTmax(); // Return the current path segment distance, needed for absorption calculations in the integrator.
  
  //thePrd->pos = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
  thePrd->pos += thePrd->wi * thePrd->distance; // DEBUG Check which version is more efficient.

  // Explicitly include edge-on cases as frontface condition!
  // Keeps the material stack from overflowing at silhouettes.
  // Prevents that silhouettes of thin-walled materials use the backface material.
  // Using the true geometry normal attribute as originally defined on the frontface!
  thePrd->flags |= (0.0f <= dot(thePrd->wo, state.normalGeo)) ? FLAG_FRONTFACE : 0;

  if ((thePrd->flags & FLAG_FRONTFACE) == 0) // Looking at the backface?
  {
    // Means geometric normal and shading normal are always defined on the side currently looked at.
    // This gives the backfaces of opaque BSDFs a defined result.
    state.normalGeo = -state.normalGeo;
    //state.tangent   = -state.tangent;
    state.normal    = -state.normal;
    // Explicitly DO NOT recalculate the frontface condition!
  }
  
  thePrd->radiance = make_float3(0.0f);

  // Start fresh with the next BSDF sample. (Either of these values remaining zero is an end-of-path condition.)
  // The pdf of the previous evene was needed for the emission calculation above.
  thePrd->f_over_pdf = make_float3(0.0f);
  thePrd->pdf        = 0.0f;

  const MaterialDefinition& material = sysData.materialDefinitions[theData.idMaterial];

  state.albedo = material.albedo;

  if (material.textureAlbedo != 0)
  {
    const float3 texColor = make_float3(tex2D<float4>(material.textureAlbedo, state.texcoord.x, state.texcoord.y));

    // Modulate the incoming color with the texture.
    state.albedo *= texColor;               // linear color, resp. if the texture has been uint8 and readmode set to use sRGB, then sRGB.
    //state.albedo *= powf(texColor, 2.2f); // sRGB gamma correction done manually.
  }
 
  // Only the last diffuse hit is tracked for multiple importance sampling of implicit light hits.
  thePrd->flags = (thePrd->flags & ~FLAG_DIFFUSE) | FLAG_HIT | material.flags; // FLAG_THINWALLED can be set directly from the material.

  // BXDF sampling (brdf_diffuse)

  // Cosine weighted hemisphere sampling for Lambert material.
  unitSquareToCosineHemisphere(rng2(thePrd->seed), state.normal, thePrd->wi, thePrd->pdf);

  if (thePrd->pdf <= 0.0f || dot(thePrd->wi, state.normalGeo) <= 0.0f)
  {
    thePrd->flags |= FLAG_TERMINATE;
    return;
  }

  thePrd->f_over_pdf = state.albedo;
  thePrd->flags     |= FLAG_DIFFUSE; // Direct lighting will be done with multiple importance sampling.

  // Direct lighting if the sampled BSDF was diffuse and any light is in the scene.
  const int numLights = sysData.numLights;

  if (numLights == 0 || sysData.directLighting == 0)
  {
    return;
  }

  // Sample one of many lights. 
  // The caller picks the light to sample. Make sure the index stays in the bounds of the sysData.lightDefinitions array.
  const int indexLight = (1 < numLights) ? clamp(static_cast<int>(floorf(rng(thePrd->seed) * numLights)), 0, numLights - 1) : 0;
    
  const LightDefinition& light = sysData.lightDefinitions[indexLight];
    
  const int callLight = NUM_LENS_TYPES + light.typeLight;

  LightSample lightSample = optixDirectCall<LightSample, const LightDefinition&, PerRayData*>(callLight, light, thePrd);

  if (lightSample.pdf <= 0.0f)
  {
    return;
  }

  // Evaluate the BXDF in the light sample direction. Normally cheaper than shooting rays.
  // Lambert:
  const float3 bxdf = state.albedo * M_1_PIf;
  const float  pdf  = fmaxf(0.0f, dot(lightSample.direction, state.normal) * M_1_PIf);

  if (pdf <= 0.0 || isNull(bxdf))
  {
    return;
  }

  // The shadow ray is only a single payload to indicate the visibility test result.
  // Default to visibilty being blocked by geometry. If the miss shader is reached this gets set to 1.
  unsigned int isVisible = 0; 

  // Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval 
  // to prevent self-intersections with the actual light geometry in the scene.
  optixTrace(sysData.topObject,
              thePrd->pos, lightSample.direction, // origin, direction
              sysData.sceneEpsilon, lightSample.distance - sysData.sceneEpsilon, 0.0f, // tmin, tmax, time
              OptixVisibilityMask(0xFF), 
              OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
              0, 0, TYPE_RAY_SHADOW, // The shadow ray type only uses the miss program.
              isVisible);

  if (!isVisible)
  {
    return;
  }

  if (thePrd->flags & FLAG_VOLUME) // Supporting nested materials includes having lights inside a volume.
  {
    // Calculate the transmittance along the light sample's distance in case it's inside a volume.
    // The light must be in the same volume or it would have been shadowed.
    lightSample.emission *= expf(-lightSample.distance * thePrd->sigma_t);
  }

  if (TYPE_LIGHT_POINT <= light.typeLight)
  {
    // Singular light, cannot be hit implicitly, no light PDF and no MIS here.
    thePrd->radiance += bxdf * lightSample.emission * dot(lightSample.direction, state.normal);
  }
  else 
  {
    const float weightMis = balanceHeuristic(lightSample.pdf, pdf);
            
    thePrd->radiance += bxdf * lightSample.emission * (weightMis * dot(lightSample.direction, state.normal) / lightSample.pdf);
  }
}
