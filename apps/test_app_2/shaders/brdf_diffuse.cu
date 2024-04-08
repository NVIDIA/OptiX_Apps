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

  float3 normalGeo = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex);
  //float3 tangent   = attr0.tangent  * alpha + attr1.tangent  * theBarycentrics.x + attr2.tangent  * theBarycentrics.y; // PERF tangent is not used in this shader.
  float3 normal    = attr0.normal   * alpha + attr1.normal   * theBarycentrics.x + attr2.normal   * theBarycentrics.y;
  float3 texcoord  = attr0.texcoord * alpha + attr1.texcoord * theBarycentrics.x + attr2.texcoord * theBarycentrics.y;
  
  float4 objectToWorld[3];
  float4 worldToObject[3];

  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject); // Single instance level transformation list only.
  
  // All in world space coordinates!
  normalGeo = normalize(transformNormal(worldToObject, normalGeo));
  //tangent   = normalize(transformVector(objectToWorld, tangent));
  normal    = normalize(transformNormal(worldToObject, normal));

  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->flags   |= FLAG_HIT;
  thePrd->distance = optixGetRayTmax();
  thePrd->pos     += thePrd->wi * thePrd->distance;

  // If we're inside a volume and hit something, the path throughput needs to be modulated
  // with the transmittance along this segment before adding surface or light radiance!
  if (0 < thePrd->idxStack) // This assumes the first stack entry is vaccuum.
  {
    thePrd->throughput *= expf(thePrd->sigma_t * -thePrd->distance);

    // Increment the volume scattering random walk counter.
    // Unused when FLAG_VOLUME_SCATTERING is not set.
    ++thePrd->walk;
  }

  const MaterialDefinition& material = sysData.materialDefinitions[theData.idMaterial];

  float3 albedo = material.albedo;

  if (material.textureAlbedo != 0)
  {
    const float3 texColor = make_float3(tex2D<float4>(material.textureAlbedo, texcoord.x, texcoord.y));

    // Modulate the incoming color with the texture.
    albedo *= texColor;               // linear color, resp. if the texture has been uint8 and readmode set to use sRGB, then sRGB.
    //albedo *= powf(texColor, 2.2f); // sRGB gamma correction done manually.
  }

  // Start fresh with the next BSDF sample.
  // Save the current path throughput for the direct lighting contribution.
  // The path throughput will be modulated with the BSDF sampling results before that.
  const float3 throughput = thePrd->throughput;

  // BXDF sampling (brdf_diffuse)

  // Explicitly include edge-on cases as frontface condition!
  // Keeps the material stack from overflowing at silhouettes.
  // Prevents that silhouettes of thin-walled materials use the backface material.
  // Using the true geometry normal attribute as originally defined on the frontface!
  const bool isFrontFace = (0.0f <= dot(thePrd->wo, normalGeo));

  // When the ray is looking at the back side, flip the normals from their frontface definition to the backface.
  if (!isFrontFace)
  {
    normalGeo = -normalGeo;
    normal    = -normal;
  }

  // Cosine weighted hemisphere sampling for Lambert material.
  unitSquareToCosineHemisphere(rng2(thePrd->seed), normal, thePrd->wi, thePrd->pdf);

  if (thePrd->pdf <= 0.0f || dot(thePrd->wi, normalGeo) <= 0.0f)
  {
    thePrd->eventType = BSDF_EVENT_ABSORB;
    return;
  }

  // The cosine-weighted hemisphere sampling matches the diffuse distribution perfectly and all cosine terms cancel out.
  thePrd->throughput *= albedo; // bsdf * dot(thePrd->wi, normal) / pdf
  thePrd->eventType   = BSDF_EVENT_DIFFUSE_REFLECTION;

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

  // BXDF evaluation (brdf_diffuse)

  // Mind that the shading normal has been flipped to the ray side above.
  const float  pdf  = fmaxf(0.0f, dot(lightSample.direction, normal) * M_1_PIf); 
  // For a white Lambert material, the bxdf components match the evaluation pdf. (See MDL_renderer.)
  const float3 bxdf = albedo * pdf;

  if (pdf <= 0.0f || isNull(bxdf))
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

  const float weightMIS = (TYPE_LIGHT_POINT <= light.typeLight) ? 1.0f : balanceHeuristic(lightSample.pdf, pdf);
          
  // The sampled emission needs to be scaled by the inverse probability to have selected this light,
  // Selecting one of many lights means the inverse of 1.0f / numLights.
  // This is using the path throughput before the sampling modulated it above.
  thePrd->radiance += throughput * bxdf * lightSample.radiance_over_pdf * (float(numLights) * weightMIS);
}
