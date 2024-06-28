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


extern "C" __global__ void __closesthit__edf_diffuse()
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
  //float3 tangent   = attr0.tangent * alpha + attr1.tangent * theBarycentrics.x + attr2.tangent * theBarycentrics.y; // PERF tangent is not used in this shader.
  float3 normal    = attr0.normal  * alpha + attr1.normal  * theBarycentrics.x + attr2.normal  * theBarycentrics.y;
  float3 texcoord  = attr0.texcoord * alpha + attr1.texcoord * theBarycentrics.x + attr2.texcoord * theBarycentrics.y;
  
  float4 objectToWorld[3];
  float4 worldToObject[3];

  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject); // Single instance level transformation list only.
  
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

  // When hitting a geometric light, evaluate the emission first, because this needs the previous diffuse hit's pdf.
  const int idLight = theData.idLight;

  // Explicitly include edge-on cases as frontface condition!
  // Keeps the material stack from overflowing at silhouettes.
  // Prevents that silhouettes of thin-walled materials use the backface material.
  // Using the true geometry normal attribute as originally defined on the frontface!
  const float cosTheta = dot(thePrd->wo, normalGeo);

  // If the material is not emissive or we're looking at the back face,
  // do not add any radiance and end the path.
  if (idLight < 0 || cosTheta < DENOMINATOR_EPSILON)
  {
    thePrd->eventType = BSDF_EVENT_ABSORB;
    return;
  }

  // This material is emissive and we're looking at the front face.
  const LightDefinition& light = sysData.lightDefinitions[idLight];

  float pdf = 1.0f; // Neutral factor in case there is no light.texture.

  // Radiant exitance to radiance conversion: Multiply with the EDF which for the diffuse EDF is 1/pi.
  // The pdf (non-projected hemisphere) ist cos/pi.
  float3 radiance = light.emission * M_1_PIf;
     
  if (light.textureEmission)
  {
    const float3 emission = make_float3(tex2D<float4>(light.textureEmission, texcoord.x, texcoord.y));
      
    radiance *= emission;
      
    // Rectangle lights are importance sampled! Mesh lights are uniformly sampled and don't have this additional pdf factor.
    if (light.typeLight == TYPE_LIGHT_RECT)
    {
      // The pdf to have picked this emission on the texture.
      pdf *= intensity(emission) * light.invIntegral; // This must be the emission from the texture only!
    }
  }

  // Both PDFs multiplied! Latter is light area to solid angle (projected area) pdf. Assumes light.area != 0.0f.
  pdf *= thePrd->distance * thePrd->distance / (light.area * cosTheta); // Solid angle measure.

  float weightMIS = 1.0f;

  // If the last event was diffuse or glossy, calculate the opposite MIS weight for this implicit light hit.
  // DAR FIXME PERF None of the light pdf calculations are required when there is no direct lighting or of the previous event type was specular.
  if (sysData.directLighting && (thePrd->eventType & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY)))
  {
    weightMIS = balanceHeuristic(thePrd->pdf, pdf);
  }
     
  thePrd->radiance += thePrd->throughput * radiance * weightMIS;
}
