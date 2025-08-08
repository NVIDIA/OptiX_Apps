/* 
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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
#include "bxdf_common.h"

extern "C" __constant__ SystemData sysData;

// Combining state, sample and eval data into one structure.
// That way the normals only need to be flipped once to the side of the ray (k1 outgoing direction)
// The ior doesn't need to be set twice,
struct __align__(8) State_GGX_BSDF
{
  // 8 byte aligned
  float2 roughness; // material: .x == roughness_u, .y = roughness_v
  // 4 byte aligned
  float3 tint;      // material.albedo
  // Geometry state in world space. 
  // normalGeo and normal are flipped to the ray side by the caller.
  float3 normalGeo;
  // Shading space.
  float3 tangent;
  float3 bitangent;
  float3 normal;
  
  float3 xi;                  // sample input: pseudo-random sample numbers in range [0, 1). bsdf_ggx_smith needs three samples.
  float ior1;                 // sample and eval input: IOR current medium
  float ior2;                 // sample and eval input: IOR other side
  float3 k1;                  // sample and eval input: outgoing direction (== prd.wo == negative optixGetWorldRayDirection())
  float3 k2;                  // sample output: incoming direction (continuation ray, prd.wi)
                              // eval input:    incoming direction (direction to light sample point)
  float3 bsdf_over_pdf;       // sample output: bsdf * dot(k2, normal) / pdf
  float  pdf;                 // sample and eval output: pdf (non-projected hemisphere) 
  Bsdf_event_type event_type; // sample output: the type of event for the generated sample (absorb, glossy_reflection, glossy_transmission)

  // 1 byte aligned
  bool isThinWalled; // material: FLAG_THINWALLED
};


__forceinline__ __device__ void bsdf_ggx_smith_sample(State_GGX_BSDF& state)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  state.bsdf_over_pdf = make_float3(0.0f);
  state.pdf           = 0.0f;

  const float2 ior = make_float2(state.ior1, state.ior2); 

  const float nk1 = fabsf(dot(state.k1, state.normal));

  const float3 k10 = make_float3(dot(state.k1, state.tangent),
                                 dot(state.k1, state.bitangent),
                                 nk1);

  // Sample half-vector, microfacet normal.
  const float3 h0 = hvd_ggx_sample_vndf(k10, state.roughness, make_float2(state.xi.x, state.xi.y));

  if (fabsf(h0.z) == 0.0f)
  {
    state.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * state.tangent + h0.y * state.bitangent + h0.z * state.normal;

  const float kh = dot(state.k1, h);

  if (kh <= 0.0f)
  {
    state.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Case scatter_reflect_transmit with default Fresnel function.
  const float fr = ior_fresnel(ior.y / ior.x, kh); // Fresnel reflection value. (1.0f always reflect).

  // Compute probability of selection refraction over reflection
  float prob;

  if (state.xi.z < fr)
  {
    prob = fr; // Reflection probability.

    // BRDF: reflect
    state.k2            = (2.0f * kh) * h - state.k1;
    state.bsdf_over_pdf = make_float3(1.0f); // Was: f_refl_c / f_refl; // PERF Always white with the original setup.
    state.event_type    = BSDF_EVENT_GLOSSY_REFLECTION;
  }
  else
  {
    prob = 1.0f - fr; // Transmission probability.

    bool tir = false;

    if (state.isThinWalled) // No refraction!
    {
      // pseudo-BTDF: flip a reflected reflection direction to the back side
      state.k2 = (2.0f * kh) * h - state.k1;
      state.k2 = normalize(state.k2 - 2.0f * state.normal * dot(state.k2, state.normal));
    }
    else
    {
      // BTDF: refract
      state.k2 = refract(state.k1, h, ior.x / ior.y, kh, tir);
    }

    state.bsdf_over_pdf = make_float3(1.0f); // Was: (make_float3(1.0f) - f_refl_c) / prob; // PERF Always white with the original setup.
    state.event_type    = (tir) ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;
  }

  // Check if the resulting direction is on the correct side of the actual geometry
  const float gnk2 = dot(state.k2, state.normalGeo) * ((state.event_type == BSDF_EVENT_GLOSSY_REFLECTION) ? 1.0f : -1.0f);
  
  if (gnk2 <= 0.0f)
  {
    state.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  const float nk2 = fabsf(dot(state.k2, state.normal));
  const float k2h = fabsf(dot(state.k2, h));

  float G1;
  float G2;

  const float G12 = ggx_smith_shadow_mask(G1, G2, k10, make_float3(dot(state.k2, state.tangent), dot(state.k2, state.bitangent), nk2), state.roughness);
  
  if (G12 <= 0.0f)
  {
    state.event_type = BSDF_EVENT_ABSORB;
    return;
  }
  
  state.bsdf_over_pdf *= G12 / G1;

  // Compute pdf
  state.pdf = hvd_ggx_eval(1.0f / state.roughness, h0) * G1 * prob;

  if (!state.isThinWalled && (state.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION)) // if (refraction)
  {
    const float tmp = kh * ior.x - k2h * ior.y;

    state.pdf *= kh * k2h / (nk1 * h0.z * tmp * tmp);
  }
  else
  {
    state.pdf *= 0.25f / (nk1 * h0.z);
  }

  state.bsdf_over_pdf *= state.tint; // DAR FIXME PERF state.tint isn't really necessary for just this.
}


__forceinline__ __device__ float3 bsdf_ggx_smith_eval(State_GGX_BSDF& state)
{
  const float2 ior = make_float2(state.ior1, state.ior2); 

  const float nk1 = fabsf(dot(state.k1, state.normal));
  const float nk2 = fabsf(dot(state.k2, state.normal));
  
  // BTDF or BRDF eval? If the incoming light direction is on the backface.
  const bool backside = (dot(state.k2, state.normalGeo) < 0.0f);
  
  const float3 h = compute_half_vector(state.k1, state.k2, state.normal, ior, nk2, backside, state.isThinWalled);

  // Invalid for reflection / refraction?
  const float nh  = dot(state.normal, h);
  const float k1h = dot(state.k1, h);
  const float k2h = dot(state.k2, h) * (backside ? -1.0f : 1.0f);

  if (nh < 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    state.pdf = 0.0f; // absorb
    return make_float3(0.0f);
  }

  // Case scatter_reflect_transmit with default Fresnel function:
  float fr = ior_fresnel(ior.y / ior.x, k1h);

  // Compute BSDF and pdf
  const float3 h0 = make_float3(dot(state.tangent, h), dot(state.bitangent, h), nh);

  state.pdf = hvd_ggx_eval(1.0f / state.roughness, h0);

  float G1;
  float G2;

  const float G12 = ggx_smith_shadow_mask(G1, G2, 
                                          make_float3(dot(state.tangent, state.k1), dot(state.bitangent, state.k1), nk1),
                                          make_float3(dot(state.tangent, state.k2), dot(state.bitangent, state.k2), nk2),
                                          state.roughness);
  
  if (!state.isThinWalled && backside) // Refraction?
  {
    // Refraction pdf and BTDF
    const float tmp = k1h * ior.x - k2h * ior.y;

    state.pdf *= k1h * k2h / (nk1 * nh * tmp * tmp);
  }
  else
  {
    // Reflection pdf and BRDF (and pseudo-BTDF for thin-walled)
    state.pdf *= 0.25f / (nk1 * nh);
  }

  const float prob = (backside) ? 1.0f - fr : fr;

  const float3 bsdf = make_float3(prob * G12 * state.pdf);
  
  state.pdf *= prob * G1;

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  return bsdf * state.tint; // DAR FIXME clamp(tint, 0.0f, 1.0f); The caller should do this.
}



extern "C" __global__ void __closesthit__bsdf_ggx_smith()
{
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  GeometryInstanceData theData = sysData.geometryInstanceData[optixGetInstanceId()];

  // Cast the CUdeviceptr to the actual format for Triangles geometry.
  const unsigned int thePrimitiveIndex = optixGetPrimitiveIndex();

  const uint3* indices = reinterpret_cast<uint3*>(theData.indices);
  const uint3  tri = indices[thePrimitiveIndex];

  const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(theData.attributes);

  const TriangleAttributes& attr0 = attributes[tri.x];
  const TriangleAttributes& attr1 = attributes[tri.y];
  const TriangleAttributes& attr2 = attributes[tri.z];

  const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
  const float  bary_a = 1.0f - theBarycentrics.x - theBarycentrics.y;

  State_GGX_BSDF state;

  state.normalGeo = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex);
  state.tangent   = attr0.tangent  * bary_a + attr1.tangent  * theBarycentrics.x + attr2.tangent  * theBarycentrics.y;
  state.normal    = attr0.normal   * bary_a + attr1.normal   * theBarycentrics.x + attr2.normal   * theBarycentrics.y;
  float3 texcoord = attr0.texcoord * bary_a + attr1.texcoord * theBarycentrics.x + attr2.texcoord * theBarycentrics.y;

  float4 objectToWorld[3];
  float4 worldToObject[3];

  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject); // Single instance level transformation list only.

  // All in world space coordinates!
  state.normalGeo = normalize(transformNormal(worldToObject, state.normalGeo));
  state.tangent   = normalize(transformVector(objectToWorld, state.tangent)); // No need to normalize this.
  state.normal    = normalize(transformNormal(worldToObject, state.normal));

  // Generate an ortho-normal basis where the normal stays intact and the tangent keeps its orientation.
  // From TBN() class. 
  state.bitangent = normalize(cross(state.normal, state.tangent));
  state.tangent   = cross(state.bitangent, state.normal);

  thePrd->flags |= FLAG_HIT;
  thePrd->distance = optixGetRayTmax();
  thePrd->pos += thePrd->wi * thePrd->distance;

  // If we're inside a volume and hit something, the path throughput needs to be modulated
  // with the transmittance along this segment before adding surface or light radiance!
  if (0 < thePrd->idxStack) // This assumes the first stack entry is vaccuum.
  {
    thePrd->throughput *= expf(thePrd->sigma_t * -thePrd->distance);

    // Increment the volume scattering random walk counter.
    // Unused when FLAG_VOLUME_SCATTERING is not set.
    ++thePrd->walk;
  }

  // Start fresh with the next BSDF sample.
  // Save the current path throughput for the direct lighting contribution.
  // The path throughput will be modulated with the BSDF sampling results before that.
  const float3 throughput = thePrd->throughput;

  const MaterialDefinition& material = sysData.materialDefinitions[theData.idMaterial];

  state.tint = material.albedo;

  if (material.textureAlbedo != 0)
  {
    const float3 texColor = make_float3(tex2D<float4>(material.textureAlbedo, texcoord.x, texcoord.y));

    // Modulate the incoming color with the texture.
    state.tint *= texColor;               // linear color, resp. if the texture has been uint8 and readmode set to use sRGB, then sRGB.
    //state.tint *= powf(texColor, 2.2f); // sRGB gamma correction done manually.
  }

  // Explicitly include edge-on cases as frontface condition!
  // Keeps the material stack from overflowing at silhouettes.
  // Prevents that silhouettes of thin-walled materials use the backface material.
  // Using the true geometry normal attribute as originally defined on the frontface!
  const bool isFrontFace = (0.0f <= dot(thePrd->wo, state.normalGeo));

  // Flip the normals to the side the ray hit.
  if (!isFrontFace)
  {
    state.normalGeo = -state.normalGeo;
    state.normal    = -state.normal;
  }

  state.isThinWalled = ((material.flags & FLAG_THINWALLED) != 0);
  state.roughness    = material.roughness;

  int idx = thePrd->idxStack;

  // If the hit is either on the surface or a thin-walled material,
  // the ray is inside the surrounding material and the material ior is on the other side.
  if (isFrontFace || state.isThinWalled)
  {
    state.ior1 = thePrd->stack[idx].absorption_ior.w; // From surrounding medium ior
    state.ior2 = material.absorption_ior.w;           // to material ior.
  }
  else
  {
    // When hitting the backface of a non-thin-walled material, 
    // the ray is inside the current material and the surrounding material is on the other side.
    // That material's IOR is the current top-of-stack after the previous transmission. We need the one further down!
    idx = max(0, idx - 1);

    state.ior1 = material.absorption_ior.w;           // From material ior
    state.ior2 = thePrd->stack[idx].absorption_ior.w; // to surrounding medium ior.
  }

  // DAR FIXME This GGX implementation doesn't handle identical IORs!
  // Fix the state.ior values to have a tiny difference.
  const float iorDiff = state.ior2 - state.ior1;

  if (fabsf(iorDiff) < IOR_THRESHOLD)
  {
    state.ior2 = state.ior1 + copysignf(IOR_THRESHOLD, iorDiff);
  }

  state.xi = rng3(thePrd->seed); // The bsdf_ggx_smith_sample needs three random values.
  state.k1 = thePrd->wo;         // == -optixGetWorldRayDirection()

  bsdf_ggx_smith_sample(state);

  thePrd->wi          = state.k2;            // Continuation direction.
  thePrd->throughput *= state.bsdf_over_pdf; // Adjust the path throughput for all following incident lighting.
  thePrd->pdf         = state.pdf;           // Note that specular events in MDL return pdf == 0.0f! (=> Not a path termination condition.)
  thePrd->eventType   = state.event_type;    // If this is BSDF_EVENT ABSORB, the path ends inside the integrator and the radiance is returned.
                                             // Keep calculating the radiance of the current hit point though.
  // End of sampling

  if (!state.isThinWalled && (state.event_type & BSDF_EVENT_TRANSMISSION) != 0)
  {
    updateMaterialStack(thePrd, material, isFrontFace);
  }

  // Direct lighting if the sampled BSDF was diffuse or glossy and any light is in the scene.
  // PERF We know  that the sampling was glossy when this is reached. 
  // No need to to check the thePrd->eventType to see if it's diffuse oor glossy which can handle direct lighting.
  const int numLights = sysData.numLights;

  if (!sysData.directLighting || numLights <= 0)
  {
    return;
  }

  // Sample one of many lights.
  // The caller picks the light to sample. Make sure the index stays in the bounds of the sysData.lightDefinitions array.
  const int indexLight = (1 < numLights) ? clamp(static_cast<int>(floorf(rng(thePrd->seed) * numLights)), 0, numLights - 1) : 0;
  
  const LightDefinition& light = sysData.lightDefinitions[indexLight];

  const int callLight = NUM_LENS_TYPES + light.typeLight;

  LightSample lightSample = optixDirectCall<LightSample, const LightDefinition&, PerRayData*>(callLight, light, thePrd);

  // No direct lighting if the light sample is invalid.
  if (lightSample.pdf <= 0)
  {
    return;
  }

  // Now that we have an incoming light direction, evaluate the bsdf with that. 
  // All other state is unchanged from the sampling above.
  state.k2 = lightSample.direction;
      
  const float3 bxdf = bsdf_ggx_smith_eval(state); // eval output: (glossy part of the) bsdf * dot(k2, normal)
  const float  pdf  = state.pdf;

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

  if (isVisible)
  {
    const float weightMIS = (TYPE_LIGHT_POINT <= light.typeLight) ? 1.0f : balanceHeuristic(lightSample.pdf, pdf);

    // The sampled emission needs to be scaled by the inverse probability to have selected this light,
    // Selecting one of many lights means the inverse of 1.0f / numLights.
    // This is using the path throughput before the sampling modulated it above.
    thePrd->radiance += throughput * bxdf * lightSample.radiance_over_pdf * (float(numLights) * weightMIS);
  }
}
