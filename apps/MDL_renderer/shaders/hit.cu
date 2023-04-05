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
#include "per_ray_data.h"
#include "vertex_attributes.h"
#include "function_indices.h"
#include "material_definition_mdl.h"
#include "light_definition.h"
#include "shader_common.h"
#include "transform.h"
#include "random_number_generators.h"
#include "curve.h"
#include "curve_attributes.h"

// Contained in per_ray_data.h:
//#include <mi/neuraylib/target_code_types.h>

// The MDL texture runtime functions: texture, MBSDF, light profile, and scene data (dummy) lookup functions.
// These are declared extern and can only appear in one module inside the pipeline or there will be OptiX compilation errors.
// Means all functions potentially accessing any of these MDL runtime functions must be implemented in this module.
// That's the reason why the arbitrary mesh light sampling routine is here and not in light_sample.cu
#define TEX_SUPPORT_NO_VTABLES
#define TEX_SUPPORT_NO_DUMMY_SCENEDATA
#include "texture_lookup.h"

// This renderer is not implementing support for derivatives (ray differentials).
// It only needs this Shading_state_materialy structure without derivatives support.
typedef mi::neuraylib::Shading_state_material Mdl_state;


// DEBUG Helper code.
//uint3 theLaunchIndex = optixGetLaunchIndex();
//if (theLaunchIndex.x == 256 && theLaunchIndex.y == 256)
//{
//  printf("value = %f\n", value);
//}

//thePrd->radiance += make_float3(value);
//thePrd->eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
//return;


extern "C" __constant__ SystemData sysData;

// This shader handles every supported feature of the the renderer.
extern "C" __global__ void __closesthit__radiance()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->flags |= FLAG_HIT; // Required to distinguish surface hits from random walk miss.

  thePrd->distance = optixGetRayTmax(); // Return the current path segment distance, needed for absorption calculations in the integrator.
  
  // PRECISION Calculate this from the object space vertex positions and transform to world for better accuracy when needed.
  // Same as: thePrd->pos = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
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

  GeometryInstanceData theData = sysData.geometryInstanceData[optixGetInstanceId()];
  // theData.ids: .x = idMaterial, .y = idLight, .z = idObject

  const unsigned int thePrimitiveIndex = optixGetPrimitiveIndex();

  // Cast the CUdeviceptr to the actual format of the Triangles attributes and indices.
  const uint3* indices = reinterpret_cast<uint3*>(theData.indices);
  const uint3  tri     = indices[thePrimitiveIndex];

  const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(theData.attributes);

  const TriangleAttributes& attr0 = attributes[tri.x];
  const TriangleAttributes& attr1 = attributes[tri.y];
  const TriangleAttributes& attr2 = attributes[tri.z];

  const float2 theBarycentrics = optixGetTriangleBarycentrics(); // .x = beta, .y = gamma
  const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

  float4 objectToWorld[3];
  float4 worldToObject[3];

  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject); // Single instance level transformation list only.

  // Object space vertex attributes at the hit point.
  float3 ns = attr0.normal * alpha + attr1.normal * theBarycentrics.x + attr2.normal * theBarycentrics.y;
  float3 ng = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex);
  float3 tg = attr0.tangent * alpha + attr1.tangent * theBarycentrics.x + attr2.tangent * theBarycentrics.y;

  // Transform attributes into internal space == world space.
  ns = normalize(transformNormal(worldToObject, ns));
  ng = normalize(transformNormal(worldToObject, ng));
  // This is actually the geometry tangent which for the runtime generated geometry objects
  // (plane, box, sphere, torus) match exactly with the texture space tangent.
  // FIXME Generate these from the triangle's texture derivatives instead, but that's more expensive.
  // Mind that tangents and bitangents are transformed as vectors, not normals, because they lie inside the surface's plane.
  tg = normalize(transformVector(objectToWorld, tg));
  // Calculate an ortho-normal system respective to the shading normal.
  // Expanding the TBN tbn(tg, ns) constructor because TBN members can't be used as pointers for the Mdl_state with NUM_TEXTURE_SPACES > 1.
  float3 bt = normalize(cross(ns, tg));
  tg = cross(bt, ns); // Now the tangent is orthogonal to the shading normal.

  // The Mdl_state holds the texture attributes per texture space in separate arrays.
  float3 texture_coordinates[NUM_TEXTURE_SPACES];
  float3 texture_tangents[NUM_TEXTURE_SPACES];
  float3 texture_bitangents[NUM_TEXTURE_SPACES];

  // NUM_TEXTURE_SPACES is always at least 1.
  texture_coordinates[0] = attr0.texcoord * alpha + attr1.texcoord * theBarycentrics.x + attr2.texcoord * theBarycentrics.y;
  texture_bitangents[0]  = bt;
  texture_tangents[0]    = tg;
  
#if NUM_TEXTURE_SPACES == 2
  // HACK Simply copy the vertex attributes of texture space 0, simply because there is no second texcood inside TriangleAttributes.
  texture_coordinates[1] = texture_coordinates[0];
  texture_bitangents[1]  = bt;
  texture_tangents[1]    = tg;
#endif 

  // Setup the Mdl_state.
  Mdl_state state;

  // The result of state::normal(). It represents the shading normal as determined by the renderer.
  // This field will be updated to the result of "geometry.normal" by the material or BSDF init functions,
  // if requested during code generation with set_option("include_geometry_normal", true) which is the default.
  state.normal = ns;

  // The result of state::geometry_normal().
  // It represents the geometry normal as determined by the renderer.
  state.geom_normal = ng;

  // The result of state::position().
  // It represents the position where the material should be evaluated.
  state.position = thePrd->pos;

  // The result of state::animation_time().
  // It represents the time of the current sample in seconds.
  state.animation_time = 0.0f; // This renderer implements no support for animations.

  // An array containing the results of state::texture_coordinate(i).
  // The i-th entry represents the texture coordinates of the i-th texture space at the current position.
  // Only one element here because "num_texture_spaces" option has been set to 1.
  state.text_coords = texture_coordinates;

  // An array containing the results of state::texture_tangent_u(i).
  // The i-th entry represents the texture tangent vector of the i-th texture space at the
  // current position, which points in the direction of the projection of the tangent to the
  // positive u axis of this texture space onto the plane defined by the original surface normal.
  state.tangent_u = texture_tangents;

  // An array containing the results of state::texture_tangent_v(i).
  // The i-th entry represents the texture bitangent vector of the i-th texture space at the
  // current position, which points in the general direction of the positive v axis of this
  // texture space, but is orthogonal to both the original surface normal and the tangent
  // of this texture space.
  state.tangent_v = texture_bitangents;

  // The texture results lookup table.
  // The size must match the backend set_option("num_texture_results") value.
  // Values will be modified by the init functions to avoid duplicate texture fetches 
  // and duplicate calculation of values (texture coordinate system).
  // This implementation is using the single material init function, not the individual init per distribution function.
  // PERF This influences how many things can be precalculated inside the init() function.
  // If the number of result elements in this array is lower than what is required,
  // the expressions for the remaining results will be compiled into the sample() and eval() functions
  // which will make the compilation and runtime performance slower. 
  float4 texture_results[NUM_TEXTURE_RESULTS];

  state.text_results = texture_results;

  // A pointer to a read-only data segment.
  // For "PTX", "LLVM-IR" and "native" JIT backend.
  // For other backends, this should be NULL.
  state.ro_data_segment = nullptr;

  // A 4x4 transformation matrix in row-major order transforming from world to object coordinates.
  // The last row is always implied to be (0, 0, 0, 1) and does not have to be provided.
  // It is used by the state::transform_*() methods.
  // This field is only used if the uniform state is included.
  state.world_to_object = worldToObject;

  // A 4x4 transformation matrix in row-major order transforming from object to world coordinates.
  // The last row is always implied to be (0, 0, 0, 1) and does not have to be provided.
  // It is used by the state::transform_*() methods.
  // This field is only used if the uniform state is included.
  state.object_to_world = objectToWorld;

  // The result of state::object_id().
  // It is an application-specific identifier of the hit object as provided in a scene.
  // It can be used to make instanced objects look different in spite of the same used material.
  // This field is only used if the uniform state is included.
  state.object_id = theData.ids.z; // idObject, this is the sg::Instance node ID.

  // The result of state::meters_per_scene_unit().
  // The field is only used if the "fold_meters_per_scene_unit" option is set to false.
  // Otherwise, the value of the "meters_per_scene_unit" option will be used in the code.
  state.meters_per_scene_unit = 1.0f;
    
  const MaterialDefinitionMDL& material = sysData.materialDefinitionsMDL[theData.ids.x];

  mi::neuraylib::Resource_data res_data = { nullptr, material.texture_handler };

  const DeviceShaderConfiguration& shaderConfiguration = sysData.shaderConfigurations[material.indexShader];

  // Using a single material init function instead of per distribution init functions.
  // This is always present, even if it just returns.
  optixDirectCall<void>(shaderConfiguration.idxCallInit, &state, &res_data, nullptr, material.arg_block);

  // Explicitly include edge-on cases as frontface condition!
  // Keeps the material stack from overflowing at silhouettes.
  // Prevents that silhouettes of thin-walled materials use the backface material.
  // Using the true geometry normal attribute as originally defined on the frontface!
  const bool isFrontFace = (0.0f <= dot(thePrd->wo, state.geom_normal));

  // thin_walled value in case the expression is a constant.
  bool thin_walled = ((shaderConfiguration.flags & IS_THIN_WALLED) != 0);

  if (0 <= shaderConfiguration.idxCallThinWalled)
  {
    optixDirectCall<void>(shaderConfiguration.idxCallThinWalled, &thin_walled, &state, &res_data, nullptr, material.arg_block);
  }

  // IOR value in case the material ior expression is constant.
  float3 ior = shaderConfiguration.ior;

  if (0 <= shaderConfiguration.idxCallIor)
  {
    optixDirectCall<void>(shaderConfiguration.idxCallIor, &ior, &state, &res_data, nullptr, material.arg_block);
  }

  // Handle optional surface and backface emission expressions.
  // Default to no EDF.
  int idxCallEmissionEval          = -1; 
  int idxCallEmissionIntensity     = -1;
  int idxCallEmissionIntensityMode = -1;
  // These are not used when there is no emission, no need to initialize.
  float3 emission_intensity;
  int    emission_intensity_mode;
  
  // MDL Specs: There is no emission on the back-side unless an EDF is specified with the backface field and thin_walled is set to true.
  if (isFrontFace)
  {
    idxCallEmissionEval          = shaderConfiguration.idxCallSurfaceEmissionEval;
    idxCallEmissionIntensity     = shaderConfiguration.idxCallSurfaceEmissionIntensity;
    idxCallEmissionIntensityMode = shaderConfiguration.idxCallSurfaceEmissionIntensityMode;
    
    emission_intensity      = shaderConfiguration.surface_intensity;
    emission_intensity_mode = shaderConfiguration.surface_intensity_mode;
  }
  else if (thin_walled) // && !isFrontFace
  {
    // These can be the same callable indices if the expressions from surface and backface were identical.
    idxCallEmissionEval          = shaderConfiguration.idxCallBackfaceEmissionEval;
    idxCallEmissionIntensity     = shaderConfiguration.idxCallBackfaceEmissionIntensity;
    idxCallEmissionIntensityMode = shaderConfiguration.idxCallBackfaceEmissionIntensityMode;

    emission_intensity      = shaderConfiguration.backface_intensity;
    emission_intensity_mode = shaderConfiguration.backface_intensity_mode;
  }

  // Check if the hit geometry contains any emission.
  if (0 <= idxCallEmissionEval) 
  {
    if (0 <= idxCallEmissionIntensity) // Emission intensity is not a constant.
    {
      optixDirectCall<void>(idxCallEmissionIntensity, &emission_intensity, &state, &res_data, nullptr, material.arg_block);
    }
    if (0 <= idxCallEmissionIntensityMode) // Emission intensity mode is not a constant.
    {
      optixDirectCall<void>(idxCallEmissionIntensityMode, &emission_intensity_mode, &state, &res_data, nullptr, material.arg_block);
    }
    if (isNotNull(emission_intensity))
    {
      mi::neuraylib::Edf_evaluate_data<mi::neuraylib::DF_HSM_NONE> eval_data;

      eval_data.k1 = thePrd->wo; // input: outgoing direction (-ray.direction)
      //eval_data.cos : output: dot(normal, k1)
      //eval_data.edf : output: edf
      //eval_data.pdf : output: pdf (non-projected hemisphere)

      optixDirectCall<void>(idxCallEmissionEval, &eval_data, &state, &res_data, nullptr, material.arg_block);

      const float area = sysData.lightDefinitions[theData.ids.y].area; // This must be a mesh light, and then it has a valid idLight.

      eval_data.pdf = thePrd->distance * thePrd->distance / (area * eval_data.cos); // Solid angle measure.

      float weightMIS = 1.0f;
      // If the last event was diffuse or glossy, calculate the opposite MIS weight for this implicit light hit.
      if (sysData.directLighting && (thePrd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY)))
      {
        weightMIS = balanceHeuristic(thePrd->pdf, eval_data.pdf);
      }

      // Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
      const float factor = (emission_intensity_mode == 0) ? 1.0f : 1.0f / area;
      
      thePrd->radiance += thePrd->throughput * emission_intensity * eval_data.edf * (factor * weightMIS);
    }
  }

  // Start fresh with the next BSDF sample.
  // Save the current path throughput for the direct lighting contribution.
  // The path throughput will be modulated with the BSDF sampling results before that.
  const float3 throughput = thePrd->throughput;
  // The pdf of the previous event was needed for the emission calculation above.
  thePrd->pdf = 0.0f;

  // Determine which BSDF to use when the material is thin-walled. 
  int idxCallScatteringSample = shaderConfiguration.idxCallSurfaceScatteringSample;
  int idxCallScatteringEval   = shaderConfiguration.idxCallSurfaceScatteringEval;

  // thin-walled and looking at the backface and backface.scattering expression available?
  if (thin_walled && !isFrontFace && 0 <= shaderConfiguration.idxCallBackfaceScatteringSample)
  {
    // Use the backface.scattering BSDF sample and evaluation functions.
    // Apparently the MDL code can handle front- and backfacing calculations appropriately with the original state and the properly setup volume IORs.
    // No need to flip normals to the ray side.
    idxCallScatteringSample = shaderConfiguration.idxCallBackfaceScatteringSample;
    idxCallScatteringEval   = shaderConfiguration.idxCallBackfaceScatteringEval; // Assumes both are valid.
  }

  // Importance sample the BSDF. 
  if (0 <= idxCallScatteringSample)
  {
    mi::neuraylib::Bsdf_sample_data sample_data;

    int idx = thePrd->idxStack;

    // If the hit is either on the surface or a thin-walled material,
    // the ray is inside the surrounding material and the material ior is on the other side.
    if (isFrontFace || thin_walled)
    {
      sample_data.ior1 = thePrd->stack[idx].ior; // From surrounding medium ior
      sample_data.ior2 = ior;                    // to material ior.
    }
    else 
    {
      // When hitting the backface of a non-thin-walled material, 
      // the ray is inside the current material and the surrounding material is on the other side.
      // The material's IOR is the current top-of-stack. We need the one further down!
      idx = max(0, idx - 1);

      sample_data.ior1 = ior;                    // From material ior 
      sample_data.ior2 = thePrd->stack[idx].ior; // to surrounding medium ior
    }
    sample_data.k1 = thePrd->wo; // == -optixGetWorldRayDirection()
    sample_data.xi = rng4(thePrd->seed);

    optixDirectCall<void>(idxCallScatteringSample, &sample_data, &state, &res_data, nullptr, material.arg_block);

    thePrd->wi          = sample_data.k2;            // Continuation direction.
    thePrd->throughput *= sample_data.bsdf_over_pdf; // Adjust the path throughput for all following incident lighting.
    thePrd->pdf         = sample_data.pdf;           // Note that specular events return pdf == 0.0f! (=> Not a path termination condition.)
    thePrd->eventType   = sample_data.event_type;    // This replaces the PRD flags used inside the other examples.
  }
  else
  {
    // If there is no valid scattering BSDF, it's the black bsdf() which ends the path.
    // This is usually happening with arbitrary mesh lights when only specifying emission.
    thePrd->eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
    // None of the following code will have any effect in that case.
    return;
  }

  // Direct lighting if the sampled BSDF was diffuse and any light is in the scene.
  const int numLights = sysData.numLights;

  if (sysData.directLighting && 0 < numLights && (thePrd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY)))
  {
    // Sample one of many lights.
    // The caller picks the light to sample. Make sure the index stays in the bounds of the sysData.lightDefinitions array.
    const int indexLight = (1 < numLights) ? clamp(static_cast<int>(floorf(rng(thePrd->seed) * numLights)), 0, numLights - 1) : 0;
    
    const LightDefinition& light = sysData.lightDefinitions[indexLight];
    
    LightSample lightSample = optixDirectCall<LightSample, const LightDefinition&, PerRayData*>(NUM_LENS_TYPES + light.typeLight, light, thePrd);

    if (0.0f < lightSample.pdf && 0 <= idxCallScatteringEval)
    {
      mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> eval_data;

      int idx = thePrd->idxStack;
      
      if (isFrontFace || thin_walled)
      {
        eval_data.ior1 = thePrd->stack[idx].ior;
        eval_data.ior2 = ior;
      }
      else
      {
        idx = max(0, idx - 1);

        eval_data.ior1 = ior;
        eval_data.ior2 = thePrd->stack[idx].ior;
      }
      
      eval_data.k1 = thePrd->wo;
      eval_data.k2 = lightSample.direction;

      optixDirectCall<void>(idxCallScatteringEval, &eval_data, &state, &res_data, nullptr, material.arg_block);

      // This already contains the fabsf(dot(lightSample.direction, state.normal)) factor!
      // For a white Lambert material, the bxdf components match the eval_data.pdf
      const float3 bxdf = eval_data.bsdf_diffuse + eval_data.bsdf_glossy;

      if (0.0f < eval_data.pdf && isNotNull(bxdf))
      {
        // Pass the current payload registers through to the shadow ray.
        unsigned int p0 = optixGetPayload_0();
        unsigned int p1 = optixGetPayload_1();

        thePrd->flags &= ~FLAG_SHADOW; // Clear the shadow flag.

        // Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval 
        // to prevent self-intersections with the actual light geometry in the scene.
        optixTrace(sysData.topObject,
                   thePrd->pos, lightSample.direction, // origin, direction
                   sysData.sceneEpsilon, lightSample.distance - sysData.sceneEpsilon, 0.0f, // tmin, tmax, time
                   OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // The shadow ray type only uses anyhit programs.
                   TYPE_RAY_SHADOW, NUM_RAY_TYPES, TYPE_RAY_SHADOW,
                   p0, p1); // Pass through thePrd to the shadow ray.

        if ((thePrd->flags & FLAG_SHADOW) == 0) // Shadow flag not set?
        {
          const float weightMIS = (TYPE_LIGHT_POINT <= light.typeLight) ? 1.0f : balanceHeuristic(lightSample.pdf, eval_data.pdf);
          
          // The sampled emission needs to be scaled by the inverse probability to have selected this light,
          // Selecting one of many lights means the inverse of 1.0f / numLights.
          // This is using the path throughput before the sampling modulated it above.
          thePrd->radiance += throughput * bxdf * lightSample.radiance_over_pdf * (float(numLights) * weightMIS);
        }
      } 
    }
  }

  // Now after everything has been handled using the current material stack,
  // adjust the material stack if there was a transmission crossing a boundary surface.
  if (!thin_walled && (thePrd->eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
  {
    if (isFrontFace) // Entered a volume. 
    {
      float3 absorption = shaderConfiguration.absorption_coefficient;
      if (0 < shaderConfiguration.idxCallVolumeAbsorptionCoefficient)
      {
        optixDirectCall<void>(shaderConfiguration.idxCallVolumeAbsorptionCoefficient, &absorption, &state, &res_data, nullptr, material.arg_block);
      }

      float3 scattering = shaderConfiguration.scattering_coefficient;
      if (0 < shaderConfiguration.idxCallVolumeScatteringCoefficient)
      {
        optixDirectCall<void>(shaderConfiguration.idxCallVolumeScatteringCoefficient, &scattering, &state, &res_data, nullptr, material.arg_block);
      }

      float bias = shaderConfiguration.directional_bias;
      if (0 < shaderConfiguration.idxCallVolumeDirectionalBias)
      {
        optixDirectCall<void>(shaderConfiguration.idxCallVolumeDirectionalBias, &bias, &state, &res_data, nullptr, material.arg_block);
      }

      const int idx = min(thePrd->idxStack + 1, MATERIAL_STACK_LAST); // Push current medium parameters.

      thePrd->idxStack = idx;
      thePrd->stack[idx].ior     = ior;
      thePrd->stack[idx].sigma_a = absorption;
      thePrd->stack[idx].sigma_s = scattering;
      thePrd->stack[idx].bias    = bias;
      
      thePrd->sigma_t = absorption + scattering; // Update the current extinction coefficient.
    }
    else // if !isFrontFace. Left a volume.
    {
      const int idx = max(0, thePrd->idxStack - 1); // Pop current medium parameters.

      thePrd->idxStack = idx; 

      thePrd->sigma_t = thePrd->stack[idx].sigma_a + thePrd->stack[idx].sigma_s; // Update the current extinction coefficient.
    }
   
    thePrd->walk = 0; // Reset the number of random walk steps taken when crossing any volume boundary.
  }
}


// PERF Identical to radiance shader above, but used for materials without emission, which is the majority of materials.
extern "C" __global__ void __closesthit__radiance_no_emission()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->flags |= FLAG_HIT; // Required to distinguish surface hits from random walk miss.

  thePrd->distance = optixGetRayTmax(); // Return the current path segment distance, needed for absorption calculations in the integrator.
  
  // PRECISION Calculate this from the object space vertex positions and transform to world for better accuracy when needed.
  // Same as: thePrd->pos = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
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

  GeometryInstanceData theData = sysData.geometryInstanceData[optixGetInstanceId()];
  // theData.ids: .x = idMaterial, .y = idLight, .z = idObject

  const unsigned int thePrimitiveIndex = optixGetPrimitiveIndex();

  // Cast the CUdeviceptr to the actual format of the Triangles attributes and indices.
  const uint3* indices = reinterpret_cast<uint3*>(theData.indices);
  const uint3  tri     = indices[thePrimitiveIndex];

  const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(theData.attributes);

  const TriangleAttributes& attr0 = attributes[tri.x];
  const TriangleAttributes& attr1 = attributes[tri.y];
  const TriangleAttributes& attr2 = attributes[tri.z];

  const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
  const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

  float4 objectToWorld[3];
  float4 worldToObject[3];

  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject); // Single instance level transformation list only.

  // Object space vertex attributes at the hit point.
  float3 ns = attr0.normal * alpha + attr1.normal * theBarycentrics.x + attr2.normal * theBarycentrics.y;
  float3 ng = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex);
  float3 tg = attr0.tangent * alpha + attr1.tangent * theBarycentrics.x + attr2.tangent * theBarycentrics.y;

  // Transform attributes into internal space == world space.
  ns = normalize(transformNormal(worldToObject, ns));
  ng = normalize(transformNormal(worldToObject, ng));
  tg = normalize(transformVector(objectToWorld, tg));
  // Calculate an ortho-normal system respective to the shading normal.
  float3 bt = normalize(cross(ns, tg));
  tg = cross(bt, ns); // Now the tangent is orthogonal to the shading normal.

  // The Mdl_state holds the texture attributes per texture space in separate arrays.
  float3 texture_coordinates[NUM_TEXTURE_SPACES];
  float3 texture_tangents[NUM_TEXTURE_SPACES];
  float3 texture_bitangents[NUM_TEXTURE_SPACES];

  // NUM_TEXTURE_SPACES is always at least 1.
  texture_coordinates[0] = attr0.texcoord * alpha + attr1.texcoord * theBarycentrics.x + attr2.texcoord * theBarycentrics.y;
  texture_bitangents[0]  = bt;
  texture_tangents[0]    = tg;
  
#if NUM_TEXTURE_SPACES == 2
  // HACK Simply copy the vertex attributes of texture space 0, simply because there is no second texcood inside TriangleAttributes.
  texture_coordinates[1] = texture_coordinates[0];
  texture_bitangents[1]  = bt;
  texture_tangents[1]    = tg;
#endif 

  // Setup the Mdl_state.
  Mdl_state state;

  float4 texture_results[NUM_TEXTURE_RESULTS];

  // For explanations of these fields see comments inside __closesthit__radiance above.
  state.normal                = ns;
  state.geom_normal           = ng;
  state.position              = thePrd->pos;
  state.animation_time        = 0.0f;
  state.text_coords           = texture_coordinates;
  state.tangent_u             = texture_tangents;
  state.tangent_v             = texture_bitangents;
  state.text_results          = texture_results;
  state.ro_data_segment       = nullptr;
  state.world_to_object       = worldToObject;
  state.object_to_world       = objectToWorld;
  state.object_id             = theData.ids.z;
  state.meters_per_scene_unit = 1.0f;
    
  const MaterialDefinitionMDL& material = sysData.materialDefinitionsMDL[theData.ids.x];

  mi::neuraylib::Resource_data res_data = { nullptr, material.texture_handler };

  const DeviceShaderConfiguration& shaderConfiguration = sysData.shaderConfigurations[material.indexShader];

  // Using a single material init function instead of per distribution init functions.
  // This is always present, even if it just returns.
  optixDirectCall<void>(shaderConfiguration.idxCallInit, &state, &res_data, nullptr, material.arg_block);

  // Explicitly include edge-on cases as frontface condition!
  // Keeps the material stack from overflowing at silhouettes.
  // Prevents that silhouettes of thin-walled materials use the backface material.
  // Using the true geometry normal attribute as originally defined on the frontface!
  const bool isFrontFace = (0.0f <= dot(thePrd->wo, state.geom_normal));

  // thin_walled value in case the expression is a constant.
  bool thin_walled = ((shaderConfiguration.flags & IS_THIN_WALLED) != 0);

  if (0 <= shaderConfiguration.idxCallThinWalled)
  {
    optixDirectCall<void>(shaderConfiguration.idxCallThinWalled, &thin_walled, &state, &res_data, nullptr, material.arg_block);
  }

  // IOR value in case the material ior expression is constant.
  float3 ior = shaderConfiguration.ior;

  if (0 <= shaderConfiguration.idxCallIor)
  {
    optixDirectCall<void>(shaderConfiguration.idxCallIor, &ior, &state, &res_data, nullptr, material.arg_block);
  }

  // Start fresh with the next BSDF sample.
  // Save the current path throughput for the direct lighting contribution.
  // The path throughput will be modulated with the BSDF sampling results before that.
  const float3 throughput = thePrd->throughput;
  // The pdf of the previous event was needed for the emission calculation above.
  thePrd->pdf = 0.0f;

  // Determine which BSDF to use when the material is thin-walled. 
  int idxCallScatteringSample = shaderConfiguration.idxCallSurfaceScatteringSample;
  int idxCallScatteringEval   = shaderConfiguration.idxCallSurfaceScatteringEval;

  // thin-walled and looking at the backface and backface.scattering expression available?
  if (thin_walled && !isFrontFace && 0 <= shaderConfiguration.idxCallBackfaceScatteringSample)
  {
    // Use the backface.scattering BSDF sample and evaluation functions.
    // Apparently the MDL code can handle front- and backfacing calculations appropriately with the original state and the properly setup volume IORs.
    // No need to flip normals to the ray side.
    idxCallScatteringSample = shaderConfiguration.idxCallBackfaceScatteringSample;
    idxCallScatteringEval   = shaderConfiguration.idxCallBackfaceScatteringEval; // Assumes both are valid.
  }

  // Importance sample the BSDF. 
  if (0 <= idxCallScatteringSample)
  {
    mi::neuraylib::Bsdf_sample_data sample_data;

    int idx = thePrd->idxStack;

    // If the hit is either on the surface or a thin-walled material,
    // the ray is inside the surrounding material and the material ior is on the other side.
    if (isFrontFace || thin_walled)
    {
      sample_data.ior1 = thePrd->stack[idx].ior; // From surrounding medium ior
      sample_data.ior2 = ior;                    // to material ior.
    }
    else 
    {
      // When hitting the backface of a non-thin-walled material, 
      // the ray is inside the current material and the surrounding material is on the other side.
      // The material's IOR is the current top-of-stack. We need the one further down!
      idx = max(0, idx - 1);

      sample_data.ior1 = ior;                    // From material ior 
      sample_data.ior2 = thePrd->stack[idx].ior; // to surrounding medium ior
    }
    sample_data.k1 = thePrd->wo; // == -optixGetWorldRayDirection()
    sample_data.xi = rng4(thePrd->seed);

    optixDirectCall<void>(idxCallScatteringSample, &sample_data, &state, &res_data, nullptr, material.arg_block);

    thePrd->wi          = sample_data.k2;            // Continuation direction.
    thePrd->throughput *= sample_data.bsdf_over_pdf; // Adjust the path throughput for all following incident lighting.
    thePrd->pdf         = sample_data.pdf;           // Note that specular events return pdf == 0.0f! (=> Not a path termination condition.)
    thePrd->eventType   = sample_data.event_type;    // This replaces the PRD flags used inside the other examples.
  }
  else
  {
    // If there is no valid scattering BSDF, it's the black bsdf() which ends the path.
    // This is usually happening with arbitrary mesh lights when only specifying emission.
    thePrd->eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
    // None of the following code will have any effect in that case.
    return;
  }

  // Direct lighting if the sampled BSDF was diffuse and any light is in the scene.
  const int numLights = sysData.numLights;

  if (sysData.directLighting && 0 < numLights && (thePrd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY)))
  {
    // Sample one of many lights.
    // The caller picks the light to sample. Make sure the index stays in the bounds of the sysData.lightDefinitions array.
    const int indexLight = (1 < numLights) ? clamp(static_cast<int>(floorf(rng(thePrd->seed) * numLights)), 0, numLights - 1) : 0;
    
    const LightDefinition& light = sysData.lightDefinitions[indexLight];
    
    LightSample lightSample = optixDirectCall<LightSample, const LightDefinition&, PerRayData*>(NUM_LENS_TYPES + light.typeLight, light, thePrd);

    if (0.0f < lightSample.pdf && 0 <= idxCallScatteringEval)
    {
      mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> eval_data;

      int idx = thePrd->idxStack;
      
      if (isFrontFace || thin_walled)
      {
        eval_data.ior1 = thePrd->stack[idx].ior;
        eval_data.ior2 = ior;
      }
      else
      {
        idx = max(0, idx - 1);

        eval_data.ior1 = ior;
        eval_data.ior2 = thePrd->stack[idx].ior;
      }
      
      eval_data.k1 = thePrd->wo;
      eval_data.k2 = lightSample.direction;

      optixDirectCall<void>(idxCallScatteringEval, &eval_data, &state, &res_data, nullptr, material.arg_block);

      // This already contains the fabsf(dot(lightSample.direction, state.normal)) factor!
      // For a white Lambert material, the bxdf components match the eval_data.pdf
      const float3 bxdf = eval_data.bsdf_diffuse + eval_data.bsdf_glossy;

      if (0.0f < eval_data.pdf && isNotNull(bxdf))
      {
        // Pass the current payload registers through to the shadow ray.
        unsigned int p0 = optixGetPayload_0();
        unsigned int p1 = optixGetPayload_1();

        thePrd->flags &= ~FLAG_SHADOW; // Clear the shadow flag.

        // Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval 
        // to prevent self-intersections with the actual light geometry in the scene.
        optixTrace(sysData.topObject,
                   thePrd->pos, lightSample.direction, // origin, direction
                   sysData.sceneEpsilon, lightSample.distance - sysData.sceneEpsilon, 0.0f, // tmin, tmax, time
                   OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // The shadow ray type only uses anyhit programs.
                   TYPE_RAY_SHADOW, NUM_RAY_TYPES, TYPE_RAY_SHADOW,
                   p0, p1); // Pass through thePrd to the shadow ray.

        if ((thePrd->flags & FLAG_SHADOW) == 0) // Shadow flag not set?
        {
          const float weightMIS = (TYPE_LIGHT_POINT <= light.typeLight) ? 1.0f : balanceHeuristic(lightSample.pdf, eval_data.pdf);
          
          // The sampled emission needs to be scaled by the inverse probability to have selected this light,
          // Selecting one of many lights means the inverse of 1.0f / numLights.
          // This is using the path throughput before the sampling modulated it above.
          thePrd->radiance += throughput * bxdf * lightSample.radiance_over_pdf * (float(numLights) * weightMIS);
        }
      } 
    }
  }

  // Now after everything has been handled using the current material stack,
  // adjust the material stack if there was a transmission crossing a boundary surface.
  if (!thin_walled && (thePrd->eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
  {
    if (isFrontFace) // Entered a volume. 
    {
      float3 absorption = shaderConfiguration.absorption_coefficient;
      if (0 < shaderConfiguration.idxCallVolumeAbsorptionCoefficient)
      {
        optixDirectCall<void>(shaderConfiguration.idxCallVolumeAbsorptionCoefficient, &absorption, &state, &res_data, nullptr, material.arg_block);
      }

      float3 scattering = shaderConfiguration.scattering_coefficient;
      if (0 < shaderConfiguration.idxCallVolumeScatteringCoefficient)
      {
        optixDirectCall<void>(shaderConfiguration.idxCallVolumeScatteringCoefficient, &scattering, &state, &res_data, nullptr, material.arg_block);
      }

      float bias = shaderConfiguration.directional_bias;
      if (0 < shaderConfiguration.idxCallVolumeDirectionalBias)
      {
        optixDirectCall<void>(shaderConfiguration.idxCallVolumeDirectionalBias, &bias, &state, &res_data, nullptr, material.arg_block);
      }

      const int idx = min(thePrd->idxStack + 1, MATERIAL_STACK_LAST); // Push current medium parameters.

      thePrd->idxStack = idx;
      thePrd->stack[idx].ior     = ior;
      thePrd->stack[idx].sigma_a = absorption;
      thePrd->stack[idx].sigma_s = scattering;
      thePrd->stack[idx].bias    = bias;
      
      thePrd->sigma_t = absorption + scattering; // Update the current extinction coefficient.
    }
    else // if !isFrontFace. Left a volume.
    {
      const int idx = max(0, thePrd->idxStack - 1); // Pop current medium parameters.

      thePrd->idxStack = idx; 

      thePrd->sigma_t = thePrd->stack[idx].sigma_a + thePrd->stack[idx].sigma_s; // Update the current extinction coefficient.
    }
   
    thePrd->walk = 0; // Reset the number of random walk steps taken when crossing any volume boundary.
  }
}


// One anyhit program for the radiance ray for all materials with cutout opacity!
extern "C" __global__ void __anyhit__radiance_cutout()
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

  float4 objectToWorld[3];
  float4 worldToObject[3];

  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject); // Single instance level transformation list only.

  // Object space vertex attributes at the hit point.
  float3 po = attr0.vertex   * alpha + attr1.vertex   * theBarycentrics.x + attr2.vertex   * theBarycentrics.y;

  float3 ns = attr0.normal * alpha + attr1.normal * theBarycentrics.x + attr2.normal * theBarycentrics.y;
  float3 ng = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex);
  float3 tg = attr0.tangent * alpha + attr1.tangent * theBarycentrics.x + attr2.tangent * theBarycentrics.y;

  // Transform attributes into internal space == world space.
  po = transformPoint(objectToWorld,  po);
  ns = normalize(transformNormal(worldToObject, ns));
  ng = normalize(transformNormal(worldToObject, ng));
  // This is actually the geometry tangent which for the runtime generated geometry objects
  // (plane, box, sphere, torus) match exactly with the texture space tangent.
  // FIXME Generate these from the triangle's texture derivatives instead, but that's more expensive.
  // Mind that tangents and bitangents are transformed as vectors, not normals, because they lie inside the surface's plane.
  tg = normalize(transformVector(objectToWorld, tg));
  // Calculate an ortho-normal system respective to the shading normal.
  // Expanding the TBN tbn(tg, ns) constructor because TBN members can't be used as pointers for the Mdl_state with NUM_TEXTURE_SPACES > 1.
  float3 bt = normalize(cross(ns, tg));
  tg = cross(bt, ns); // Now the tangent is orthogonal to the shading normal.

  // The Mdl_state holds the texture attributes per texture space in separate arrays.
  float3 texture_coordinates[NUM_TEXTURE_SPACES];
  float3 texture_tangents[NUM_TEXTURE_SPACES];
  float3 texture_bitangents[NUM_TEXTURE_SPACES];

  // NUM_TEXTURE_SPACES is always at least 1.
  texture_coordinates[0] = attr0.texcoord * alpha + attr1.texcoord * theBarycentrics.x + attr2.texcoord * theBarycentrics.y;
  texture_bitangents[0]  = bt;
  texture_tangents[0]    = tg;
  
#if NUM_TEXTURE_SPACES == 2
  // HACK Simply copy the vertex attributes of texture space 0, simply because there is no second texcood inside TriangleAttributes.
  texture_coordinates[1] = texture_coordinates[0];
  texture_bitangents[1]  = bt;
  texture_tangents[1]    = tg;
#endif 

  // Setup the Mdl_state.
  Mdl_state state;

  float4 texture_results[NUM_TEXTURE_RESULTS];

  // For explanations of these fields see comments inside __closesthit__radiance above.
  state.normal                = ns;
  state.geom_normal           = ng;
  state.position              = po;
  state.animation_time        = 0.0f;
  state.text_coords           = texture_coordinates;
  state.tangent_u             = texture_tangents;
  state.tangent_v             = texture_bitangents;
  state.text_results          = texture_results;
  state.ro_data_segment       = nullptr;
  state.world_to_object       = worldToObject;
  state.object_to_world       = objectToWorld;
  state.object_id             = theData.ids.z;
  state.meters_per_scene_unit = 1.0f;

  const MaterialDefinitionMDL& material = sysData.materialDefinitionsMDL[theData.ids.x];

  mi::neuraylib::Resource_data res_data = { nullptr, material.texture_handler };

  // The cutout opacity value needs to be determined based on the ShaderConfiguration data and geometry.cutout expression when needed.
  const DeviceShaderConfiguration& shaderConfiguration = sysData.shaderConfigurations[material.indexShader];

  // Using a single material init function instead of per distribution init functions.
  // PERF See how that affects cutout opacity which only needs the geometry.cutout expression.
  float opacity = shaderConfiguration.cutout_opacity;

  if (0 <= shaderConfiguration.idxCallGeometryCutoutOpacity)
  {
    // This is always present, even if it just returns.
    optixDirectCall<void>(shaderConfiguration.idxCallInit, &state, &res_data, nullptr, material.arg_block);

    optixDirectCall<void>(shaderConfiguration.idxCallGeometryCutoutOpacity, &opacity, &state, &res_data, nullptr, material.arg_block);
  }

  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  // Stochastic alpha test to get an alpha blend effect.
  // No need to calculate an expensive random number if the test is going to fail anyway.
  if (opacity < 1.0f && opacity <= rng(thePrd->seed))
  {
    optixIgnoreIntersection();
  }
}


// The shadow ray program for all materials with no cutout opacity.
extern "C" __global__ void __anyhit__shadow()
{
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  // Always set payload values before calling optixIgnoreIntersection or optixTerminateRay because they return immediately!
  thePrd->flags |= FLAG_SHADOW; // Visbility check failed.

  optixTerminateRay();
}


extern "C" __global__ void __anyhit__shadow_cutout() // For the radiance ray type.
{
  GeometryInstanceData theData = sysData.geometryInstanceData[optixGetInstanceId()];

  const unsigned int thePrimitiveIndex = optixGetPrimitiveIndex();

  const uint3* indices = reinterpret_cast<uint3*>(theData.indices);
  const uint3  tri     = indices[thePrimitiveIndex];

  // Cast the CUdeviceptr to the actual format for Triangles geometry.
  const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(theData.attributes);

  const TriangleAttributes& attr0 = attributes[tri.x];
  const TriangleAttributes& attr1 = attributes[tri.y];
  const TriangleAttributes& attr2 = attributes[tri.z];

  const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
  const float  alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

  float4 objectToWorld[3];
  float4 worldToObject[3];

  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject); // Single instance level transformation list only.

  // Object space vertex attributes at the hit point.
  float3 po = attr0.vertex * alpha + attr1.vertex * theBarycentrics.x + attr2.vertex * theBarycentrics.y;
  float3 ns = attr0.normal * alpha + attr1.normal * theBarycentrics.x + attr2.normal * theBarycentrics.y;
  float3 ng = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex);
  float3 tg = attr0.tangent * alpha + attr1.tangent * theBarycentrics.x + attr2.tangent * theBarycentrics.y;

  // Transform attributes into internal space == world space.
  po = transformPoint(objectToWorld, po);
  ns = normalize(transformNormal(worldToObject, ns));
  ng = normalize(transformNormal(worldToObject, ng));
  // This is actually the geometry tangent which for the runtime generated geometry objects
  // (plane, box, sphere, torus) match exactly with the texture space tangent.
  // FIXME Generate these from the triangle's texture derivatives instead, but that's more expensive.
  // Mind that tangents and bitangents are transformed as vectors, not normals, because they lie inside the surface's plane.
  tg = normalize(transformVector(objectToWorld, tg));
  // Calculate an ortho-normal system respective to the shading normal.
  // Expanding the TBN tbn(tg, ns) constructor because TBN members can't be used as pointers for the Mdl_state with NUM_TEXTURE_SPACES > 1.
  float3 bt = normalize(cross(ns, tg));
  tg = cross(bt, ns); // Now the tangent is orthogonal to the shading normal.

  // The Mdl_state holds the texture attributes per texture space in separate arrays.
  float3 texture_coordinates[NUM_TEXTURE_SPACES];
  float3 texture_tangents[NUM_TEXTURE_SPACES];
  float3 texture_bitangents[NUM_TEXTURE_SPACES];

  // NUM_TEXTURE_SPACES is always at least 1.
  texture_coordinates[0] = attr0.texcoord * alpha + attr1.texcoord * theBarycentrics.x + attr2.texcoord * theBarycentrics.y;
  texture_bitangents[0]  = bt;
  texture_tangents[0]    = tg;
  
#if NUM_TEXTURE_SPACES == 2
  // HACK Simply copy the vertex attributes of texture space 0, simply because there is no second texcood inside TriangleAttributes.
  texture_coordinates[1] = texture_coordinates[0];
  texture_bitangents[1]  = bt;
  texture_tangents[1]    = tg;
#endif 

  // Setup the Mdl_state.
  Mdl_state state;

  float4 texture_results[NUM_TEXTURE_RESULTS];

  // For explanations of these fields see comments inside __closesthit__radiance above.
  state.normal                = ns;
  state.geom_normal           = ng;
  state.position              = po;
  state.animation_time        = 0.0f;
  state.text_coords           = texture_coordinates;
  state.tangent_u             = texture_tangents;
  state.tangent_v             = texture_bitangents;
  state.text_results          = texture_results;
  state.ro_data_segment       = nullptr;
  state.world_to_object       = worldToObject;
  state.object_to_world       = objectToWorld;
  state.object_id             = theData.ids.z;
  state.meters_per_scene_unit = 1.0f;

  const MaterialDefinitionMDL& material = sysData.materialDefinitionsMDL[theData.ids.x];

  mi::neuraylib::Resource_data res_data = { nullptr, material.texture_handler };

  // The cutout opacity value needs to be determined based on the ShaderConfiguration data and geometry.cutout expression when needed.
  const DeviceShaderConfiguration& shaderConfiguration = sysData.shaderConfigurations[material.indexShader];

  float opacity = shaderConfiguration.cutout_opacity;

  if (0 <= shaderConfiguration.idxCallGeometryCutoutOpacity)
  {
    optixDirectCall<void>(shaderConfiguration.idxCallInit, &state, &res_data, nullptr, material.arg_block);

    optixDirectCall<void>(shaderConfiguration.idxCallGeometryCutoutOpacity, &opacity, &state, &res_data, nullptr, material.arg_block);
  }

  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  // Stochastic alpha test to get an alpha blend effect.
  // No need to calculate an expensive random number if the test is going to fail anyway.
  if (opacity < 1.0f && opacity <= rng(thePrd->seed))
  {
    optixIgnoreIntersection();
  }
  else
  {
    // Always set payload values before calling optixIgnoreIntersection or optixTerminateRay because they return immediately!
    thePrd->flags |= FLAG_SHADOW;

    optixTerminateRay();
  }
}


// Explicit light sampling of a triangle mesh geometry with an emissive MDL material.
// Defined here to be able to use the MDL runtime functions included via texture_lookup.h.
extern "C" __device__ LightSample __direct_callable__light_mesh(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;
 
  lightSample.pdf = 0.0f;

  const float3 sampleTriangle = rng3(prd->seed);

  // Uniformly sample the triangles over their surface area.
  // Note that zero-area triangles (e.g. at the poles of spheres) are automatically never sampled with this method!
  // The cdfU is one bigger than light.width.
  const float* cdfArea = reinterpret_cast<const float*>(light.cdfU);
  const unsigned int idxTriangle = binarySearchCDF(cdfArea, light.width, sampleTriangle.z);

  // Unit square to triangle via barycentric coordinates.
  const float su = sqrtf(sampleTriangle.x);
  // Barycentric coordinates.
  const float alpha = 1.0f - su;
  const float beta  = sampleTriangle.y * su;
  const float gamma = 1.0f - alpha - beta; 
  
  // This cast works because both unsigned int and uint3 have an alignment of 4 bytes.
  const uint3* indices = reinterpret_cast<uint3*>(light.indices);
  const uint3  tri     = indices[idxTriangle];
  
  const TriangleAttributes* attributes = reinterpret_cast<TriangleAttributes*>(light.attributes);

  const TriangleAttributes& attr0 = attributes[tri.x];
  const TriangleAttributes& attr1 = attributes[tri.y];
  const TriangleAttributes& attr2 = attributes[tri.z];

  // Object space vertex attributes at the hit point.
  float3 po = attr0.vertex * alpha + attr1.vertex * beta + attr2.vertex * gamma;

  // Transform attributes into internal space == world space.
  po = transformPoint(light.matrix, po);

  // Calculate the outgoing direction from light sample position to surface point.
  lightSample.direction = po - prd->pos;  // Sample direction from surface point to light sample position.
  lightSample.distance  = length(lightSample.direction);
    
  if (lightSample.distance < DENOMINATOR_EPSILON)
  {
    return lightSample;
  }

  lightSample.direction *= 1.0f / lightSample.distance; // Normalized vector from light sample position to surface point.

  float3 ns = attr0.normal * alpha + attr1.normal * beta + attr2.normal * gamma;
  float3 ng = cross(attr1.vertex - attr0.vertex, attr2.vertex - attr0.vertex);
  float3 tg = attr0.tangent * alpha + attr1.tangent * beta + attr2.tangent * gamma;

  // Transform attributes into internal space == world space.
  ns = normalize(transformNormal(light.matrixInv, ns));
  ng = normalize(transformNormal(light.matrixInv, ng));
  tg = normalize(transformVector(light.matrix,    tg));

  float3 bt = normalize(cross(ns, tg));
  tg        = cross(bt, ns); // Now the tangent is orthogonal to the shading normal.

  // The Mdl_state holds the texture attributes per texture space in separate arrays.
  float3 texture_coordinates[NUM_TEXTURE_SPACES];
  float3 texture_tangents[NUM_TEXTURE_SPACES];
  float3 texture_bitangents[NUM_TEXTURE_SPACES];

  // NUM_TEXTURE_SPACES is always at least 1.
  texture_coordinates[0] = attr0.texcoord * alpha + attr1.texcoord * beta + attr2.texcoord * gamma;
  texture_bitangents[0]  = bt;
  texture_tangents[0]    = tg;
  
#if NUM_TEXTURE_SPACES == 2
  // HACK Simply copy the vertex attributes of texture space 0, simply because there is no second texcood inside TriangleAttributes.
  texture_coordinates[1] = texture_coordinates[0];
  texture_bitangents[1]  = bt;
  texture_tangents[1]    = tg;
#endif 

  // Setup the Mdl_state.
  Mdl_state state;

  float4 texture_results[NUM_TEXTURE_RESULTS];

  state.normal                = ns;
  state.geom_normal           = ng;
  state.position              = po;
  state.animation_time        = 0.0f;
  state.text_coords           = texture_coordinates;
  state.tangent_u             = texture_tangents;
  state.tangent_v             = texture_bitangents;
  state.text_results          = texture_results;
  state.ro_data_segment       = nullptr;
  state.world_to_object       = light.matrixInv;
  state.object_to_world       = light.matrix;
  state.object_id             = light.idObject;
  state.meters_per_scene_unit = 1.0f;

  const MaterialDefinitionMDL& material = sysData.materialDefinitionsMDL[light.idMaterial];

  mi::neuraylib::Resource_data res_data = { nullptr, material.texture_handler };

  const DeviceShaderConfiguration& shaderConfiguration = sysData.shaderConfigurations[material.indexShader];

  // This is always present, even if it just returns.
  optixDirectCall<void>(shaderConfiguration.idxCallInit, &state, &res_data, nullptr, material.arg_block);

  // Arbitrary mesh lights can have cutout opacity!
  float opacity = shaderConfiguration.cutout_opacity;

  if (0 <= shaderConfiguration.idxCallGeometryCutoutOpacity)
  {
    optixDirectCall<void>(shaderConfiguration.idxCallGeometryCutoutOpacity, &opacity, &state, &res_data, nullptr, material.arg_block);
  }

  // If the current light sample is inside a fully cutout region, reject that sample.
  if (opacity <= 0.0f)
  {
    return lightSample;
  }

  // Note that lightSample.direction is from surface point to light sample position.
  const bool isFrontFace = (dot(lightSample.direction, state.geom_normal) < 0.0f);

  // thin_walled value in case the expression was a constant (idxCallThinWalled < 0).
  bool thin_walled = ((shaderConfiguration.flags & IS_THIN_WALLED) != 0);

  if (0 <= shaderConfiguration.idxCallThinWalled)
  {
    optixDirectCall<void>(shaderConfiguration.idxCallThinWalled, &thin_walled, &state, &res_data, nullptr, material.arg_block);
  }

  // Default to no EDF.
  int idxCallEmissionEval          = -1; 
  int idxCallEmissionIntensity     = -1;
  int idxCallEmissionIntensityMode = -1;
  // These are not used when there is no emission, no need to initialize.
  float3 emission_intensity;
  int    emission_intensity_mode;
  
  // MDL Specs: There is no emission on the back-side unless an EDF is specified with the backface field and thin_walled is set to true.
  if (isFrontFace)
  {
    idxCallEmissionEval          = shaderConfiguration.idxCallSurfaceEmissionEval;
    idxCallEmissionIntensity     = shaderConfiguration.idxCallSurfaceEmissionIntensity;
    idxCallEmissionIntensityMode = shaderConfiguration.idxCallSurfaceEmissionIntensityMode;

    emission_intensity      = shaderConfiguration.surface_intensity;
    emission_intensity_mode = shaderConfiguration.surface_intensity_mode;
  }
  else if (thin_walled) // && !isFrontFace
  {
    // These can be the same callable indices if the expressions from surface and backface were identical.
    idxCallEmissionEval          = shaderConfiguration.idxCallBackfaceEmissionEval;
    idxCallEmissionIntensity     = shaderConfiguration.idxCallBackfaceEmissionIntensity;
    idxCallEmissionIntensityMode = shaderConfiguration.idxCallBackfaceEmissionIntensityMode;
    
    emission_intensity      = shaderConfiguration.backface_intensity;
    emission_intensity_mode = shaderConfiguration.backface_intensity_mode;
  }

  // Check if the hit geometry contains any emission.
  if (0 <= idxCallEmissionEval)
  {
    if (0 <= idxCallEmissionIntensity) // Emission intensity is not a constant.
    {
      optixDirectCall<void>(idxCallEmissionIntensity, &emission_intensity, &state, &res_data, nullptr, material.arg_block);
    }
    if (0 <= idxCallEmissionIntensityMode) // Emission intensity mode is not a constant.
    {
      optixDirectCall<void>(idxCallEmissionIntensityMode, &emission_intensity_mode, &state, &res_data, nullptr, material.arg_block);
    }

    if (isNotNull(emission_intensity))
    {
      mi::neuraylib::Edf_evaluate_data<mi::neuraylib::DF_HSM_NONE> eval_data;

      eval_data.k1 = -lightSample.direction; // input: outgoing direction (from light sample position to surface point).
      //eval_data.cos : output: dot(normal, k1)
      //eval_data.edf : output: edf
      //eval_data.pdf : output: pdf (non-projected hemisphere)

      optixDirectCall<void>(idxCallEmissionEval, &eval_data, &state, &res_data, nullptr, material.arg_block);

      // Modulate the emission with the cutout opacity value to get the correct value.
      // The opacity value must not be greater than one here, which could happen for HDR textures.
      opacity = min(opacity, 1.0f);

      // Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
      const float factor = (emission_intensity_mode == 0) ? opacity : opacity / light.area;

      lightSample.pdf = lightSample.distance * lightSample.distance / (light.area * eval_data.cos); // Solid angle measure.

      lightSample.radiance_over_pdf = emission_intensity * eval_data.edf * (factor / lightSample.pdf);
    }
  }

  return lightSample;
}


extern "C" __global__ void __closesthit__curves()
{
  // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->flags |= FLAG_HIT; // Required to distinguish surface hits from random walk miss.

  thePrd->distance = optixGetRayTmax(); // Return the current path segment distance, needed for absorption calculations in the integrator.
  
  // Note that no adjustment to the hit position is done here!
  // The OptiX curve primitives are not intersecting with backfaces, which means the sceneEpsilon 
  // offset on the continuation ray t_min is enough to prevent self-intersections independently
  // of the continuation ray direction being a reflection or transmisssion.
  //thePrd->pos = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
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

  float4 objectToWorld[3];
  float4 worldToObject[3];

  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject); // Single instance level transformation list only.

  GeometryInstanceData theData = sysData.geometryInstanceData[optixGetInstanceId()];

  const unsigned int thePrimitiveIndex = optixGetPrimitiveIndex();

  const unsigned int* indices = reinterpret_cast<unsigned int*>(theData.indices);
  const unsigned int  index   = indices[thePrimitiveIndex];

  const CurveAttributes* attributes = reinterpret_cast<CurveAttributes*>(theData.attributes);
  
  float4 spline[4];

  spline[0] = attributes[index    ].vertex;
  spline[1] = attributes[index + 1].vertex;
  spline[2] = attributes[index + 2].vertex;
  spline[3] = attributes[index + 3].vertex;

  // Fixed bitangent-like reference vector for the vFiber calculation.
  float3 rf = make_float3(attributes[index].reference); 

  const float4 t = make_float4(attributes[index    ].texcoord.w,
                               attributes[index + 1].texcoord.w,
                               attributes[index + 2].texcoord.w,
                               attributes[index + 3].texcoord.w);

  CubicInterpolator interpolator;

  interpolator.initializeFromBSpline(spline);

  // Convenience function for __uint_as_float( optixGetAttribute_0() );
  const float u = optixGetCurveParameter(); 

  // Optimized (see curve.h):
  const float o_s = 1.0f / 6.0f;
  const float ts0 = (t.w - t.x) * o_s + (t.y - t.z) * 0.5f;
  const float ts1 = (t.x + t.z) * 0.5f - t.y;
  const float ts2 = (t.z - t.x) * 0.5f;
  const float ts3 = (t.x + t.y * 4.0f + t.z) * o_s;

  // Coordinate in range [0.0f, 1.0f] from root to tip along the whole fiber.
  const float uFiber = ( ( ( ts0 * u ) + ts1 ) * u + ts2 ) * u + ts3;

  const float4 fiberPosition = interpolator.position4(u); // .xyz = object space position, .w = radius of the curve's center line at the interpolant u.

  float3 tg = interpolator.velocity3(u); // Unnormalized object space tangent along the fiber from root to tip at the interpolant u.

  float3 position = transformPoint(worldToObject, thePrd->pos); // Transform to object space.
  float3 ns = surfaceNormal(interpolator, u, position);         // This moves position onto the surface of the curve in object space.

  // Transform into renderer internal space == world space.
  rf = normalize(transformVector(objectToWorld, rf));
  tg = normalize(transformVector(objectToWorld, tg));
  ns = normalize(transformNormal(worldToObject, ns));

  // Calculate an ortho-normal system for the fiber surface hit point. The shading normal is the fixed vector here!
  // Expanding the TBN tbn(tg, ns) constructor because TBN members can't be used as pointers for the Mdl_state with NUM_TEXTURE_SPACES > 1.
  float3 bt = normalize(cross(ns, tg));
  tg        = cross(bt, ns); // Now the tangent is orthogonal to the shading normal.

  // Transform the constant reference fiber bitangent vector into the local fiber coordinate system.
  // This makes the coordinate [0.0f, 1.0f] around the cross section of the hair relative to the hit point.
  // Expanded proj = fbn.transformToLocal(rf);
  // Only need the projected y- and z-components.(This works without normalization.
  const float2 proj = make_float2(dot(rf, bt), dot(rf, ns));

  // The (bitangent, normal) plane contains the cross section of the intersection. 
  // I want vFiber go from 0.0f to 1.0f counter-clockwise around the fiber when looking from the fiber root along the fiber.
  // As texture coordinate this means lower-left origin texture coordinates like in OpenGL.
  const float vFiber = (atan2f(-proj.x, proj.y) + M_PIf) * 0.5f * M_1_PIf;

  // The Mdl_state holds the texture attributes per texture space in separate arrays.
  // NUM_TEXTURE_SPACES is always at least 1.
  float3 texture_coordinates[NUM_TEXTURE_SPACES];
  float3 texture_tangents[NUM_TEXTURE_SPACES];
  float3 texture_bitangents[NUM_TEXTURE_SPACES];
  
  // In hair shading, texture spaces contain the following values:
  // texture_coordinate(0).x: The normalized position of the intersection point along the hair fiber in the range from zero for the root of the fiber to one for the tip of the fiber.
  // texture_coordinate(0).y: The normalized position of the intersection point around the hair fiber in the range from zero to one.
  // texture_coordinate(0).z: The thickness of the hair fiber at the intersection point in internal space.
  texture_coordinates[0] = make_float3(uFiber, vFiber, fiberPosition.w * 2.0f); // .z = thickness = radius * 2.0f
  texture_bitangents[0]  = bt;
  texture_tangents[0]    = tg;

#if NUM_TEXTURE_SPACES == 2
  // PERF Only ever set NUM_TEXTURE_SPACES to 2 if you need the hair BSDF to fully work, 
  // texture_coordinate(1): A position of the root of the hair fiber, for example, from a texture space of a surface supporting the hair fibers. This position is constant for a fiber.
  // Fixed texture coordinate for the fiber. Only loaded when NUM_TEXTURE_SPACES == 2.
  texture_coordinates[1] = make_float3(attributes[index].texcoord); 
  texture_bitangents[1]  = bt; // HACK Just copy the values from the first entry.
  texture_tangents[1]    = tg;
#endif 
  
  Mdl_state state;

  float4 texture_results[NUM_TEXTURE_RESULTS];

  state.normal                = ns;
  state.geom_normal           = ns;
  state.position              = thePrd->pos;
  state.animation_time        = 0.0f;
  state.text_coords           = texture_coordinates;
  state.tangent_u             = texture_tangents;
  state.tangent_v             = texture_bitangents;
  state.text_results          = texture_results;
  state.ro_data_segment       = nullptr;
  state.world_to_object       = worldToObject;
  state.object_to_world       = objectToWorld;
  state.object_id             = theData.ids.z;
  state.meters_per_scene_unit = 1.0f;
    
  const MaterialDefinitionMDL& material = sysData.materialDefinitionsMDL[theData.ids.x];

  mi::neuraylib::Resource_data res_data = { nullptr, material.texture_handler };

  const DeviceShaderConfiguration& shaderConfiguration = sysData.shaderConfigurations[material.indexShader];

  // This is always present, even if it just returns.
  optixDirectCall<void>(shaderConfiguration.idxCallInit, &state, &res_data, nullptr, material.arg_block);

  // Start fresh with the next BSDF sample.
  // Save the current path throughput for the direct lighting contribution.
  // The path throughput will be modulated with the BSDF sampling results before that.
  const float3 throughput = thePrd->throughput;
  // The pdf of the previous event was needed for the emission calculation above.
  thePrd->pdf = 0.0f;

  // Importance sample the hair BSDF. 
  if (0 <= shaderConfiguration.idxCallHairSample)
  {
    mi::neuraylib::Bsdf_sample_data sample_data;

    int idx = thePrd->idxStack;

    // FIXME The MDL-SDK libbsdf_hair.h ignores these ior values and only uses the ior value from the chiang_hair_bsdf structure!
    sample_data.ior1   = thePrd->stack[idx].ior;             // From surrounding medium ior
    sample_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR; // to material ior.
    sample_data.k1     = thePrd->wo;                         // == -optixGetWorldRayDirection()
    sample_data.xi     = rng4(thePrd->seed);

    optixDirectCall<void>(shaderConfiguration.idxCallHairSample, &sample_data, &state, &res_data, nullptr, material.arg_block);

    thePrd->wi          = sample_data.k2;            // Continuation direction.
    thePrd->throughput *= sample_data.bsdf_over_pdf; // Adjust the path throughput for all following incident lighting.
    thePrd->pdf         = sample_data.pdf;           // Specular events return pdf == 0.0f!
    thePrd->eventType   = sample_data.event_type;    // This replaces the previous PRD flags.
  }
  else
  {
    // If there is no valid scattering BSDF, it's the black bsdf() which ends the path.
    // This is usually happening with arbitrary mesh lights which only specify emission.
    thePrd->eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
    // None of the following code will have any effect in that case.
    return;
  }

  // Direct lighting if the sampled BSDF was diffuse and any light is in the scene.
  const int numLights = sysData.numLights;

  if (sysData.directLighting && 0 < numLights && (thePrd->eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY)))
  {
    // Sample one of many lights. 
    // The caller picks the light to sample. Make sure the index stays in the bounds of the sysData.lightDefinitions array.
    const int indexLight = (1 < numLights) ? clamp(static_cast<int>(floorf(rng(thePrd->seed) * numLights)), 0, numLights - 1) : 0;
    
    const LightDefinition& light = sysData.lightDefinitions[indexLight];
    
    LightSample lightSample = optixDirectCall<LightSample, const LightDefinition&, PerRayData*>(NUM_LENS_TYPES + light.typeLight, light, thePrd);

    if (0.0f < lightSample.pdf && 0 <= shaderConfiguration.idxCallHairEval) // Useful light sample and valid shader?
    {
      mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> eval_data;

      int idx = thePrd->idxStack;
      
      // DAR FIXME The MDL-SDK libbsdf_hair.h ignores these values and only uses the ior value from the chiang_hair-bsdf structure!
      eval_data.ior1   = thePrd->stack[idx].ior;             // From surrounding medium ior
      eval_data.ior2.x = MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR; // to material ior.
      eval_data.k1     = thePrd->wo;
      eval_data.k2     = lightSample.direction;

      optixDirectCall<void>(shaderConfiguration.idxCallHairEval, &eval_data, &state, &res_data, nullptr, material.arg_block);

      // DAR DEBUG This already contains the fabsf(dot(lightSample.direction, state.normal)) factor!
      // For a white Lambert material, the bxdf components match the eval_data.pdf
      const float3 bxdf = eval_data.bsdf_diffuse + eval_data.bsdf_glossy;

      if (0.0f < eval_data.pdf && isNotNull(bxdf))
      {
        // Pass the current payload registers through to the shadow ray.
        unsigned int p0 = optixGetPayload_0();
        unsigned int p1 = optixGetPayload_1();

        thePrd->flags &= ~FLAG_SHADOW; // Clear the shadow flag.

        // Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval 
        // to prevent self-intersections with the actual light geometry in the scene.
        optixTrace(sysData.topObject,
                   thePrd->pos, lightSample.direction, // origin, direction
                   sysData.sceneEpsilon, lightSample.distance - sysData.sceneEpsilon, 0.0f, // tmin, tmax, time
                   OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // The shadow ray type only uses anyhit programs.
                   TYPE_RAY_SHADOW, NUM_RAY_TYPES, TYPE_RAY_SHADOW,
                   p0, p1); // Pass through thePrd to the shadow ray.

        if ((thePrd->flags & FLAG_SHADOW) == 0) // Shadow flag not set?
        {
          const float weightMIS = (TYPE_LIGHT_POINT <= light.typeLight) ? 1.0f : balanceHeuristic(lightSample.pdf, eval_data.pdf);
          
          // The sampled emission needs to be scaled by the inverse probability to have selected this light,
          // Selecting one of many lights means the inverse of 1.0f / numLights.
          // This is using the path throughput before the sampling modulated it above.
          thePrd->radiance += throughput * bxdf * lightSample.radiance_over_pdf * (float(numLights) * weightMIS);
        }
      } 
    }
  }
}
