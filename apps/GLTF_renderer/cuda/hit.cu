//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <optix.h>

#include "launch_parameters.h"
#include "per_ray_data.h"
#include "hit_group_data.h"
#include "transform.h"
#include "shader_common.h"
#include "random.h"
#include "bxdf_common.h"


// DEBUG
//const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());
//if (theLaunchIndex.x == 256 && theLaunchIndex.y == 256)
//{
//  printf("value = %f\n", value);
//}


extern "C" {
  __constant__ LaunchParameters theLaunchParameters;
}


enum EnumLobes
{
  LOBE_DIFFUSE_REFLECTION,
  LOBE_SPECULAR_TRANSMISSION,
  LOBE_SPECULAR_REFLECTION,
  LOBE_METAL_REFLECTION,
  LOBE_SHEEN_REFLECTION,
  LOBE_CLEARCOAT_REFLECTION,

  NUM_LOBES
};


struct __align__(8) State
{
  // 16 byte aligned

  //  8 byte aligned
  float2 xi; // sample input: pseudo-random sample numbers in range [0, 1). Two for the direction.
  float2 roughness;
  
  //  4 byte aligned
  // Geometry attributes.
  float3 Ng; // Geometry normal, flipped to the side of the ray.
  
  float3 Tc; // Shading clearcoat tangent.
  float3 Nc; // Shading clearcoat normal.
  
  // Shading space.
  float3 T; // Shading tangent.
  float3 B; // Shading bitangent. Multiplied by the handedness.
  float3 N; // Shading normal, flipped to the side of the ray.
  
  float handedness; // 1.0f for right-handed, -1.0f for left-handed.

  float clearcoat; // The weight of the clearcoat layer.
  float clearcoatRoughness;
  
  float metallic;
  float occlusion;

  float transmission; 

  float3 emission;
  float3 baseColor;
  
  float  specular; // The weight of the dielectric specular layer.
  float3 specularColor;

  float3 sheenColor;
  float  sheenRoughness;

  float iridescence;
  float iridescenceIor;
  float iridescenceThickness; // In nanometers.

  // Sampling and eval parameters.
  float  ior1;             // sample and eval input: IOR current medium
  float  ior2;             // sample and eval input: IOR other side
  float3 k1;               // sample and eval input: outgoing direction (== prd.wo == negative optixGetWorldRayDirection())
  float3 k2;               // sample output: incoming direction (continuation ray, prd.wi)
                           // eval input:    incoming direction (direction to light sample point)
  float3 bsdf_over_pdf;    // sample output: bsdf * dot(k2, normal) / pdf
  float  pdf;              // sample and eval output: pdf (non-projected hemisphere) 
  BsdfEventType typeEvent; // sample output: the type of event for the generated sample
  
  //  2 byte aligned

  //  1 byte aligned
  bool isThinWalled;
};


template <> __forceinline__ __device__  
const GeometryData::TriangleMesh& GeometryData::getMesh<GeometryData::TriangleMesh>() const
{ 
  assert(type == TRIANGLE_MESH);  
  return triangleMesh; 
}


template <> __forceinline__ __device__
const GeometryData::SphereMesh& GeometryData::getMesh<GeometryData::SphereMesh>() const
{ 
  assert(type == SPHERE_MESH); 
  return sphereMesh; 
}


template<typename T>
__forceinline__ __device__ T sampleTexture(const MaterialData::Texture& texture, const float2 uv)
{
  // Assumes the caller checked that texture.object is valid.
  const float2 UV       = uv * texture.scale;
  const float2 rotation = texture.rotation; // .x = sin, .y = cos
  const float2 UV_trans = make_float2(dot(make_float2( rotation.y, rotation.x), UV),
                                      dot(make_float2(-rotation.x, rotation.y), UV))
                        + texture.translation;

  return tex2D<T>(texture.object, UV_trans.x, UV_trans.y);
}

// Fill the given state for the hit primitive.
template<typename MESH> void initializeState(State& state,
                                             const MESH& mesh,
                                             const MaterialData& material);


// To pass local variables to the epilogue of initializeState(). For Tris,Spheres, ...
// The epilogue is needed to avoid code duplication.
struct LocalVars
{
  __device__ LocalVars()
  {
    color = make_float4(1.0f); // Always initialize. Most models do not use COLOR attributes.
  }

  float4 color;
  float3 Ng_obj;
  float3 Ng;
  float3 N_obj;
  float3 N;
  float3 Nc;    // Normalized clearcoat shading normal. Defaults to shading normal but can have an own clearcoat normal map.
  float3 T_obj; // Unnormalized object space tangent. (Needed inside the texture tangent space calculation).
  float3 T;     // Normalized world space shading tangent.

  // TEXCOORD_0, TEXCOORD_1
  float2 texcoord[NUM_ATTR_TEXCOORDS];
  float3 texTangent[NUM_ATTR_TEXCOORDS];
  float3 texBitangent[NUM_ATTR_TEXCOORDS];
};


__forceinline__ __device__
void initializeStateEpilogue(State& state, const MaterialData& material, LocalVars& vars)
{
  if (material.normalTexture)
  {
    const int index = material.normalTexture.index;
    const float3 N_tex = make_float3(sampleTexture<float4>(material.normalTexture, vars.texcoord[index])) * 2.0f - 1.0f;
    // Transform normal from texture space to rotated UV space.
    const float2 N_proj = make_float2(N_tex) * material.normalTextureScale; // .xy * normalTextureScale
    const float2 rotation = material.normalTexture.rotation; // .x = sin, .y = cos
    const float2 N_trns = make_float2(dot(make_float2(rotation.y, -rotation.x), N_proj), // Opposite rotation to sampleTexture()
                                      dot(make_float2(rotation.x, rotation.y), N_proj));
    // Shading normal in world space (because tangent, bitangent and N are in world space).
    vars.N = normalize(N_trns.x * normalize(vars.texTangent[index]) +
                       N_trns.y * normalize(vars.texBitangent[index]) +
                       N_tex.z * vars.N);
  }

  // KHR_materials_clearcoat
  float clearcoatFactor = material.clearcoatFactor;
  if (material.clearcoatTexture)
  {
    clearcoatFactor *= sampleTexture<float4>(material.clearcoatTexture, vars.texcoord[material.clearcoatTexture.index]).x;
  }
  state.clearcoat = clearcoatFactor;

  if (0.0f < clearcoatFactor) // PERF None of the other clearcoat values is referenced when this is false.
  {
    float clearcoatRoughness = material.clearcoatRoughnessFactor;
    if (material.clearcoatRoughnessTexture)
    {
      clearcoatRoughness *= sampleTexture<float4>(material.clearcoatRoughnessTexture, vars.texcoord[material.clearcoatRoughnessTexture.index]).y;
    }
    state.clearcoatRoughness = fmaxf(MICROFACET_MIN_ROUGHNESS, clearcoatRoughness); // Perceptual roughness, not squared!

    if (material.clearcoatNormalTexture)
    {
      const int index = material.clearcoatNormalTexture.index;
      const float3 N_tex = make_float3(sampleTexture<float4>(material.clearcoatNormalTexture, vars.texcoord[index])) * 2.0f - 1.0f;
      // Transform normal from texture space to rotated UV space.
      float2 N_proj = make_float2(N_tex);
      if (material.isClearcoatNormalBaseNormal)
      {
        N_proj *= material.normalTextureScale;
      }
      const float2 rotation = material.clearcoatNormalTexture.rotation; // .x = sin, .y = cos
      const float2 N_trns = make_float2(dot(make_float2(rotation.y, -rotation.x), N_proj), // Opposite rotation to sampleTexture()
                                        dot(make_float2(rotation.x, rotation.y), N_proj));
      // Shading normal in world space (because tangent, bitangent and N are in world space).
      vars.Nc = normalize(N_trns.x * normalize(vars.texTangent[index]) +
                     N_trns.y * normalize(vars.texBitangent[index]) +
                     N_tex.z * vars.Nc);
    }
    state.Nc = vars.Nc;
  }

  // baseColor
  // Ignore the alpha channel. That is only needed for the opacity which is evaluated separately in getOpacity().
  float3 baseColor = make_float3(material.baseColorFactor) * make_float3(vars.color);
  if (material.baseColorTexture)
  {
    baseColor *= make_float3(sampleTexture<float4>(material.baseColorTexture, vars.texcoord[material.baseColorTexture.index])); // sRGB
  }
  state.baseColor = baseColor; // The "tint" is what is used inside the BXDFs. (It can change depending on the sampled lobe.)

  // metallic, roughness
  float roughness = material.roughnessFactor;
  float metallic = material.metallicFactor;
  if (material.metallicRoughnessTexture)
  {
    const float4 tex = sampleTexture<float4>(material.metallicRoughnessTexture, vars.texcoord[material.metallicRoughnessTexture.index]);

    roughness *= tex.y; // The green channel contains the roughness value.
    metallic *= tex.z; // The blue channel contains the metallic value.
  }
  roughness = fmaxf(MICROFACET_MIN_ROUGHNESS, roughness);

  state.roughness = make_float2(roughness * roughness); // Isotropic.
  state.metallic = metallic;

  float3 emission = material.emissiveFactor;
  if (material.emissiveTexture)
  {
    emission *= make_float3(sampleTexture<float4>(material.emissiveTexture, vars.texcoord[material.emissiveTexture.index])); // sRGB
  }
  state.emission = emission * material.emissiveStrength;

  float occlusion = 1.0f;
  if (theLaunchParameters.ambientOcclusion && material.occlusionTexture)
  {
    occlusion += material.occlusionTextureStrength * (sampleTexture<float4>(material.occlusionTexture, vars.texcoord[material.occlusionTexture.index]).x - 1.0f);
  }
  state.occlusion = occlusion;

  // KHR_materials_transmission
  float transmission = material.transmissionFactor;
  if (material.transmissionTexture)
  {
    transmission *= sampleTexture<float4>(material.transmissionTexture, vars.texcoord[material.transmissionTexture.index]).x;
  }
  state.transmission = transmission;

  // KHR_materials_specular
  float specularFactor = material.specularFactor;
  if (material.specularTexture)
  {
    specularFactor *= sampleTexture<float4>(material.specularTexture, vars.texcoord[material.specularTexture.index]).w;
  }
  state.specular = specularFactor;

  float3 specularColor = material.specularColorFactor;
  if (material.specularColorTexture)
  {
    specularColor *= make_float3(sampleTexture<float4>(material.specularColorTexture, vars.texcoord[material.specularColorTexture.index])); // sRGB
  }
  // Note that the SpecularTest.gltf uses specularColorFactor brighter than white which generates light.
  // Intentionally do NOT slow down the renderer for incorrect input!
  state.specularColor = specularColor;

  // KHR_materials_sheen
  float3 sheenColor = material.sheenColorFactor;
  if (material.sheenColorTexture)
  {
    sheenColor *= make_float3(sampleTexture<float4>(material.sheenColorTexture, vars.texcoord[material.sheenColorTexture.index])); // sRGB
  }
  state.sheenColor = sheenColor; // No sheen if this is black.

  float sheenRoughness = material.sheenRoughnessFactor;
  if (material.sheenRoughnessTexture)
  {
    sheenRoughness *= sampleTexture<float4>(material.sheenRoughnessTexture, vars.texcoord[material.sheenRoughnessTexture.index]).w;
  }
  sheenRoughness = fmaxf(MICROFACET_MIN_ROUGHNESS, sheenRoughness);
  state.sheenRoughness = sheenRoughness;

  float iridescence = material.iridescenceFactor;
  if (material.iridescenceTexture)
  {
    iridescence *= sampleTexture<float4>(material.iridescenceTexture, vars.texcoord[material.iridescenceTexture.index]).x;
  }
  float iridescenceThickness = material.iridescenceThicknessMaximum;
  if (material.iridescenceThicknessTexture)
  {
    const float t = sampleTexture<float4>(material.iridescenceThicknessTexture, vars.texcoord[material.iridescenceThicknessTexture.index]).y;
    iridescenceThickness = lerp(material.iridescenceThicknessMinimum, material.iridescenceThicknessMaximum, t);
  }
  state.iridescence = (0.0f < iridescenceThickness) ? iridescence : 0.0f; // No iridescence when the thickness is zero.
  state.iridescenceIor = material.iridescenceIor;
  state.iridescenceThickness = iridescenceThickness;

  // KHR_materials_anisotropy
  float  anisotropyStrength = material.anisotropyStrength;
  float2 anisotropyDirection = make_float2(1.0f, 0.0f); // By default the anisotropy strength is along the tangent.
  if (material.anisotropyTexture)
  {
    const float4 anisotropyTex = sampleTexture<float4>(material.anisotropyTexture, vars.texcoord[material.anisotropyTexture.index]);

    // .xy encodes the direction in (tangent, bitangent) space. Remap from [0, 1] to [-1, 1].
    anisotropyDirection = normalize(make_float2(anisotropyTex) * 2.0f - 1.0f);
    // .z encodes the strength in range [0, 1].
    anisotropyStrength *= anisotropyTex.z;
  }

  // Ortho-normal shading space.
  state.N = vars.N;
  state.B = normalize(cross(vars.N, vars.T)); // Assumes T and N are not collinear!
  state.T = cross(state.B, vars.N);

  // If the anisotropyStrength == 0.0f (default), the roughness is isotropic. 
  // No need to rotate the anisotropyDirection or tangent space.
  if (0.0f < anisotropyStrength)
  {
    state.roughness.x = lerp(state.roughness.y, 1.0f, anisotropyStrength * anisotropyStrength);

    const float s = sinf(material.anisotropyRotation); // FIXME PERF Precalculate sin, cos on host.
    const float c = cosf(material.anisotropyRotation);

    anisotropyDirection = make_float2(c * anisotropyDirection.x + s * anisotropyDirection.y,
                                      c * anisotropyDirection.y - s * anisotropyDirection.x);

    const float3 T_aniso = state.T * anisotropyDirection.x +
      state.B * anisotropyDirection.y;

    state.B = normalize(cross(vars.N, T_aniso));
    state.T = cross(state.B, vars.N);
  }
  state.B *= state.handedness;

  // Geometry is handled as thin-walled
  // * if alpha mode is MASK or BLEND (not OPAQUE) or
  // * if not using KHR_materials_volume (because otherwise backfaces show TIR effects) or
  // * if using KHR_materials_volume and the material.thicknessFactor is zero.
  state.isThinWalled = material.alphaMode != MaterialData::ALPHA_MODE_OPAQUE ||
    (material.flags & FLAG_KHR_MATERIALS_VOLUME) == 0 ||
    material.thicknessFactor <= 0.0f;
}


// Called by CH radiance, triangles.
template<>
__forceinline__ __device__ void initializeState<GeometryData::TriangleMesh>(State& state,
                                                                            const GeometryData::TriangleMesh& mesh,
                                                                            const MaterialData& material)
{
  LocalVars vars;
  const unsigned int prim_idx = optixGetPrimitiveIndex();

  const float2 barys = optixGetTriangleBarycentrics();   // .x = beta, .y = gamma
  const float  barys_a = 1.0f - barys.x - barys.y;       // alpha

  // PERF Get the matrices once and use the explicit transform functions.
  float4 objectToWorld[3];
  float4 worldToObject[3];

  // This works for a single instance level transformation list only!
  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject);

  // INDICES (triangle specific!)
  uint3 tri;
  if (mesh.indices)
  {
    tri = mesh.indices[prim_idx];
  }
  else
  {
    const unsigned int base_idx = prim_idx * 3;
    tri = make_uint3(base_idx, base_idx + 1, base_idx + 2);
  }

  // POSITION
  const float3 p0 = mesh.positions[tri.x];
  const float3 p1 = mesh.positions[tri.y];
  const float3 p2 = mesh.positions[tri.z];

  //float3 P = barys_a * p0 + barys.x * p1 + barys.y * p2;
  //P = transformPoint(objectToWorld, P); // This is not required inside the State.

  // COLOR_0
  if (mesh.colors)
  {
    vars.color = barys_a * mesh.colors[tri.x] +
                 barys.x * mesh.colors[tri.y] +
                 barys.y * mesh.colors[tri.z];
  }

  // Object space geometry normal.
  vars.Ng_obj = cross(p1 - p0, p2 - p0); // Unnormalized. Mind the CCW order.
  // transformNormal() takes the inverse matrix to do the inverse transpose transformation.
  vars.Ng = normalize(transformNormal(worldToObject, vars.Ng_obj));
  state.Ng = vars.Ng;

  // NORMAL
  // Shading normals default to geometric normals when there are no normal attributes.
  vars.N_obj = vars.Ng_obj; // Unnormalized object space shading normal.
  vars.N = vars.Ng;              // Normalized world space shading normal.
  if (mesh.normals)
  {
    vars.N_obj = barys_a * mesh.normals[tri.x] +
                 barys.x * mesh.normals[tri.y] +
                 barys.y * mesh.normals[tri.z]; // Unnormalized.

    vars.N = normalize(transformNormal(worldToObject, vars.N_obj));
  }

  vars.Nc = vars.N;

  // TANGENT

  state.handedness = 1.0f; // Default is a right-handed coordinate system.
  // When the mesh contains tangent attributes, the tangent and normal define the TBN reference system for the normalTexture.
  if (mesh.tangents)
  {
    const float4 t0 = mesh.tangents[tri.x];
    const float4 t1 = mesh.tangents[tri.y];
    const float4 t2 = mesh.tangents[tri.z];

    // The tangent attribute .w component is 1.0f or -1.0f for the handedness. 
    // It multiplies the bitangent and is the same for all three tangents.
    state.handedness = t0.w;

    vars.T_obj = barys_a * make_float3(t0) +
                 barys.x * make_float3(t1) +
                 barys.y * make_float3(t2); // Unnormalized.

    vars.T = normalize(transformVector(objectToWorld, vars.T_obj));
  }

  // WARNING: Unfortunately not all GLTF models provide correct tangents.
  // When there are no valid tangent attributes, just generate some tangent vector
  // which is not collinear with the shading normal.
  if (!mesh.tangents || (1.0f - fabsf(dot(vars.T, vars.N))) < DENOMINATOR_EPSILON)
  {
    // This will have discontinuities on spherical objects when using anisotropic roughness.
    // Make sure the object space tangent is generated because that is required
    // inside the texture tangent space calculation for normal maps.
    if (fabsf(vars.N_obj.z) < fabsf(vars.N_obj.x))
    {
      vars.T_obj.x = vars.N_obj.z;
      vars.T_obj.y = 0.0f;
      vars.T_obj.z = -vars.N_obj.x;
    }
    else
    {
      vars.T_obj.x = 0.0f;
      vars.T_obj.y = vars.N_obj.z;
      vars.T_obj.z = -vars.N_obj.y;
    }
    // T_obj is unnormalized.
    vars.T = normalize(transformVector(objectToWorld, vars.T_obj));
  }

  state.T = vars.T;
  state.Tc = vars.T; // Save the original shading tangent for the clearcoat TBN space calculation when required.

  // TEXCOORD_0, TEXCOORD_1

  // The texture tangent space calculation assumes normalized vectors.
  vars.N_obj = normalize(vars.N_obj);
  vars.T_obj = normalize(vars.T_obj);
  const float3 B_obj = normalize(cross(vars.N_obj, vars.T_obj)) * state.handedness; // N_obj is unnormalized.

  for (int j = 0; j < NUM_ATTR_TEXCOORDS; j++)
  {
    vars.texcoord[j] = make_float2(0.0f); // This is what the reference glTF-SampleViewer does when there are no texcoords.

    // FIXME PERF These can't be required when there are no texcoords on that slot!
    // If tangents are provided, they define the normal texture shading space.
    // If not, these are also the defaults. They only get changed for normal textures.
    vars.texTangent[j] = vars.T_obj;
    vars.texBitangent[j] = B_obj;

    if (mesh.texcoords[j])
    {
      // Texture coordinates of triangle vertices. (Texture coordinates are always in object (texture) space).
      const float2 uv0 = mesh.texcoords[j][tri.x];
      const float2 uv1 = mesh.texcoords[j][tri.y];
      const float2 uv2 = mesh.texcoords[j][tri.z];

      // Texture coordinate of the hit point.
      vars.texcoord[j] = barys_a * uv0 + barys.x * uv1 + barys.y * uv2;

      // PERF Generate a local texture tangent space only when necessary.
      if (!mesh.tangents &&                                                                  // If there were no tangent attributes and
          ((material.normalTexture && material.normalTexture.index == j) ||         // the texcoord index is used for normal mapping or 
           (material.clearcoatNormalTexture && material.clearcoatNormalTexture.index == j))) // the texcoord index is used for clearcoat normal mapping.
      {
        // Calculate a texture tangent space from the positions and texture coordinate derivatives.
        // This is matching mostly to what is done inside Khronos GLTF sample viewer application.
        // There are no dFdx(), dFdy() equivalents without ray differentials, so do that on the surface geometry instead.

        // Object space.
        const float3 dp_dx = p1 - p0; // Position vector change when going from vertex 0 to 1. Think of x-axis basis vector.
        const float3 dp_dy = p2 - p0; // Position vector change when going from vertex 0 to 2. Think of y-axis basis vector.

        float2 duv_dx = uv1 - uv0; // Texture coordinate change when going from vertex 0 to 1.
        float2 duv_dy = uv2 - uv0; // Texture coordinate change when going from vertex 0 to 2.

        // Check if the texture coordinate delta values are reasonably big.
        // Some Khronos GLTF example models have degenerate texture coordinates (same coordinate on two or three vertices) 
        // which doesn't span an area in texture space and the denominator becomes zero.
        // When correcting these individually there can be null denominators in the T_tex calculation below
        // so force both values to ortho-normal vectors when one is too small.
        if (length(duv_dx) <= 0.0001f || length(duv_dy) <= 0.0001f) // FIXME Magic number.
        {
          duv_dx = make_float2(1.0f, 0.0f);
          duv_dy = make_float2(0.0f, 1.0f);
        }

        const float denom = duv_dx.x * duv_dy.y - duv_dy.x * duv_dx.y;

        const float3 T_tex = (DENOMINATOR_EPSILON < fabsf(denom)) // Prevent NaN results inside the tangent.
          ? (duv_dy.y * dp_dx - duv_dx.y * dp_dy) / denom
          : vars.T_obj; // Use the object space tangent when none can be derived from texture coordinates.

        // Unnormalized vectors because the texture tangent space will only be used
        // when there are normal maps and that normalizes the tangent and bitangent later.
        // These are without the normalTexture rotation applied, so these are in texture space, not in rotated texture space.
        // The final shading normal calculation does the inverse rotation on the .xy components of the normalTexture vector to bring them 
        // from rotated texture space, where the lookup happened, back into the texture space to get the correct world space normal.
        // This is all only necessary for the shading normal space generation.
        vars.texTangent[j]   = T_tex - vars.N_obj * dot(T_tex, vars.N_obj);
        vars.texBitangent[j] = normalize(cross(vars.N_obj, T_tex)) * state.handedness;
      }
    }
  }

  // Now determine the material dependent attributes.
  initializeStateEpilogue(state, material, vars);
}


// Called by CH radiance, spheres.
template<>
__forceinline__ __device__ void initializeState<GeometryData::SphereMesh>(State& state,
                                                                          const GeometryData::SphereMesh& mesh,
                                                                          const MaterialData& material)
{
  LocalVars vars;
  const unsigned int prim_idx = optixGetPrimitiveIndex();
  // PERF Get the matrices once and use the explicit transform functions.
  float4 objectToWorld[3];
  float4 worldToObject[3];

  // This works for a single instance level transformation list only!
  getTransforms(optixGetTransformListHandle(0), objectToWorld, worldToObject);

  //float3 P = barys_a * p0 + barys.x * p1 + barys.y * p2;
  //P = transformPoint(objectToWorld, P); // This is not required inside the State.

  // COLOR_0
  if (mesh.colors)
  {
    vars.color = mesh.colors[prim_idx];
  }

  // Hit sphere data
  const float3 sphereP = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
  const float3 sphereP_obj = transformPoint(worldToObject, sphereP);

  const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
  const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

  // get centre and radius, obj space
  float4 sphereObj;
  optixGetSphereData(gas, prim_idx, sbtGASIndex, 0.f, &sphereObj);

  const float3 sphereC_obj = make_float3(sphereObj.x, sphereObj.y, sphereObj.z);
  const float3 sphereC     = transformPoint(objectToWorld, sphereC_obj);
  const float3 sphereN_u   = make_float3(sphereP.x - sphereC.x,       // unnormalised
                                         sphereP.y - sphereC.y,
                                         sphereP.z - sphereC.z);
  state.Ng = sphereN_u / sphereObj.w; // Normalized world space shading normal.
  vars.N = state.Ng;                  // Normalized world space shading normal.

  // NORMAL
  // Shading normals default to geometric normals when there are no normal attributes.
  vars.N_obj = mesh.normals ? mesh.normals[prim_idx] : make_float3(sphereP_obj.x - sphereC_obj.x,
                                                                   sphereP_obj.y - sphereC_obj.y,
                                                                   sphereP_obj.z - sphereC_obj.z);; // Unnormalized object space shading normal.
  if (mesh.normals)
  {
    vars.N = normalize(transformNormal(worldToObject, vars.N_obj));
  }

  vars.Nc = vars.N;

  // TANGENT
  state.handedness = 1.0f; // Default is a right-handed coordinate system.
  // When the mesh contains tangent attributes, the tangent and normal define the TBN reference system for the normalTexture.
  /* TODO? if (mesh.tangents)
     Currently we assume that tangents are missing. (By the way, it's not hard to build an ONB around the hit point).
  {
    const float4 t0 = mesh.tangents[prim_idx];

    // The tangent attribute .w component is 1.0f or -1.0f for the handedness. 
    // It multiplies the bitangent and is the same for all three tangents.
    state.handedness = t0.w;

    T_obj = make_float3(t0); // Unnormalized.

    T = normalize(transformVector(objectToWorld, T_obj));
  }*/

  // vars.T is not init at this point.

  // WARNING: Unfortunately not all GLTF models provide correct tangents.
  // When there are no valid tangent attributes, just generate some tangent vector
  // which is not collinear with the shading normal.
  //if (!mesh.tangents || (1.0f - fabsf(dot(T, N))) < DENOMINATOR_EPSILON)
  {
    // This will have discontinuities on spherical objects when using anisotropic roughness.
    // Make sure the object space tangent is generated because that is required
    // inside the texture tangent space calculation for normal maps.
    if (fabsf(vars.N_obj.z) < fabsf(vars.N_obj.x))
    {
      vars.T_obj.x = vars.N_obj.z;
      vars.T_obj.y = 0.0f;
      vars.T_obj.z = -vars.N_obj.x;
    }
    else
    {
      vars.T_obj.x = 0.0f;
      vars.T_obj.y = vars.N_obj.z;
      vars.T_obj.z = -vars.N_obj.y;
    }
    // T_obj is unnormalized.
    vars.T = normalize(transformVector(objectToWorld, vars.T_obj));
  }

  state.T = vars.T;
  state.Tc = vars.T; // Save the original shading tangent for the clearcoat TBN space calculation when required.

  // The texture tangent space calculation assumes normalized vectors.
  vars.N_obj = normalize(vars.N_obj);
  vars.T_obj = normalize(vars.T_obj);
  const float3 B_obj = normalize(cross(vars.N_obj, vars.T_obj)) * state.handedness; // N_obj is unnormalized.

  for (int j = 0; j < NUM_ATTR_TEXCOORDS; j++)
  {
    vars.texcoord[j] = make_float2(0.0f); // This is what the reference glTF-SampleViewer does when there are no texcoords.

    // FIXME PERF These can't be required when there are no texcoords on that slot!
    // If tangents are provided, they define the normal texture shading space.
    // If not, these are also the defaults. They only get changed for normal textures.
    vars.texTangent[j] = vars.T_obj;
    vars.texBitangent[j] = B_obj;

    /* TODO?
    if (mesh.texcoords[j])
    {
      
    }
    */
  }

  // Now determine the material dependent attributes.
  initializeStateEpilogue(state, material, vars);
}


// Get TRIANGLES' opacity.
// The anyhit programs only need the opacity not the full State.
__forceinline__ __device__ float getOpacity(const GeometryData::TriangleMesh& mesh,
                                            const MaterialData& material)
{
  const unsigned int prim_idx = optixGetPrimitiveIndex();

  // INDICES 
  uint3 tri;
  if (mesh.indices)
  {
    tri = mesh.indices[prim_idx];
  }
  else
  {
    const unsigned int base_idx = prim_idx * 3;

    tri = make_uint3(base_idx, base_idx + 1, base_idx + 2);
  }

  const float2 barys   = optixGetTriangleBarycentrics(); // .x = beta, .y = gamma
  const float  barys_a = 1.0f - barys.x - barys.y;       // alpha

  float opacity = material.baseColorFactor.w; // baseColorFactor alpha is the base opacity.

  // COLOR_0
  if (mesh.colors)
  {
    // If there are color attributes the alpha modulates the opacity.
    opacity *= barys_a * mesh.colors[tri.x].w +
               barys.x * mesh.colors[tri.y].w +
               barys.y * mesh.colors[tri.z].w;
  }

  // If the baseColor has a texture the alpha in there modulates the opacity.
  if (material.baseColorTexture)
  {
    // No need to calculate texture coordinates when there is no baseColorTexture 
    // and only one of the two supported texture coordinates is required here.
    // TEXCOORD_0 or TEXCOORD_1
    const int j = material.baseColorTexture.index;

    float2 texcoord = make_float2(0.0f);
    if (mesh.texcoords[j])
    {
      // Texture coordinate of the hit point.
      texcoord = barys_a * mesh.texcoords[j][tri.x] +
                 barys.x * mesh.texcoords[j][tri.y] +
                 barys.y * mesh.texcoords[j][tri.z];
    }

    opacity *= sampleTexture<float4>(material.baseColorTexture, texcoord).w; // Opacity is not sRGB but linear.
  }
  return opacity;
}


// Get SPHERES' opacity.
// Note: anyhit programs only need the opacity, not the full State.
// @return Linear opacity (not sRGB).
__forceinline__ __device__ float getOpacity(const GeometryData::SphereMesh& mesh,
                                            const MaterialData& material)
{
  float opacity = material.baseColorFactor.w; // baseColorFactor alpha is the base opacity.

  // COLOR_0
  if (mesh.colors)
  {
    // If there are color attributes the alpha modulates the opacity.
    opacity *= mesh.colors[optixGetPrimitiveIndex()].w; // unweighted sphere's opacity
  }

  // If the baseColor has a texture the alpha in there modulates the opacity.
  if (material.baseColorTexture)
  {
    // No need to calculate texture coordinates when there is no baseColorTexture 
    // and only one of the two supported texture coordinates is required here.
    // TEXCOORD_0 or TEXCOORD_1
    //const int j = material.baseColorTexture.index;

    float2 texcoord = make_float2(0.0f);
    //if (mesh.texcoords[j])
    //{
    //  // Texture coordinate of the hit point.
    //     *** TODO tex-coords for the hit point on the sphere ***
    //}

    // Opacity is not sRGB but linear.
    opacity *= sampleTexture<float4>(material.baseColorTexture, texcoord).w;
  }
  return opacity;
}

template<typename MESH>
__device__ float3 getBaseColor(const MESH& mesh, const MaterialData& material);

// Get TRIANGLES' base color.
// For the KHR_materials_unlit case, only the baseColor is required.
template<>
__forceinline__ __device__ float3 getBaseColor<GeometryData::TriangleMesh>(const GeometryData::TriangleMesh& mesh,
                                                                           const MaterialData&               material)
{
  const unsigned int prim_idx = optixGetPrimitiveIndex();

  // INDICES 
  uint3 tri;
  if (mesh.indices)
  {
    tri = mesh.indices[prim_idx];
  }
  else
  {
    const unsigned int base_idx = prim_idx * 3;

    tri = make_uint3(base_idx, base_idx + 1, base_idx + 2);
  }

  const float2 barys   = optixGetTriangleBarycentrics(); // .x = beta, .y = gamma
  const float  barys_a = 1.0f - barys.x - barys.y;       // alpha

  float3 baseColor = make_float3(material.baseColorFactor);

  // COLOR_0
  if (mesh.colors)
  {
    // If there are color attributes the alpha modulates the opacity.
    baseColor *= barys_a * make_float3(mesh.colors[tri.x]) +
                 barys.x * make_float3(mesh.colors[tri.y]) +
                 barys.y * make_float3(mesh.colors[tri.z]);
  }

  // If the baseColor has a texture the alpha in there modulates the opacity.
  if (material.baseColorTexture)
  {
    // No need to calculate texture coordinates when there is no baseColorTexture 
    // and only one of the two supported texture coordinates is required here.
    // TEXCOORD_0 or TEXCOORD_1
    const int j = material.baseColorTexture.index;

    float2 texcoord = make_float2(0.0f);

    if (mesh.texcoords[j])
    {
      // Texture coordinate of the hit point.
      texcoord = barys_a * mesh.texcoords[j][tri.x] +
                 barys.x * mesh.texcoords[j][tri.y] +
                 barys.y * mesh.texcoords[j][tri.z];
    }

    baseColor *= make_float3(sampleTexture<float4>(material.baseColorTexture, texcoord)); // sRGB
  }

  return baseColor;
}


// Get SPHERES' base color.
// For the KHR_materials_unlit case, only the baseColor is required.
template<>
__forceinline__ __device__ float3 getBaseColor<GeometryData::SphereMesh>(const GeometryData::SphereMesh& mesh,
                                                                         const MaterialData&             material)
{
  const unsigned int prim_idx = optixGetPrimitiveIndex();
  // each sphere has own color
  float3 sphereBaseColor = make_float3(material.baseColorFactor);
  
  // needed?
  //float3 matBaseColor = make_float3(material.baseColorFactor);

  // COLOR_0
  if (mesh.colors)
  {
    // If there are color attributes the alpha modulates the opacity.
    sphereBaseColor *= make_float3(mesh.colors[prim_idx]);
  }

  // If the baseColor has a texture the alpha in there modulates the opacity.
  if (material.baseColorTexture)
  {
    // No need to calculate texture coordinates when there is no baseColorTexture 
    // and only one of the two supported texture coordinates is required here.
    // TEXCOORD_0 or TEXCOORD_1
    // const int j = material.baseColorTexture.index;

    float2 texcoord = make_float2(0.0f);

    /*if (mesh.texcoords[j])
    {
      // Texture coordinate of the hit point.
      texcoord = barys_a * mesh.texcoords[j][tri.x] +
                 barys.x * mesh.texcoords[j][tri.y] +
                 barys.y * mesh.texcoords[j][tri.z];
    }*/

    sphereBaseColor *= make_float3(sampleTexture<float4>(material.baseColorTexture, texcoord)); // sRGB
  }

  return sphereBaseColor;
}



// BXDF sample and eval functions.

__forceinline__ __device__ void brdf_diffuse_sample(State& state, const float3 tint)
{
  // Cosine weighted hemisphere sampling for Lambert material.
  unitSquareToCosineHemisphere(state.xi, state.N, state.k2, state.pdf);

  state.bsdf_over_pdf = tint; // bsdf * dot(wi, normal) / pdf;
  state.typeEvent     = (0.0f < dot(state.k2, state.Ng)) ? BSDF_EVENT_DIFFUSE_REFLECTION : BSDF_EVENT_ABSORB;
}


__forceinline__ __device__ float3 brdf_diffuse_eval(State& state, const float3 tint)
{
  // If the incoming light direction is on the backside, there is nothing to evaluate for a BRDF.
  // Note that the state normals have been flipped to the ray side by the caller.
  // Include edge-on (== 0.0f) as "no light" case.
  if (dot(state.k2, state.Ng) <= 0.0f) // if (backside) 
  {
    state.pdf = 0.0f; // absorb
    return make_float3(0.0f);
  }

  state.pdf = fmaxf(0.0f, dot(state.k2, state.N) * M_1_PIf);
  
  // For a white Lambert material, the bxdf components match the evaluation pdf. (See MDL_renderer.)
  return tint * state.pdf;
}


__forceinline__ __device__ void brdf_ggx_smith_sample(State& state, const int lobe, float3 tint, const bool thinfilm)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  state.bsdf_over_pdf = make_float3(0.0f);
  state.pdf           = 0.0f;

  const float nk1 = fabsf(dot(state.k1, state.N));

  const float3 k10 = make_float3(dot(state.k1, state.T), dot(state.k1, state.B), nk1);

  // Sample half-vector, microfacet normal.
  const float3 h0 = hvd_ggx_sample_vndf(k10, state.roughness, state.xi);

  if (fabsf(h0.z) == 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * state.T + h0.y * state.B + h0.z * state.N;

  const float kh = dot(state.k1, h);

  if (kh <= 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  // BRDF: reflect
  state.k2            = (2.0f * kh) * h - state.k1;
  state.bsdf_over_pdf = make_float3(1.0f); // PERF Always white with the original setup.
  state.typeEvent     = BSDF_EVENT_GLOSSY_REFLECTION;

  // Check if the resulting direction is on the correct side of the actual geometry
  const float gnk2 = dot(state.k2, state.Ng); // * ((state.typeEvent == BSDF_EVENT_GLOSSY_REFLECTION) ? 1.0f : -1.0f);
  
  if (gnk2 <= 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  const float nk2 = fabsf(dot(state.k2, state.N));
  const float k2h = fabsf(dot(state.k2, h));

  float G1;
  float G2;

  const float G12 = ggx_smith_shadow_mask(G1, G2, k10, make_float3(dot(state.k2, state.T), dot(state.k2, state.B), nk2), state.roughness);
  
  if (G12 <= 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }
  
  state.bsdf_over_pdf *= G12 / G1;

  // Compute pdf
  state.pdf  = hvd_ggx_eval(1.0f / state.roughness, h0) * G1;
  state.pdf *= 0.25f / (nk1 * h0.z);

  if (thinfilm)
  {
    const float3 factor = thin_film_factor(state.iridescenceThickness, state.iridescenceIor, state.ior2, state.ior1, kh);

    switch (lobe)
    {
      case LOBE_SPECULAR_REFLECTION:
        tint *= factor;
        break;

      case LOBE_METAL_REFLECTION:
        tint = mix_rgb(tint, state.specularColor, factor);
        break;
    }
  }

  state.bsdf_over_pdf *= tint;
}


__forceinline__ __device__ float3 brdf_ggx_smith_eval(State& state, const int lobe, float3 tint, const bool thinfilm)
{
  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Include edge-on (== 0.0f) as "no light" case.
  const bool backside = (dot(state.k2, state.Ng) <= 0.0f);
  // Nothing to evaluate for given directions?
  if (backside) // && scatter_reflect
  {
    state.pdf = 0.0f;
    return make_float3(0.0f);
  }
  
  const float nk1 = fabsf(dot(state.k1, state.N));
  const float nk2 = fabsf(dot(state.k2, state.N));
  
  // compute_half_vector() for scatter_reflect.
  const float3 h = normalize(state.k1 + state.k2);

  // Invalid for reflection / refraction?
  const float nh  = dot(state.N,  h);
  const float k1h = dot(state.k1, h);
  const float k2h = dot(state.k2, h);

  // nk1 and nh must not be 0.0f or state.pdf == NaN.
  if (nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    state.pdf = 0.0f;
    return make_float3(0.0f);
  }

  // Compute BSDF and pdf.
  const float3 h0 = make_float3(dot(state.T, h), dot(state.B, h), nh);

  state.pdf = hvd_ggx_eval(1.0f / state.roughness, h0);

  float G1;
  float G2;

  const float G12 = ggx_smith_shadow_mask(G1, G2, 
                                          make_float3(dot(state.T, state.k1), dot(state.B, state.k1), nk1),
                                          make_float3(dot(state.T, state.k2), dot(state.B, state.k2), nk2),
                                          state.roughness);
  state.pdf *= 0.25f / (nk1 * nh);

  float3 bsdf = make_float3(G12 * state.pdf);
  
  state.pdf *= G1;

  if (thinfilm)
  {
    const float3 factor = thin_film_factor(state.iridescenceThickness, state.iridescenceIor, state.ior2, state.ior1, k1h);

    switch (lobe)
    {
      case LOBE_SPECULAR_REFLECTION:
        tint *= factor;
        break;

      case LOBE_METAL_REFLECTION:
        tint = mix_rgb(tint, state.specularColor, factor);
        break;
    }
  }

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  return bsdf * tint;
}


__forceinline__ __device__ void btdf_ggx_smith_sample(State& state, const float3 tint)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  state.bsdf_over_pdf = make_float3(0.0f);
  state.pdf           = 0.0f;

  const float2 ior = make_float2(state.ior1, state.ior2); 

  const float nk1 = fabsf(dot(state.k1, state.N));

  const float3 k10 = make_float3(dot(state.k1, state.T),
                                 dot(state.k1, state.B),
                                 nk1);

  // Sample half-vector, microfacet normal.
  const float3 h0 = hvd_ggx_sample_vndf(k10, state.roughness, state.xi);

  if (fabsf(h0.z) == 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * state.T + h0.y * state.B + h0.z * state.N;

  const float kh = dot(state.k1, h);

  if (kh <= 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  // Case scatter_transmit
  bool tir = false;

  if (state.isThinWalled) // No refraction!
  {
    // pseudo-BTDF: flip a reflected reflection direction to the back side
    state.k2 = (2.0f * kh) * h - state.k1;
    state.k2 = normalize(state.k2 - 2.0f * state.N * dot(state.k2, state.N));
  }
  else
  {
    // BTDF: refract
    state.k2 = refract(state.k1, h, ior.x / ior.y, kh, tir);
  }

  state.bsdf_over_pdf = make_float3(1.0f); // Was: (make_float3(1.0f) - fr) / prob; // PERF Always white with the original setup.
  state.typeEvent     = (tir) ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;

  // Check if the resulting direction is on the correct side of the actual geometry
  const float gnk2 = dot(state.k2, state.Ng) * ((state.typeEvent == BSDF_EVENT_GLOSSY_REFLECTION) ? 1.0f : -1.0f);
  
  if (gnk2 <= 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  const float nk2 = fabsf(dot(state.k2, state.N));
  const float k2h = fabsf(dot(state.k2, h));

  float G1;
  float G2;

  const float G12 = ggx_smith_shadow_mask(G1, G2, k10, make_float3(dot(state.k2, state.T), dot(state.k2, state.B), nk2), state.roughness);
  
  if (G12 <= 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }
  
  state.bsdf_over_pdf *= G12 / G1;

  // Compute pdf
  state.pdf = hvd_ggx_eval(1.0f / state.roughness, h0) * G1; // * prob;

  if (!state.isThinWalled && (state.typeEvent == BSDF_EVENT_GLOSSY_TRANSMISSION)) // if (refraction)
  {
    const float tmp = kh * ior.x - k2h * ior.y;

    state.pdf *= kh * k2h / (nk1 * h0.z * tmp * tmp);
  }
  else
  {
    state.pdf *= 0.25f / (nk1 * h0.z);
  }

  state.bsdf_over_pdf *= tint;
}


__forceinline__ __device__ float3 btdf_ggx_smith_eval(State& state, const float3 tint)
{
  const float2 ior = make_float2(state.ior1, state.ior2); 

  const float nk1 = fabsf(dot(state.k1, state.N));
  const float nk2 = fabsf(dot(state.k2, state.N));

  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Do NOT include edge-on (== 0.0f) as backside here to take the reflection path.
  const bool backside = (dot(state.k2, state.Ng) < 0.0f);
  
  const float3 h = compute_half_vector(state.k1, state.k2, state.N, ior, nk2, backside, state.isThinWalled);

  // Invalid for reflection / refraction?
  const float nh  = dot(state.N,  h);
  const float k1h = dot(state.k1, h);
  const float k2h = dot(state.k2, h) * (backside ? -1.0f : 1.0f);
  
  // nk1 and nh must not be 0.0f or state.pdf == NaN.
  if (nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    state.pdf = 0.0f; // absorb
    return make_float3(0.0f);
  }

  float fr;

  if (!backside)
  {
    // For scatter_transmit: Only allow TIR with BRDF eval.
    if (!isTIR(ior, k1h))
    {
      state.pdf = 0.0f; // absorb
      return make_float3(0.0f);
    }
    else
    {
      fr = 1.0f;
    }
  }
  else
  {
    fr = 0.0f;
  }

  // Compute BSDF and pdf
  const float3 h0 = make_float3(dot(state.T, h), dot(state.B, h), nh);

  state.pdf = hvd_ggx_eval(1.0f / state.roughness, h0);

  float G1;
  float G2;

  const float G12 = ggx_smith_shadow_mask(G1, G2, 
                                          make_float3(dot(state.T, state.k1), dot(state.B, state.k1), nk1),
                                          make_float3(dot(state.T, state.k2), dot(state.B, state.k2), nk2),
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
  return bsdf * tint;
}


__forceinline__ __device__ void brdf_sheen_sample(State& state, const float xiFlip)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  state.bsdf_over_pdf = make_float3(0.0f);
  state.pdf           = 0.0f;

  const float invRoughness = 1.0f / (state.sheenRoughness * state.sheenRoughness); // Perceptual roughness to alpha G.

  const float nk1 = fabsf(dot(state.k1, state.N));

  const float3 k10 = make_float3(dot(state.k1, state.T), dot(state.k1, state.B), nk1);

  const float3 h0 = flip(hvd_sheen_sample(state.xi, invRoughness), k10, xiFlip);

  if (fabsf(h0.z) == 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * state.T + h0.y * state.B + h0.z * state.N;

  const float k1h = dot(state.k1, h);

  if (k1h <= 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  // BRDF: reflect
  state.k2            = (2.0f * k1h) * h - state.k1;
  state.bsdf_over_pdf = make_float3(1.0f); // PERF Always white with the original setup.
  state.typeEvent     = BSDF_EVENT_GLOSSY_REFLECTION;

  // Check if the resulting reflection direction is on the correct side of the actual geometry.
  const float gnk2 = dot(state.k2, state.Ng);

  if (gnk2 <= 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  const float nk2 = fabsf(dot(state.k2, state.N));
  const float k2h = fabsf(dot(state.k2, h));

  float G1;
  float G2;

  const float G12 = vcavities_shadow_mask(G1, G2, h0.z, 
                                          k10, k1h, 
                                          make_float3(dot(state.k2, state.T), dot(state.k2, state.B), nk2), k2h);
  if (G12 <= 0.0f)
  {
    state.typeEvent = BSDF_EVENT_ABSORB;
    return;
  }
  
  state.bsdf_over_pdf *= G12 / G1;

  // Compute pdf.
  state.pdf = hvd_sheen_eval(invRoughness, h0.z) * G1;

  state.pdf *= 0.25f / (nk1 * h0.z);

  state.bsdf_over_pdf *= state.sheenColor;
}


__forceinline__ __device__ float3 brdf_sheen_eval(State& state)
{
  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Include edge-on (== 0.0f) as "no light" case.
  const bool backside = (dot(state.k2, state.Ng) <= 0.0f);
  // Nothing to evaluate for given directions?
  if (backside) // && scatter_reflect
  {
    state.pdf = 0.0f;
    return make_float3(0.0f);
  }
  
  const float nk1 = fabsf(dot(state.k1, state.N));
  const float nk2 = fabsf(dot(state.k2, state.N));
  
  // compute_half_vector() for scatter_reflect.
  const float3 h = normalize(state.k1 + state.k2);

  // Invalid for reflection / refraction?
  const float nh  = dot(state.N,  h);
  const float k1h = dot(state.k1, h);
  const float k2h = dot(state.k2, h);

  // nk1 and nh must not be 0.0f or state.pdf == NaN.
  if (nk1 <= 0.0f || nh <= 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    state.pdf = 0.0f;
    return make_float3(0.0f);
  }

  const float invRoughness = 1.0f / (state.sheenRoughness * state.sheenRoughness); // Perceptual roughness to alpha G.

  // Compute BSDF and pdf
  const float3 h0 = make_float3(dot(state.T, h), dot(state.B, h), nh);

  state.pdf = hvd_sheen_eval(invRoughness, h0.z);

  float G1;
  float G2;

  const float G12 = vcavities_shadow_mask(G1, G2, h0.z, 
                                          make_float3(dot(state.T, state.k1), dot(state.B, state.k1), nk1), k1h, 
                                          make_float3(dot(state.T, state.k2), dot(state.B, state.k2), nk2), k2h);
  state.pdf *= 0.25f / (nk1 * nh);

  const float3 bsdf = make_float3(G12 * state.pdf);
  
  state.pdf *= G1;

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  return bsdf * state.sheenColor; // Note, not using state.tint here.
}


// CH for radiance rays, triangles, spheres.
// This shader handles every supported feature of the renderer.
// Recursive: shoots a shadow ray.
template<typename MESH> 
__forceinline__ __device__ void chRadiance()
{
  //const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex()); // DEBUG

  const HitGroupData* hitGroupData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

  const MESH&         mesh     = hitGroupData->geometryData.getMesh<MESH>();
  const MaterialData& material = hitGroupData->materialData;

  PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

  thePrd->flags |= FLAG_HIT; // Required to distinguish surface hits from random walk miss.

  thePrd->distance = optixGetRayTmax(); // Return the current path segment distance
  
  // PRECISION Calculate this from the object space vertex positions and transform to world for better accuracy when needed.
  // Same as: thePrd->pos = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
  thePrd->pos += thePrd->wi * thePrd->distance;

  if (0.0f <= theLaunchParameters.picking.x)
  {
    thePrd->indexMaterial = material.index;
    thePrd->radiance      = make_float3(0.0f);
    thePrd->typeEvent     = BSDF_EVENT_ABSORB;
    return;
  }

  // If we're inside a volume and hit something, the path throughput needs to be modulated
  // with the transmittance along this segment before adding surface or light radiance!
  if (0 < thePrd->idxStack) // This assumes the first stack entry is vaccuum.
  {
    thePrd->throughput *= expf(thePrd->sigma_t * -thePrd->distance);

    // Increment the volume scattering random walk counter.
    // Unused when FLAG_VOLUME_SCATTERING is not set.
    //++thePrd->walk;
  }

  // Save the current path throughput for the lighting contributions.
  // The current thePrd->throughput will be modulated by the BXDF sampling result before that.
  const float3 throughput = thePrd->throughput;

  // KHR_materials_unlit handling only renders the baseColor. 
  // No need to initialize the whole state.
  if (material.unlit || theLaunchParameters.forceUnlit)
  {
    thePrd->radiance += throughput * getBaseColor(mesh, material);
    thePrd->typeEvent = BSDF_EVENT_ABSORB;
    return;
  }

  State state;

  initializeState(state/* out */, mesh, material);

  // Explicitly include edge-on cases as frontface condition!
  // Keeps the material stack from overflowing at silhouettes.
  // Prevents that silhouettes of thin-walled materials use the backface material.
  // Using the true geometry normal attribute as originally defined on the frontface!
  const bool isFrontFace = (0.0f <= dot(thePrd->wo, state.Ng));

  // Flip the normals to the side the ray hit.
  if (!isFrontFace)
  {
    state.Ng = -state.Ng;
    state.N  = -state.N;
  }

  int idx = thePrd->idxStack;

  // If the hit is either on the surface or a thin-walled material,
  // the ray is inside the surrounding material and the material ior is on the other side.
  if (isFrontFace || state.isThinWalled)
  {
    state.ior1 = thePrd->stack[idx].absorption_ior.w; // From surrounding medium ior
    state.ior2 = material.ior;                        // to material ior.
  }
  else
  {
    // When hitting the backface of a non-thin-walled material, 
    // the ray is inside the current material and the surrounding material is on the other side.
    // That material's IOR is the current top-of-stack after the previous transmission. We need the one further down!
    idx = max(0, idx - 1);

    state.ior1 = material.ior;                        // From material ior
    state.ior2 = thePrd->stack[idx].absorption_ior.w; // to surrounding medium ior.
  }

  // This GGX implementation doesn't handle identical IORs!
  // Fix the state.ior values to have a tiny difference.
  const float iorDiff = state.ior2 - state.ior1;

  if (fabsf(iorDiff) < IOR_THRESHOLD)
  {
    state.ior2 = state.ior1 + copysignf(IOR_THRESHOLD, iorDiff);
  }

  // Do not apply clearcoat to backfaces of non-thinwalled geometry 
  // because that would interact with the material.ior which the spec says it doesn't.
  // fabsf(cosTheta)) because this is before flipping the state.Nc to the ray side.
  const float frCoat = (0.0f < state.clearcoat && (isFrontFace || state.isThinWalled)) 
                     ? state.clearcoat * ior_fresnel(1.5f / state.ior1, fabsf(dot(thePrd->wo, state.Nc)))
                     : 0.0f;

  // FIXME Implement arbitrary mesh lights. This only handles geometry emission on implicit hits.
  // Unfortunately many emission textures in glTF models are applied to the whole models 
  // with only little emissive regions and the rest of the texture being black.
  // That would need a pre-process emissive texture integration over the geometry to identify
  // which triangles are emissive at all and only put these into the light definitions
  // for explicit mesh light sampling.
  // 
  // This implements a diffuse EDF for implicit light hits.
  // (I disagree with the KHR_materials_clearcoat extension.)
  thePrd->radiance += throughput * state.emission; 

  // Start fresh with the next BSDF sample.
  // The pdf of the previous event was needed for the emission calculation above.
  thePrd->pdf = 0.0f;

  // BXDF sampling

  // Calculate the weights of the individual lobes inside the standard PBR material.
  // Sampling has no value for the half-vector because there is no continuation ray. 
  // Means the fresnel weight needs to be estimated.
  // KHR_materials_specular allows controlling the weight (and color) of the dielectric specular layer.
  
  // Dielectric diffuse base and transmission are affected by the iridescence on the dielectric layer.
  // Calculate a tint coor for these when necessary.
  float3 tint = state.baseColor; // Default for the dielectric base lobes is the baseColor.

  // This Fresnel value defines the weighting between dielectric specular reflection and
  // the base dielectric BXDFs (diffuse reflection and specular transmission).
  float  frDielectric  = 0.0f;
  float3 frIridescence = make_float3(0.0f);

  const float VdotN = dot(thePrd->wo, state.N); // cosTheta of the view vector to the shading normal.
  // This Monte Carlo decision takes care of the mix between non-iridescence and iridescence cases.
  // For state.iridescence == 0.0f thinfilm is always false, for state.iridescence == 1.0f thinfilm is always true.
  const bool thinfilm = (rng(thePrd->seed) < state.iridescence);

  // Estimate the iridescence Fresnel factor with the angle to the normal. That's good enough for specular reflections.
  if (thinfilm)
  {
    // When there is iridescence enabled, use the maximum of the estimated iridescence factor. (Estimated with VdotN, no half-vector H here.) 
    // With the thinfilm decision this handles the mix between non-iridescence and iridescence strength automatically.
    frIridescence = thin_film_factor(state.iridescenceThickness, state.iridescenceIor, state.ior2, state.ior1, VdotN);
    frDielectric  = fmaxf(frIridescence);
    // Modulate the dielectric base lobe (diffuse, transmission) colors by the inverse of the iridescence factor,
    // though use the maximum component to not actually generate inverse colors.
    tint = mix_rgb(tint, state.specularColor, frIridescence);
  }
  else if (0.0f < state.specular)
  {
    frDielectric = state.specular * ior_fresnel(state.ior2 / state.ior1, VdotN);
  }

  float weightLobe[NUM_LOBES]; // The sum of these weights is always 1.0f by construction! No need for a CDF.
  float weightBase;            // The current weight factor on the remaining lobes.

  weightLobe[LOBE_CLEARCOAT_REFLECTION] = frCoat; // BRDF clearcoat (GGX-Smith). (Never sampled when state.clearcoat == 0.0f.)
  weightBase = 1.0f - frCoat;

  // The sheen doesn't have an explicit weight factor. Instead it's disabled when the sheenColor is black.
  weightLobe[LOBE_SHEEN_REFLECTION] = 0.0f; // Default for sheen is off, weightBase doesn't change then.
  if (isNotNull(state.sheenColor)) // Add sheen when the state.sheenColor is not black!
  {
    // The sheen_lut.hdr for this sampling weight estimation was generated offline
    // and is uploaded as R32F texture to the textureSheenLUT object.
    const float sheen = tex2D<float>(theLaunchParameters.textureSheenLUT, fabsf(VdotN), state.sheenRoughness); // BRDF sheen

    weightLobe[LOBE_SHEEN_REFLECTION] = weightBase * sheen;
    weightBase *= 1.0f - sheen;
  }

  weightLobe[LOBE_METAL_REFLECTION] = weightBase * state.metallic; // BRDF metal (GGX-Smith)
  weightBase *= 1.0f - state.metallic;

  weightLobe[LOBE_SPECULAR_REFLECTION] = weightBase * frDielectric; // BRDF dielectric specular reflection (GGX-Smith)
  weightBase *= 1.0f - frDielectric;

  weightLobe[LOBE_SPECULAR_TRANSMISSION] = weightBase * state.transmission; // BTDF dielectric specular transmission (GGX-Smith)
  weightLobe[LOBE_DIFFUSE_REFLECTION]    = weightBase * (1.0f - state.transmission); // BRDF diffuse dielectric reflection (Lambert). // PERF Currently not referenced below.

  // Sample one of the material lobes according to their estimated weight.
  const float sampleLobe = rng(thePrd->seed);
  int lobe = NUM_LOBES;
  float weight = 0.0f;
  while (--lobe) // Stops when lobe reaches 0!
  {
    weight += weightLobe[lobe];
    if (sampleLobe < weight)
    {
      break; // Sample and evaluate this lobe!
    }
  }

  state.xi = rng2(thePrd->seed);
  state.k1 = thePrd->wo;         // == -optixGetWorldRayDirection()

  // Ambient occlusion should not really be required with a global illumination renderer
  // but many glTF models are very low-resolution geometry and details are baked into normal and occlusion maps.
  // Apply the occlusion value to diffuse and metal reflection lobes when lit by environment lights.
  thePrd->occlusion = 1.0f;
  
  switch (lobe) 
  {
    case LOBE_DIFFUSE_REFLECTION:
      brdf_diffuse_sample(state, tint);
      // Store the occlusion factor of the last event to be able to modulate the implicit next environment light hit with it.
      thePrd->occlusion = state.occlusion; // diffuse reflection is affected by ambient occlusion.
      break;

    case LOBE_SPECULAR_TRANSMISSION:
      btdf_ggx_smith_sample(state, tint);
      break;

    case LOBE_SPECULAR_REFLECTION:
      brdf_ggx_smith_sample(state, lobe, state.specularColor, thinfilm);
      break;

    case LOBE_METAL_REFLECTION:
      brdf_ggx_smith_sample(state, lobe, state.baseColor, thinfilm);
      thePrd->occlusion = state.occlusion; // metal (glossy) reflection is affected by ambient occlusion.
      break;

    case LOBE_SHEEN_REFLECTION:
      // Sheen is using the state.sheenColor and state.sheenInvRoughness values directly.
      // Only brdf_sheen_sample needs a third random sample for the v-cavities flip. Put this as argument.
      brdf_sheen_sample(state, rng(thePrd->seed)); 
      break;

    case LOBE_CLEARCOAT_REFLECTION:
      // Sample the clearcoat (with the clearcoat normal and roughness).
      // DEBUG Does KHR_materials_anisotropy apply to clearcoat? This assumes it does not.
      state.roughness = make_float2(state.clearcoatRoughness * state.clearcoatRoughness);
      // Regenerate shading space using the clearcoat normal, using the original shading tangent attribute.
      state.N = state.Nc; 
      state.B = normalize(cross(state.Nc, state.Tc)); // Assumes Nc and Tc are not collinear!
      state.T = cross(state.B, state.Nc);
      state.B *= state.handedness;
      if (!isFrontFace)
      {
        state.N = -state.N;
      }
      // Clearcoat is always white and not affected by iridescence.
      brdf_ggx_smith_sample(state, lobe, make_float3(1.0f), false);
      break;
  }

  thePrd->wi          = state.k2;            // Continuation direction.
  thePrd->throughput *= state.bsdf_over_pdf; // Adjust the path throughput for all following incident lighting.
  thePrd->pdf         = state.pdf;           // Note that specular events in MDL return pdf == 0.0f! (=> Not a path termination condition.)
  thePrd->typeEvent   = state.typeEvent;     // If this is BSDF_EVENT_ABSORB, the path ends inside the integrator and the radiance is returned.

  // End of BXDF sampling

  // Update the material stack after sampling when there was a transmission event.
  if (!state.isThinWalled && (state.typeEvent & BSDF_EVENT_TRANSMISSION) != 0)
  {
    int idx;

    if (isFrontFace) // Entered a volume. 
    {
      idx = min(thePrd->idxStack + 1, MATERIAL_STACK_LAST); // Push current medium parameters.

      thePrd->idxStack = idx;

      // KHR_material_volume
      // attenuationDistance is in the range (0.0f, +inf).
      // That's an open interval, so glTF specs require attenuationDistance != 0.0f.
      // The application default is attenuationDistance = RT_DEFAULT_MAX instead of INF though.
      // FIXME This logf() won't work with any zero attenuationColor components.
      const float3 absorption = (material.attenuationDistance < RT_DEFAULT_MAX)
                              ? logf(material.attenuationColor) / -material.attenuationDistance
                              : make_float3(0.0f); // No absorption.

      thePrd->stack[idx].absorption_ior  = make_float4(absorption, material.ior);
      //thePrd->stack[idx].scattering_bias = material.scattering_bias;
    }
    else // if !isFrontFace. Left a volume.
    {
      idx = max(0, thePrd->idxStack - 1); // Pop current medium parameters.

      thePrd->idxStack = idx;
    }

    // Update the extinction coefficient sigma_t.
    thePrd->sigma_t = make_float3(thePrd->stack[idx].absorption_ior);   // sigma_a
                    //+ make_float3(thePrd->stack[idx].scattering_bias); // + sigma_s

    //thePrd->walk = 0; // Reset the number of random walk steps taken when crossing any volume boundary.
  }

  // Direct lighting if the sampled BSDF was diffuse or glossy and any light is in the scene.
  // PERF We know  that the sampling was glossy when this is reached. 
  // No need to to check the thePrd->eventType to see if it's diffuse oor glossy which can handle direct lighting.
  const int numLights = theLaunchParameters.numLights;

  if (!theLaunchParameters.directLighting || numLights <= 0)
  {
    return;
  }

  // Sample one of many lights.
  // The caller picks the light to sample. Make sure the index stays in the bounds of the sysData.lightDefinitions array.
  const int indexLight = (1 < numLights) ? clamp(static_cast<int>(floorf(rng(thePrd->seed) * numLights)), 0, numLights - 1) : 0;
  
  const LightDefinition& light = theLaunchParameters.lightDefinitions[indexLight];

  // There are only light sampling callables inside the SBT. The typeLight is also the index into the callables.
  LightSample lightSample = optixDirectCall<LightSample, const LightDefinition&, PerRayData*>(light.typeLight, light, thePrd);

  // No direct lighting if the light sample is invalid.
  if (lightSample.pdf <= 0.0f)
  {
    return;
  }

  // Now that we have an incoming light direction, evaluate the sampled BXDF lobe with that. 
  // All other state is unchanged from the sampling above.
  state.k2 = lightSample.direction;
  
  float3 bxdf;

  // Mind that the state is still setup for the respective lobe!
  switch (lobe) 
  {
    case LOBE_DIFFUSE_REFLECTION:
      bxdf = brdf_diffuse_eval(state, tint);
      break;

    case LOBE_SPECULAR_TRANSMISSION:
      bxdf = btdf_ggx_smith_eval(state, tint);
      break;

    case LOBE_SPECULAR_REFLECTION:
      bxdf = brdf_ggx_smith_eval(state, lobe, state.specularColor, thinfilm);
      break;

    case LOBE_METAL_REFLECTION:
      bxdf = brdf_ggx_smith_eval(state, lobe, state.baseColor, thinfilm);
      break;

    case LOBE_SHEEN_REFLECTION:
      bxdf = brdf_sheen_eval(state);
      break;

    case LOBE_CLEARCOAT_REFLECTION:
      bxdf = brdf_ggx_smith_eval(state, lobe, make_float3(1.0f), false);
      break;
  }

  const float pdf = state.pdf;

  if (pdf <= 0.0f || isNull(bxdf))
  {
    return;
  }

  // The shadow ray is only a single payload to indicate the visibility test result.
  // Default to visibilty being blocked by geometry. If the miss shader is reached this gets set to 1.
  unsigned int isVisible = 0;
  unsigned int seed      = thePrd->seed;

  // Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval 
  // to prevent self-intersections with the actual light geometry in the scene.
  optixTrace(theLaunchParameters.handle,
             thePrd->pos, lightSample.direction, // origin, direction
             theLaunchParameters.sceneEpsilon, lightSample.distance - theLaunchParameters.sceneEpsilon, 0.0f, // tmin, tmax, time
             OptixVisibilityMask(0xFF),
             OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
             TYPE_RAY_SHADOW, NUM_RAY_TYPES, TYPE_RAY_SHADOW, // The shadow ray type only uses the miss program.
             isVisible, seed);

  thePrd->seed = seed; // Update the seed from sampling inside the anyhit program invocations.

  if (isVisible)
  {
    // Only do multiple-importance sampling for area lights, not singular lights.
    const float weightMIS = (light.typeLight <= TYPE_LIGHT_ENV_SPHERE) ? balanceHeuristic(lightSample.pdf, pdf) : 1.0f;

    // The sampled emission needs to be scaled by the inverse probability to have selected this light,
    // Selecting one of many lights means the inverse of 1.0f / numLights.
    // This is using the path throughput before the sampling modulated it above.
    thePrd->radiance += throughput * bxdf * lightSample.radiance_over_pdf * (float(numLights) * weightMIS);
  }
}


// AH for radiance rays. Mesh types: triangles, spheres.
template <typename MESH>
__forceinline__ __device__ void ahRadiance()
{
  // Mind that geometric primitives with ALPHA_MODE_OPAQUE never reach anyhit programs due to the geometry flags.
  const HitGroupData* hitGroupData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const MaterialData& material = hitGroupData->materialData;

  const float alpha = getOpacity(hitGroupData->geometryData.getMesh<MESH>(), material);

  float cutoff = material.alphaCutoff;

  if (material.alphaMode == MaterialData::ALPHA_MODE_BLEND)
  {
    PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

    // This stochastic opacity must only be evaluated once per primitive.
    // The AS is built with OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL for ALPHA_MODE_BLEND.
    cutoff = rng(thePrd->seed);
  }

  if (alpha < cutoff)
  {
    optixIgnoreIntersection(); // Transparent.
  }
}


extern "C" __global__ void __anyhit__radiance()
{
  ahRadiance<GeometryData::TriangleMesh>();
}


extern "C" __global__ void __anyhit__radiance_sphere()
{
  ahRadiance<GeometryData::SphereMesh>();
}


extern "C" __global__ void __closesthit__radiance()
{
  chRadiance<GeometryData::TriangleMesh>();
}


extern "C" __global__ void __closesthit__radiance_sphere()
{
  chRadiance<GeometryData::SphereMesh>();
}


// AH for shadow rays. Mesh types: triangles, spheres.
template <typename MESH>
__forceinline__ __device__ void ahShadow()
{
  // Mind that geometric primitives with ALPHA_MODE_OPAQUE never reach anyhit programs due to the geometry flags.
  const HitGroupData* hitGroupData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const MaterialData& material = hitGroupData->materialData;

  const float alpha = getOpacity(hitGroupData->geometryData.getMesh<MESH>(), material);

  float cutoff = material.alphaCutoff;

  if (material.alphaMode == MaterialData::ALPHA_MODE_BLEND)
  {
    unsigned int seed = optixGetPayload_1();

    // This stochastic opacity must only be evaluated once per primitive.
    // The AS is built with OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL for ALPHA_MODE_BLEND.
    cutoff = rng(seed);

    optixSetPayload_1(seed); // Write changed seed back to payload.
  }

  if (alpha < cutoff)
  {
    optixIgnoreIntersection(); // Transparent.
  }
  else
  {
    optixTerminateRay(); // Opaque. isVisible == 0 because the __miss__shadow program is not invoked.
  }
}

// Triangles
extern "C" __global__ void __anyhit__shadow()
{
  ahShadow<GeometryData::TriangleMesh>();
}


extern "C" __global__ void __anyhit__shadow_sphere()
{
  ahShadow<GeometryData::SphereMesh>();
}

// NOTE PERFO handling different primitives (Triangles,Spheres) in the same function is faster
// than having different functions, according to OptiX programming guide.