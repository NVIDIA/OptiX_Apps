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
//

#pragma once

#include <cuda_runtime.h>
#include "config.h" // RT_DEFAULT_MAX

// GLTF material extensions used inside the asset encoded as bitfield.
// #define FLAG_KHR_MATERIALS_IOR          0x00000001u
#define FLAG_KHR_MATERIALS_SPECULAR     0x00000002u
#define FLAG_KHR_MATERIALS_TRANSMISSION 0x00000004u
#define FLAG_KHR_MATERIALS_VOLUME       0x00000008u
#define FLAG_KHR_MATERIALS_CLEARCOAT    0x00000010u
#define FLAG_KHR_MATERIALS_ANISOTROPY   0x00000020u
#define FLAG_KHR_MATERIALS_SHEEN        0x00000040u
#define FLAG_KHR_MATERIALS_IRIDESCENCE  0x00000080u


struct MaterialData
{
  enum AlphaMode
  {
    ALPHA_MODE_OPAQUE = 0,
    ALPHA_MODE_MASK   = 1,
    ALPHA_MODE_BLEND  = 2
  };


  struct Texture
  {
    __device__ __forceinline__ operator bool() const
    {
      return object != 0;
    }

    // 4 byte alignment
    int   index = 0;    // texcoord index.
    //float angle = 0.0f; // texture rotation angle in radians. Only required when exposing the KHR_texture_transform parameters inside the GUI.

    // 8 byte alignment
    cudaTextureObject_t object = 0;

    // KHR_texture_transform extension:
    float2 scale       = { 1.0f, 1.0f };
    float2 rotation    = { 0.0f, 1.0f }; // .x = sin(angle), .y = cos(angle)
    float2 translation = { 0.0f, 0.0f };
  };

  // PBR Metallic Roughness parameters:
  float4  baseColorFactor          = { 1.0f, 1.0f, 1.0f, 1.0f };
  float   metallicFactor           = 1.0f;
  float   roughnessFactor          = 1.0f ;
  Texture baseColorTexture;
  Texture metallicRoughnessTexture;
  
  int index; // Material index inside the asset. Needed for picking.

  // Standard Material parameters:
  bool doubleSided = false;

  AlphaMode alphaMode   = ALPHA_MODE_OPAQUE;
  float     alphaCutoff = 0.0f;

  float   normalTextureScale = 1.0f; // "scale" is an additional field inside the NormalTextureInfo.
  Texture normalTexture;

  // Ambient occlusion is not required with a global illumination renderer.
  // Unless the geometry used for baking normals and occlusion information was much higher resolution.
  // Can be disabled globally inside the GUI with m_useAmbientOcclusion.
  // glTF spec: occlusion = 1.0f + occlusionTextureStrength * (occlusionTexture.r - 1.0f);
  float   occlusionTextureStrength = 1.0f; // "strength" is an additional field inside the OcclusionTextureInfo.
  Texture occlusionTexture;                // The red channel contains the occlusion. 

  float   emissiveStrength = 1.0f; // KHR_materials_emissive_strength. Keep separate to be able to set this inside the GUI.
  float3  emissiveFactor   = { 0.0f, 0.0f, 0.0f };
  Texture emissiveTexture;

  // Material features added by GLTF extensions.
  // https://github.com/KhronosGroup/glTF/blob/main/extensions/README.md
  
  unsigned int flags = 0; // Bitfield with FLAG_KHR_* extensions indicating which extension is used by a material.

  // KHR_materials_ior (and the metallic-roughness material models itself)
  float ior = 1.5f;

  // KHR_materials_specular
  float   specularFactor = 1.0f;
  Texture specularTexture;
  float3  specularColorFactor = { 1.0f, 1.0f, 1.0f };
  Texture specularColorTexture;

  // KHR_materials_transmission
  float   transmissionFactor = 0.0f;
  Texture transmissionTexture;

  // KHR_materials_volume
  // PERF Ray tracers know the traveled distance inside the volume and there shouldn't be such thicknessFactor emulating thick-walled geometry.
  // Unfortunately some glTF models like the IridescenceLamp.gltf are using thicknessFactor = 0.05f to define the glass sphere
  // and when not handling that scaling, the absorption would look incorrect inside a ray tracer.
  float   thicknessFactor = 0.0f; // In object space, transformations to world space apply. 0.0f means thin-walled!
  //Texture thicknessTexture; // FIXME Can't really use this reliably inside a raytracer.
  float   attenuationDistance = RT_DEFAULT_MAX; // Default is +Inf. Must not be 0.0f!
  float3  attenuationColor = { 1.0f, 1.0f, 1.0f }; // That means no volume absorption.

  // KHR_materials_clearcoat
  float   clearcoatFactor = 0.0f;
  Texture clearcoatTexture;
  float   clearcoatRoughnessFactor = 0.0f;
  Texture clearcoatRoughnessTexture;
  Texture clearcoatNormalTexture;
  // The KHR_materials_clearcoat says: 
  // "clearcoatNormalTexture may be a reference to the same normal map used by the base material, or any other compatible normal map."
  // But using the same normal texture as the base material implies that the same normal texture scale(!) would need to be applied,
  // otherwise the clearcoat normal wouldn't match the base normal.
  // Indicate that behavior with a boolean which is set to true when the host code finds 
  // that all values inside normalTexture and clearcoatNormalTexture are identical.
  bool isClearcoatNormalBaseNormal = false;

  // KHR_materials_sheen
  float3  sheenColorFactor = { 0.0f, 0.0f, 0.0f };
  Texture sheenColorTexture;
  float   sheenRoughnessFactor = 0.0f;
  Texture sheenRoughnessTexture;

  // KHR_materials_anisotropy
  float   anisotropyStrength = 0.0f;
  float   anisotropyRotation = 0.0f; // FIXME PERF Precalculate sin, cos of this.
  Texture anisotropyTexture;

  // KHR_materials_iridescence
  float   iridescenceFactor = 0.0f;
  Texture iridescenceTexture;
  float   iridescenceIor = 1.3f;
  float   iridescenceThicknessMinimum = 100.0f;
  float   iridescenceThicknessMaximum = 400.0f;
  Texture iridescenceThicknessTexture;

  // KHR_materials_unlit
  bool unlit = false;
};
