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

#pragma once

#ifndef MATERIAL_DEFINITION_H
#define MATERIAL_DEFINITION_H

#include "function_indices.h"

// Just some hardcoded material parameter system which allows to show a few fundamental BSDFs.
struct MaterialDefinition
{
  // 16 byte alignment
  float4 absorption_ior;  // .xyz = volume absorption coefficient, .w = index of refraction.
  float4 scattering_bias; // .xyz = volume scattering coefficient, .w = directional bias.

  // 8 byte alignment.
  cudaTextureObject_t textureAlbedo;    // Modulates albedo when valid.

  float2 roughness; // Anisotropic roughness values.

  // 4 byte alignment.
  TypeBXDF typeBXDF; // Zero-based BXDF index. Only used to select the instance sbtOffset during TLAS builds.

  float3 albedo;     // Albedo, tint, throughput change for specular surfaces. Pick your meaning.
  
  unsigned int flags; // Thin-walled on/off

  // Manual padding to 16-byte alignment goes here.
  int pad0;
  int pad1;
  int pad2;
};

#endif // MATERIAL_DEFINITION_H
