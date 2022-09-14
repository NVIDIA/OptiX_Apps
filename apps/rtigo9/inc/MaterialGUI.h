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

#ifndef MATERIAL_GUI_H
#define MATERIAL_GUI_H

#include "shaders/function_indices.h"

#include <string>

// Host side material parameters.
// FIXME Add a flag if a material is referenced inside the scene. Then avoid loading unused textures and profiles.
struct MaterialGUI
{
  TypeBXDF typeBXDF; // Zero-based BXDF type to use for sampling and evaluation of the surface material.
  TypeEDF  typeEDF;  // Zero-based EDF type. Diffuse emissive material when not default TYPE_EDF. 
                     // Used to build LightDefinitions when this material is assigned to meshes.

  std::string name; // The name used in the scene description to reference this material instance.

  std::string nameAlbedo;   // The filename of the albedo texture for this material. Empty when none.
  std::string nameCutout;   // The filename of the cutout opacity texture for this material. Empty when none.
  std::string nameEmission; // The filename of the emission texture. Empty when none.
  std::string nameProfile;  // The filename of the IES light profile when typeEDF is TYPE_EDF_IES, otherwise unused. Empty when none.

  float3 colorAlbedo;         // Tint, throughput change for specular materials.
  float3 colorEmission;       // The emission base color.
  float  multiplierEmission;  // A multiplier on top of colorEmission to get HDR lights.
  float  spotAngle;           // Full cone angle in degrees, means max. 180 degrees is a hemispherical distribution.
  float  spotExponent;        // Exponent on the cosine of the sotAngle, used to generate intensity falloff from spot cone center to outer angle. Set to 0.0 for no falloff.
  float3 colorAbsorption;     // Color and scale together build the absorption coefficient
  float  scaleAbsorption;     // Distance scale on the absorption.
  float2 roughness;           // Anisotropic roughness for microfacet distributions.
  float  ior;                 // Index of Refraction.
  bool   thinwalled;          // Indicates if a material is thin-walled.
                              // This only affects transmissive materials in this renderer implementation.
                              // There is no support for different materials on front- and back-face in this renderer.
                              // Thin-walled surfaces are not a boundary between volume, means there is no refraction or volume effect on these.
};

#endif // MATERIAL_GUI_H
