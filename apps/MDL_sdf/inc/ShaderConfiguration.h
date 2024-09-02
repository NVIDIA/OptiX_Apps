/* 
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef SHADER_CONFIGURATION_H
#define SHADER_CONFIGURATION_H

#include <mi/mdl_sdk.h>
#include <mi/base/config.h>

// This defines the host side shader configuration and is used for code reuse.
// There are as many shader configurations as there are different MDL compiled material hash values.
// This holds all boolean values and constants a renderer will need.
// This will be converted to a DeviceShaderConfiguration structure
// which contains all direct callable indices and the constant parameter values.
struct ShaderConfiguration
{
  // The state of the expressions:
  bool is_thin_walled_constant;
  bool is_surface_bsdf_valid;
  bool is_backface_bsdf_valid;
  bool is_surface_edf_valid;
  bool is_surface_intensity_constant;
  bool is_surface_intensity_mode_constant;
  bool is_backface_edf_valid;
  bool is_backface_intensity_constant;
  bool is_backface_intensity_mode_constant;
  bool use_backface_edf;
  bool use_backface_intensity;
  bool use_backface_intensity_mode;
  bool is_ior_constant;
  bool is_vdf_valid;
  bool is_absorption_coefficient_constant;
  bool use_volume_absorption;
  bool is_scattering_coefficient_constant;
  bool is_directional_bias_constant;
  bool use_volume_scattering;
  bool is_cutout_opacity_constant;
  bool use_cutout_opacity;
  bool is_hair_bsdf_valid;
  
  // The constant expression values:
  bool            thin_walled;
  mi::math::Color surface_intensity;
  mi::Sint32      surface_intensity_mode;
  mi::math::Color backface_intensity;
  mi::Sint32      backface_intensity_mode;
  mi::math::Color ior;
  mi::math::Color absorption_coefficient;
  mi::math::Color scattering_coefficient;
  mi::Float32     directional_bias;          
  mi::Float32     cutout_opacity;

  bool isEmissive() const
  {
    const bool surfaceEmissive  = is_surface_edf_valid  && (!is_surface_intensity_constant  || (is_surface_intensity_constant  && (0.0f < surface_intensity[0]  || 0.0f < surface_intensity[1]  || 0.0f < surface_intensity[2])));
    const bool backfaceEmissive = is_backface_edf_valid && (!is_backface_intensity_constant || (is_backface_intensity_constant && (0.0f < backface_intensity[0] || 0.0f < backface_intensity[1] || 0.0f < backface_intensity[2])));
    const bool thinWalled       = !is_thin_walled_constant || (is_thin_walled_constant && thin_walled);
   
    return surfaceEmissive || (thinWalled && backfaceEmissive);
  }

};

#endif // SHADER_CONFIGURATION_H
