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

#pragma once

#ifndef COMPILE_RESULT_H
#define COMPILE_RESULT_H

#include <mi/mdl_sdk.h>
#include <mi/base/config.h>

#include <string>
#include <vector>


/// Result of a material compilation.
struct Compile_result
{
  /// The compiled material.
  mi::base::Handle<mi::neuraylib::ICompiled_material const> compiled_material;

  /// The generated target code object.
  mi::base::Handle<mi::neuraylib::ITarget_code const> target_code;

  /// The argument block for the compiled material.
  mi::base::Handle<mi::neuraylib::ITarget_argument_block const> argument_block;

  /// Information required to load a texture.
  struct Texture_info
  {
    std::string                                db_name;
    mi::neuraylib::ITarget_code::Texture_shape shape;

    Texture_info()
      : shape(mi::neuraylib::ITarget_code::Texture_shape_invalid)
    {
    }

    Texture_info(char const* db_name,
                  mi::neuraylib::ITarget_code::Texture_shape shape)
      : db_name(db_name)
      , shape(shape)
    {
    }
  };

  /// Information required to load a light profile.
  struct Light_profile_info
  {
    std::string db_name;

    Light_profile_info()
    {
    }

    Light_profile_info(char const* db_name)
      : db_name(db_name)
    {
    }
  };

  /// Information required to load a BSDF measurement.
  struct Bsdf_measurement_info
  {
    std::string db_name;

    Bsdf_measurement_info()
    {
    }

    Bsdf_measurement_info(char const* db_name)
      : db_name(db_name)
    {
    }
  };

  /// Textures used by the compile result.
  std::vector<Texture_info> textures;

  /// Light profiles used by the compile result.
  std::vector<Light_profile_info> light_profiles;

  /// Bsdf_measurements used by the compile result.
  std::vector<Bsdf_measurement_info> bsdf_measurements;

  /// Constructor.
  Compile_result()
  {
    // Add invalid resources.
    textures.emplace_back();
    light_profiles.emplace_back();
    bsdf_measurements.emplace_back();
  }
};

#endif // COMPILE_RESULT_H
