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

#ifndef TEXTURE_HANDLER_H
#define TEXTURE_HANDLER_H

#include "config.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <mi/neuraylib/target_code_types.h>

typedef mi::neuraylib::Texture_handler_base Texture_handler_base;

// Custom structure representing an MDL texture.
// Containing filtered and unfiltered CUDA texture objects and the size of the texture.
struct TextureMDL
{
  explicit TextureMDL()
    : filtered_object(0)
    , unfiltered_object(0)
    , size(make_uint3(0, 0, 0))
    , inv_size(make_float3(0.0f, 0.0f, 0.0f))
  {
  }

  explicit TextureMDL(cudaTextureObject_t filtered_object,
                      cudaTextureObject_t unfiltered_object,
                      uint3               size)
    : filtered_object(filtered_object)
    , unfiltered_object(unfiltered_object)
    , size(size)
    , inv_size(make_float3(1.0f / size.x, 1.0f / size.y, 1.0f / size.z))
  {
  }

  cudaTextureObject_t filtered_object;    // Uses filter mode cudaFilterModeLinear.
  cudaTextureObject_t unfiltered_object;  // Uses filter mode cudaFilterModePoint.
  uint3               size;               // Size of the texture, needed for texel access.
  float3              inv_size;           // The inverse values of the size of the texture.
};

// Structure representing an MDL bsdf measurement.
struct Mbsdf
{
  explicit Mbsdf()
  {
    for (int i = 0; i < 2; ++i)
    {
      eval_data[i]              = 0;
      sample_data[i]            = 0;
      albedo_data[i]            = 0;
      angular_resolution[i]     = make_uint2(0u, 0u);
      inv_angular_resolution[i] = make_float2(0.0f, 0.0f);
      has_data[i]               = 0u;
      this->max_albedo[i]       = 0.0f;
      num_channels[i]           = 0;
    }
  }

  void Add(mi::neuraylib::Mbsdf_part part,
           const uint2& angular_resolution,
           unsigned int num_channels)
  {
    unsigned int part_idx = static_cast<unsigned int>(part);

    this->has_data[part_idx]               = 1u;
    this->angular_resolution[part_idx]     = angular_resolution;
    this->inv_angular_resolution[part_idx] = make_float2(1.0f / float(angular_resolution.x),
                                                         1.0f / float(angular_resolution.y));
    this->num_channels[part_idx]           = num_channels;
  }

  // 8 byte aligned
  cudaTextureObject_t eval_data[2];               // uses filter mode cudaFilterModeLinear
  float*              sample_data[2];             // CDFs for sampling a BSDF measurement
  float*              albedo_data[2];             // max albedo for each theta (isotropic)
  uint2               angular_resolution[2];      // size of the dataset, needed for texel access
  float2              inv_angular_resolution[2];  // the inverse values of the size of the dataset
  // 4 byte alignment.
  unsigned int        has_data[2];                // true if there is a measurement for this part
  float               max_albedo[2];              // max albedo used to limit the multiplier
  unsigned int        num_channels[2];            // number of color channels (1 or 3)
};

// Structure representing a Light Profile
struct Lightprofile
{
  explicit Lightprofile(cudaTextureObject_t eval_data = 0,
                        float* cdf_data = nullptr,
                        uint2 angular_resolution = make_uint2(0, 0),
                        float2 theta_phi_start = make_float2(0.0f, 0.0f),
                        float2 theta_phi_delta = make_float2(0.0f, 0.0f),
                        float candela_multiplier = 0.0f,
                        float total_power = 0.0f)
    : eval_data(eval_data)
    , cdf_data(cdf_data)
    , angular_resolution(angular_resolution)
    , inv_angular_resolution(make_float2(1.0f / float(angular_resolution.x), 
                                         1.0f / float(angular_resolution.y)))
    , theta_phi_start(theta_phi_start)
    , theta_phi_delta(theta_phi_delta)
    , theta_phi_inv_delta(make_float2(0.0f, 0.0f))
    , candela_multiplier(candela_multiplier)
    , total_power(total_power)
  {
    theta_phi_inv_delta.x = (theta_phi_delta.x) ? (1.0f / theta_phi_delta.x) : 0.0f;
    theta_phi_inv_delta.y = (theta_phi_delta.y) ? (1.0f / theta_phi_delta.y) : 0.0f;
  }

  // 8 byte aligned
  cudaTextureObject_t eval_data;              // normalized data sampled on grid
  float*              cdf_data;               // CDFs for sampling a light profile
  uint2               angular_resolution;     // angular resolution of the grid
  float2              inv_angular_resolution; // inverse angular resolution of the grid
  float2              theta_phi_start;        // start of the grid
  float2              theta_phi_delta;        // angular step size
  float2              theta_phi_inv_delta;    // inverse step size
  // 4 byte aligned
  float               candela_multiplier;     // factor to rescale the normalized data
  float               total_power;
};


// The texture handler structure required by the MDL SDK with custom additional fields.
struct Texture_handler: Texture_handler_base
{
  // Additional data for the texture access functions can be provided here.
  // These fields are used inside the MDL runtime functions implemented inside texture_lookup.h.
  // The base class is containing a pointer to a virtual table which is not used in this renderer configuration.

  // 8 byte aligned.
  const TextureMDL*   textures;      // The textures used by the material (without the invalid texture).
  const Mbsdf*        mbsdfs;        // The measured BSDFs used by the material (without the invalid mbsdf).
  const Lightprofile* lightprofiles; // The light_profile objects used by the material (without the invalid light profile).

  // 4 byte aligned.
  unsigned int num_textures;      // The number of textures used by the material (without the invalid texture).
  unsigned int num_mbsdfs;        // The number of mbsdfs used by the material (without the invalid mbsdf).
  unsigned int num_lightprofiles; // number of elements in the lightprofiles field (without the invalid light profile).

  unsigned int pad0; // Make sure the structure is a multiple of 8 bytes in size.
};

#endif // TEXTURE_HANDLER_H
