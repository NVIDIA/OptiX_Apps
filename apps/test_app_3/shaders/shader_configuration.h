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
 
#ifndef DEVICE_SHADER_CONFIGURATION_H
#define DEVICE_SHADER_CONFIGURATION_H

#include "config.h"

#include <cuda.h>

// Defines for the DeviceShaderConfiguration::flags
#define IS_THIN_WALLED     (1u << 0)
// These flags are used to control which specific hit record is used.
#define USE_EMISSION       (1u << 1)
#define USE_CUTOUT_OPACITY (1u << 2)

struct DeviceShaderConfiguration
{
  unsigned int flags; // See defines above.

  int idxCallInit; // The material global init function.
  
  int idxCallThinWalled;

  int idxCallSurfaceScatteringSample;
  int idxCallSurfaceScatteringEval;

  int idxCallBackfaceScatteringSample;
  int idxCallBackfaceScatteringEval;

  int idxCallSurfaceEmissionEval;
  int idxCallSurfaceEmissionIntensity;
  int idxCallSurfaceEmissionIntensityMode;

  int idxCallBackfaceEmissionEval;
  int idxCallBackfaceEmissionIntensity;
  int idxCallBackfaceEmissionIntensityMode;

  int idxCallIor;

  int idxCallVolumeAbsorptionCoefficient;
  int idxCallVolumeScatteringCoefficient;
  int idxCallVolumeDirectionalBias;

  int idxCallGeometryCutoutOpacity;

  int idxCallHairSample;
  int idxCallHairEval;

  // The constant expression values:
  //bool thin_walled; // Stored inside flags.
  float3 surface_intensity;
  int    surface_intensity_mode;
  float3 backface_intensity;
  int    backface_intensity_mode;
  float3 ior;
  float3 absorption_coefficient;
  float3 scattering_coefficient;
  float  directional_bias;
  float  cutout_opacity;
};

#endif // DEVICE_SHADER_CONFIGURATION_H
