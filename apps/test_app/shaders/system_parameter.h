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

#ifndef SYSTEM_PARAMETER_H
#define SYSTEM_PARAMETER_H

#include "light_definition.h"
#include "material_parameter.h"
#include "vertex_attributes.h"

struct SystemParameter
{
  // 8 byte alignment
  OptixTraversableHandle topObject;

  float4* outputBuffer;

  LightDefinition* lightDefinitions;

  MaterialParameter* materialParameters;

  cudaTextureObject_t envTexture;

  float* envCDF_U; // 2D, size (envWidth + 1) * envHeight
  float* envCDF_V; // 1D, size (envHeight + 1)
  
  int2 pathLengths;

  // 4 byte alignment 
  unsigned int envWidth; // The original size of the environment texture.
  unsigned int envHeight;
  float        envIntegral;
  float        envRotation;

  int    iterationIndex;
  float  sceneEpsilon;

  int    numLights;

  int    cameraType;
  float3 cameraPosition;
  float3 cameraU;
  float3 cameraV;
  float3 cameraW;
};


// SBT Record data for the hit group.
struct GeometryInstanceData
{
  int3*             indices;
  VertexAttributes* attributes;
  
  int materialIndex;
  int lightIndex;    // Negative means not a light.
};

#endif // SYSTEM_PARAMETER_H
