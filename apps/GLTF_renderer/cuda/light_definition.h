/* 
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef LIGHT_DEFINITION_H
#define LIGHT_DEFINITION_H

#include "function_indices.h"

struct __align__(16) LightDefinition
{
  // 16 byte alignment
  // For environment lights, only the upper 3x3 matrix with the rotational part is used.
  // There is no translation or scale in them then. 
  float4 matrix[3];    // Object to world transformation.
  float4 matrixInv[3]; // World to object transformation.
  
  // 8 byte alignment
  // Importance sampling information for the emission texture.
  // Used by environment lights when textured.
  CUdeviceptr cdfU; // float 2D, (width  + 1) * height elements.
  CUdeviceptr cdfV; // float 1D, (height + 1) elements.

  // Emisson texture. If not zero, scales emission.
  cudaTextureObject_t textureEmission;

  // 4 byte alignment
  float3 emission; // Emission of the light (HDR). (color * intensity)

  TypeLight typeLight;
  
  float area;        // Currently only needed to convert directional light lux units to radiance.
  float invIntegral; // The inverse of the spherical environment map or rectangle light texture map integral over the whole texture.

  // Emission texture width and height. Used to index the CDFs, see above.
  unsigned int width; 
  unsigned int height;

  float range; // Point light and spot light maximum distance at which they have an effect.

  float cosInner; // Cosine of spot light innerConeAngle.
  float cosOuter; // Cosine of spot light outerConeAngle

  // Structure size padding to multiple of 16 bytes.
  int pad0;
  int pad1;
  int pad2;
};


struct LightSample // In world space coordinates.
{
  float3 direction;          // Direction from surface to light sampling position.
  float  distance;           // Distance between surface and light sample positon, RT_DEFAULT_MAX for environment light.
  float3 radiance_over_pdf;  // Radiance of this light sample divided by the pdf.
  float  pdf;                // Probability density for this light sample projected to solid angle. 1.0 when singular light.
};

#endif // LIGHT_DEFINITION_H
