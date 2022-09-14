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

#ifndef LIGHT_DEFINITION_H
#define LIGHT_DEFINITION_H

#include "function_indices.h"

struct LightDefinition
{
  // 16 byte alignment
  // These are used for rectangle lights and arbitrary mesh lights.
  float4 matrix[3];    // Object to world coordinates.
  float4 matrixInv[3]; // World to object coordinates.
  
  // 8 byte alignment
  CUdeviceptr attributes; // VertexAttribtues for triangles for mesh lights.
  CUdeviceptr indices;    // unsigned int triangle indices for mesh lights.

  // Importance sampling information for the emission texture.
  // Used by environment and rectangle lights when textured.
  CUdeviceptr cdfU; // float 2D, (width  + 1) * height elements.
  CUdeviceptr cdfV; // float 1D, (height + 1) elements.

  // Emisson texture. If not zero, scales emission.
  cudaTextureObject_t textureEmission;
  cudaTextureObject_t textureProfile; // The IES light profile as luminance float texture.

  // 4 byte alignment
  // These are only used in the spherical texture environment light and singular point, spot, and IES lights.
  // They are not used in the rectangle lights and arbitrary mesh lights.
  // Note that these will contain the identity matrix for mesh lights!
  float3 ori[3];    // Object to world orientation (rotational part of matrix).
  float3 oriInv[3]; // World to object orientation (rotational part of matrixInv).

  float3 emission; // Emission of the light. (The multiplier is applied on the host.)

  TypeLight typeLight;
  //TypeEDF  typeEDF; // FIXME Currently unused. All positional lights use a diffuse EDF.
  
  float area;     // The world space area of rectangle or arbitrary mesh lights. Unused for other lights.
  float integral; // The spherical environment map or rectangle light texture map integral over the whole texture.

  // Emission texture width and height. Used to index the CDFs, see above.
  // For mesh lights the width matches the number of triangles and the cdfU is over the triangle areas.
  unsigned int width; 
  unsigned int height;

  float spotAngleHalf; // Spot light cone half angle in radians, max. PI/2 (MaterialGUI has full angle in degrees.)
  float spotExponent;  // Affects the falloff from cone center to cone edge of the spot light.

  // Structure size padding to multiple of 16 bytes.
  //int pad0;
  //int pad1;
  //int pad2;
};


struct LightSample // In world space coordinates.
{
  float3 position;  // Position of the light sample when a geometric light.
  float  distance;  // Distance between surface and light sample positon, RT_DEFAULT_MAX for environment light.
  float3 direction; // Direction from surface to light sampling position.
  float3 emission;  // Emission of this light sample, scaled by the inverse probabilty to have picked one of many lights.
  float  pdf;       // Probability density for this light sample projected to solid angle.
};

#endif // LIGHT_DEFINITION_H
