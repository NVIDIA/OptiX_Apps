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

#ifndef SYSTEM_DATA_H
#define SYSTEM_DATA_H

#include "config.h"

#include "camera_definition.h"
#include "light_definition.h"
#include "material_definition.h"
#include "vertex_attributes.h"


struct SystemData
{
  // 16 byte alignment
  //int4 rect; // FIXME Unused, not implementing a tile renderer.

  // 8 byte alignment
  OptixTraversableHandle topObject;

  // The accumulated linear color space output buffer.
  // This is always sized to the resolution, not always matching the launch dimension.
  // Using a CUdeviceptr here to allow for different buffer formats without too many casts.
  CUdeviceptr         outputBuffer;
  // These buffers are used differently among the rendering strategies.
  CUdeviceptr         tileBuffer;
  CUdeviceptr         texelBuffer;

  CameraDefinition*   cameraDefinitions; // Currently only one camera in the array. (Allows camera motion blur in the future.)
  LightDefinition*    lightDefinitions;
  MaterialDefinition* materialDefinitions;

  int2 resolution;  // The actual rendering resolution. Independent from the launch dimensions for some rendering strategies.
  int2 tileSize;    // Example: make_int2(8, 4) for 8x4 tiles. Must be a power of two to make the division a right-shift.
  int2 tileShift;   // Example: make_int2(3, 2) for the integer division by tile size. That actually makes the tileSize redundant. 
  int2 pathLengths; // .x = min path length before Russian Roulette kicks in, .y = maximum path length

  // 4 byte alignment 
  int deviceCount;   // Number of devices doing the rendering.
  int deviceIndex;   // Device index to be able to distinguish the individual devices in a multi-GPU environment.
  int iterationIndex;
  int samplesSqrt;

  float sceneEpsilon;
  float clockScale; // Only used with USE_TIME_VIEW.

  int typeLens; // Camera type.

  int numCameras;
  int numMaterials;
  int numLights;
  
  int directLighting;
};


// SBT Record data for the hit group. This is used on the device to calculate attributes!
struct GeometryInstanceData
{
  // Using CUdeviceptr here to be able to handle different attribute and index formats for Triangles and Curves.
  CUdeviceptr attributes;
  CUdeviceptr indices;
  int         idMaterial;
  int         idLight; // Negative means not a light.
};

// Helper structure to optimize the lens shader direct callable arguments.
// Return this primary ray structure instead of using references to local memory.
struct LensRay
{
  float3 org;
  float3 dir;
};
#endif // SYSTEM_DATA_H
