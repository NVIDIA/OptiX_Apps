//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include "vector_math.h"

#ifndef __CUDACC_RTC__
#include <cassert>
#else
#define assert(x) /* nop */
#endif


// These define how many TEXCOORD, JOINTS, and WEIGHTS attributes are supported.
#define NUM_ATTR_TEXCOORDS 2
#define NUM_ATTR_JOINTS    2
#define NUM_ATTR_WEIGHTS   2

// These bit definitions for the flagAttributes imply that the above numbers of attributes are all 2.
// Note that the position attribute is the only one which is not optional!
// Note that ATTR_TEXCOORD_n == (ATTR_TEXCOORD_0 << n) is required for the morph target handling.
#define ATTR_INDEX      0x00000001u
#define ATTR_POSITION   0x00000010u
#define ATTR_TANGENT    0x00000020u
#define ATTR_NORMAL     0x00000040u
#define ATTR_COLOR_0    0x00000100u
#define ATTR_TEXCOORD_0 0x00001000u
#define ATTR_TEXCOORD_1 0x00002000u
#define ATTR_JOINTS_0   0x00010000u
#define ATTR_JOINTS_1   0x00020000u
#define ATTR_WEIGHTS_0  0x00100000u
#define ATTR_WEIGHTS_1  0x00200000u

// FIXME GeometryData currently supports only TriangleMesh geometries.
// Keep the structure to be able to add more geometric primitives of the glTF specs in the future.
// For example points could be shown as spheres and lines as linear curves. 
// That wouldn't be following the specified screen-space pixel sizes for those but could be interesting nonetheless.
struct GeometryData
{
  enum Type
  {
    TRIANGLE_MESH = 0,
    SPHERE_MESH   = 1, // implements glTF points
    UNKNOWN_TYPE  = 2
  };

  struct __align__(8) TriangleMesh
  {
    uint3*   indices;                       // INDICES
    float3*  positions;                     // POSITION (Required. All other attributes and the indices are optional!)
    float4*  colors;                        // COLOR_0, per vertex
    float4*  tangents;                      // TANGENTS
    float3*  normals;                       // NORMAL
    float2*  texcoords[NUM_ATTR_TEXCOORDS]; // TEXCOORD_0, TEXCOORD_1
    ushort4* joints[NUM_ATTR_JOINTS];       // JOINTS_0, JOINTS_1
    float4*  weights[NUM_ATTR_WEIGHTS];     // WEIGHTS_0, WEIGHTS_1

    // FIXME PERF Track which attributes are provided inside the above attribute pointers.
    // No need to load the attribute pointer when it's null.
    //unsigned int flagAttributes; // Currently unused.
    //unsigned int pad0;
  };

  // We render glTF Points as spheres, same radius for all the spheres.
  struct __align__(8) SphereMesh
  {
    float3*  positions;                     // POSITION (Required. All other attributes are optional!)
    float4*  colors;                        // COLOR_0, per sphere
    float3*  normals;                       // NORMAL, per sphere
  };

  //GeometryData()
  //{
  //}

  void setTriangleMesh(const TriangleMesh& t)
  {
    assert(type == UNKNOWN_TYPE);
    type = TRIANGLE_MESH;
    triangleMesh = t;
  }

  void setSphereMesh(const SphereMesh& t)
  {
    assert(type == UNKNOWN_TYPE);
    type = SPHERE_MESH;
    sphereMesh = t;
  }


#ifdef __CUDACC__
  template <typename MESH> const MESH& getMesh() const;
#endif

  Type type = UNKNOWN_TYPE;
  union
  {
    TriangleMesh triangleMesh;
    SphereMesh   sphereMesh;
  };
};
