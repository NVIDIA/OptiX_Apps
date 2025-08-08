//
// Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef PER_RAY_DATA_H
#define PER_RAY_DATA_H

#include "config.h"

#include <optix.h>

//#include <vector_types.h>

// The type of events created by BSDF importance sampling. (Taken from the MDL SDK.)
enum BsdfEventType
{
  BSDF_EVENT_ABSORB = 0,

  BSDF_EVENT_DIFFUSE      = 1,
  BSDF_EVENT_GLOSSY       = 1 << 1,
  BSDF_EVENT_SPECULAR     = 1 << 2,

  BSDF_EVENT_REFLECTION   = 1 << 3,
  BSDF_EVENT_TRANSMISSION = 1 << 4,

  BSDF_EVENT_DIFFUSE_REFLECTION    = BSDF_EVENT_DIFFUSE  | BSDF_EVENT_REFLECTION,
  BSDF_EVENT_DIFFUSE_TRANSMISSION  = BSDF_EVENT_DIFFUSE  | BSDF_EVENT_TRANSMISSION,
  BSDF_EVENT_GLOSSY_REFLECTION     = BSDF_EVENT_GLOSSY   | BSDF_EVENT_REFLECTION,
  BSDF_EVENT_GLOSSY_TRANSMISSION   = BSDF_EVENT_GLOSSY   | BSDF_EVENT_TRANSMISSION,
  BSDF_EVENT_SPECULAR_REFLECTION   = BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION,
  BSDF_EVENT_SPECULAR_TRANSMISSION = BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION,

  BSDF_EVENT_FORCE_32_BIT = 0xFFFFFFFFu
};


// Set when reaching a closesthit program.
#define FLAG_HIT           0x00000001
// Set when reaching the __anyhit__shadow program. Indicates visibility test failed.
#define FLAG_SHADOW        0x00000002
// Set if the material stack is not empty.
// This is implicit with the idxStack value.
//#define FLAG_VOLUME                 0x00000010 
// HACK Not implementing volume scattering.
//#define FLAG_VOLUME_SCATTERING      0x00000020
//#define FLAG_VOLUME_SCATTERING_MISS 0x00000040

// Small 4 entries deep material stack.
#define MATERIAL_STACK_LAST  3
#define MATERIAL_STACK_SIZE  4

// This is the minimal size of the struct. float4 for vectorized access was slower due to more registers used.
//struct MaterialStack
//{
//  float3 ior;     // index of refraction
//  float3 sigma_a; // absorption coefficient
//  float3 sigma_s; // scattering coefficient
//  float  bias;    // directional bias
//};

struct MaterialStack
{
  float4 absorption_ior;  // .xyz = absorption coefficient (sigma_a), .w = index of refraction.
  //float4 scattering_bias; // .xyz = scattering coefficient (sigma_s), .w = directional bias.
};


// Note that the fields are ordered by CUDA alignment restrictions.
struct PerRayData
{
  // 16-byte alignment
  // Small material stack tracking IOR, absorption ansd scattering coefficients of the entered materials. Entry 0 is vacuum.
  MaterialStack stack[MATERIAL_STACK_SIZE]; // FIXME No need for structure when there is only one float4 insde it.
  
  // 8-byte alignment
  
  // 4-byte alignment
  int idxStack;       // Top of material stack index.
  
  unsigned int seed;  // Random number generator input.
  
  float3 pos;         // Current surface hit point or volume sample point, in world space
  float  distance;    // Distance from the ray origin to the current position, in world space. Needed for absorption of nested materials.
  
  float3 wo;          // Outgoing direction, to observer, in world space.
  float3 wi;          // Incoming direction, to light, in world space.

  float3 radiance;    // Radiance along the current path segment.
  float  pdf;         // The last BSDF sample's pdf, tracked for multiple importance sampling.
  
  float3 throughput;  // The current path troughput. Starts white and gets modulated with bsdf_over_pdf with each sample.
  float  occlusion;    // The ambient occlusion value, only applies to diffuse and metal reflections and environment lights.
  
  unsigned int flags; // Bitfield with flags. See FLAG_* defines above for its contents.

  float3 sigma_t;     // Extinction coefficient (sigma_a + sigma_s) in a homogeneous medium.
  //int    walk;        // Number of random walk steps done through scattering volume.
  //float3 pdfVolume;   // Volume extinction sample pdf. Used to adjust the throughput along the random walk.

  int indexMaterial; // Result of a picking ray. 

  BsdfEventType typeEvent; // The type of events created by BSDF importance sampling.
};


// Alias the PerRayData pointer and an uint2 for the payload split and merge operations. This generates only move instructions.
// (FIXME Technically reading a different type from a union than was last written
// is undefined behavior in C++ but compilers usually do the necessary memory aliasing.)
typedef union
{
  PerRayData* ptr;
  uint2       dat;
} Payload;

__forceinline__ __device__ uint2 splitPointer(PerRayData* ptr)
{
  Payload payload;

  payload.ptr = ptr;

  return payload.dat;
}

__forceinline__ __device__ PerRayData* mergePointer(unsigned int p0, unsigned int p1)
{
  Payload payload;

  payload.dat.x = p0;
  payload.dat.y = p1;

  return payload.ptr;
}

#endif // PER_RAY_DATA_H
